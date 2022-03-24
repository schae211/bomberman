import os
import numpy as np
from .dnn_model import DoubleNNModel
from .cnn_model import DoubleCNNModel
from .cnn_plus_model import DoubleCNNPlusModel
from .pretrained_model import PretrainedModel
from .dnn_model_extended import DoubleNNModel_Extended
from .config import configs

# helper lists and dictionaries for actions
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
ACTION_TRANSLATE = {"UP": 0, "RIGHT": 1, "DOWN": 2, "LEFT": 3, "WAIT": 4, "BOMB": 5}
ACTION_TRANSLATE_REV = {val: key for key, val in ACTION_TRANSLATE.items()}


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    if configs.AGENT == "MLP": self.model = DoubleNNModel()
    if configs.AGENT == "CNN": self.model = DoubleCNNModel()
    if configs.AGENT == "CNNPlus": self.model = DoubleCNNPlusModel()
    if configs.AGENT == "MLPPlus": self.model = DoubleNNModel_Extended()
    if configs.PRETRAIN: self.pretrained_model = PretrainedModel()


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """

    # only do exploration if we train, no random moves in tournament + reduce exploration in later episodes/games
    episode_n = 0 if game_state is None else game_state["round"]

    # configure pretrain
    if configs.PRETRAIN and episode_n <= configs.PRETRAIN_LEN:
        if np.random.rand() <= configs.PRETRAIN_RANDOM * configs.PRETRAIN_RANDOM_DECAY ** episode_n:
            self.logger.debug("Pretrained agent is playing random")
            return np.random.choice(ACTIONS, p=configs.DEFAULT_PROBS)
        else:
            self.logger.debug("Pretrained agent is playing deterministic")
            features = state_to_features_pretrain(game_state)
            q_values = self.pretrained_model.predict(features).reshape(-1)
            if configs.POLICY == "deterministic":
                return ACTION_TRANSLATE_REV[np.argmax(q_values)]
            elif configs.POLICY == "stochastic":
                return np.random.choice(ACTIONS, p=(np.exp(q_values) / np.sum(np.exp(q_values))))

    # reduce exploration over time, account for pretraining ("episode_n-configs.PRETRAIN_LEN")
    if self.train:
        if configs.EPSILON_DECAY_LINEAR:
            current_epsilon = max(self.epsilon - self.eps_slope*(episode_n-configs.PRETRAIN_LEN), self.epsilon_min)
        else:
            current_epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_reduction ** (episode_n-configs.PRETRAIN_LEN))
        if np.random.rand() <= current_epsilon:
            self.logger.debug("Choosing action at random due to epsilon-greedy policy")
            action = np.random.choice(ACTIONS, p=configs.DEFAULT_PROBS)
            return action
        else:
            self.logger.debug("Querying fitted model for action.")
            features = state_to_features(game_state)
            q_values = self.model.predict_policy(features).reshape(-1)  # computing q-values using our fitted model

            if configs.POLICY == "deterministic":
                action = ACTION_TRANSLATE_REV[np.argmax(q_values)]
                return action

            elif configs.POLICY == "stochastic":
                # use softmax to translate q-values to probabilities,
                probs = np.exp(q_values) / np.sum(np.exp(q_values))
                return np.random.choice(ACTIONS, p=probs)

    else:
        self.logger.debug("Querying fitted model for action.")
        features = state_to_features(game_state)
        q_values = self.model.predict_policy(features).reshape(-1)  # computing q-values using our fitted model

        if configs.POLICY == "deterministic":
            action = ACTION_TRANSLATE_REV[np.argmax(q_values)]
            return action

        elif configs.POLICY == "stochastic":
            # use softmax to translate q-values to probabilities,
            probs = np.exp(q_values) / np.sum(np.exp(q_values))
            return np.random.choice(ACTIONS, p=probs)


def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the game_state dict before the game begins and after it ends, so features cannot be extracted
    if game_state is None:
        return None

    if configs.FEATURE_ENGINEERING == "channels":

        object_map = game_state["field"]

        coin_map = np.zeros_like(game_state["field"])
        for cx, cy in game_state["coins"]: coin_map[cx, cy] = 1

        self_map = np.zeros_like(game_state["field"])
        self_map[game_state["self"][3]] = 1

        explosion_map = get_bomb_map(object_position=game_state["field"], bomb_list=game_state["bombs"],
                                     explosion_position=game_state["explosion_map"])

        other_agents = np.zeros_like(game_state["field"])
        for (_, _, _, (cx, cy)) in game_state["others"]: other_agents[cx, cy] = 1

        # create channels based on the field and coin information.
        channels = [object_map, self_map, coin_map, explosion_map, other_agents]

        # concatenate them as a feature tensor
        stacked_channels = np.stack(channels)

        # and return them as a vector
        return stacked_channels[None,:]

    if configs.FEATURE_ENGINEERING == "standard":

        # 1. 1D array, len = 4: indicating in which directions the agent can move (up, right, down, left)
        awareness = get_awareness(object_position=game_state["field"], self_position=game_state["self"][3])

        # 2. 1D array, len = 4:  indicating in which direction lays the closest coin determined by BFS (up, right, down, left)
        coin_direction = get_coin_direction(object_position=game_state["field"], coin_list=game_state["coins"],
                                            self_position=game_state["self"][3])

        # 2D array indicating which area is affected by exploded bombs or by bombs which are about to explode
        explosion_map = get_bomb_map(object_position=game_state["field"], bomb_list=game_state["bombs"],
                                     explosion_position=game_state["explosion_map"])

        # 3. 1D array, len = 5:  indicating how dangerous the current field, up, right, down, left are
        danger = get_danger(explosion_map=explosion_map, self_position=game_state["self"][3])

        # 4. 1D array, len = 5:  indicating in which direction to flee from bomb if immediate danger (current, up, right, down, left),
        # if no immediate danger/current spot is safe returns all zeros, otherwise direction
        safe_direction = get_safe_direction(object_position=game_state["field"], explosion_map=explosion_map,
                                            self_position=game_state["self"][3])

        # 5. 1D array, len = 5:  indicating in which direction lays the most lucrative position for laying a bomb
        # that destroys most crates penalized by the distance as determined by BFS (up, right, down, left)
        crate_direction = get_crate_direction(object_position=game_state["field"], bomb_list=game_state["bombs"],
                                              self_position=game_state["self"][3], explosion_map=explosion_map)

        # 6. 1D array, len = 2: indicating whether bomb can be dropped and survival is possible
        bomb_info = get_bomb_info(object_position=game_state["field"], explosion_map=explosion_map,
                                  self=game_state["self"], bomb_list=game_state["bombs"])

        others_direction = get_others_direction(object_position=game_state["field"], bomb_list=game_state["bombs"],
                                                self_position=game_state["self"][3], explosion_map=explosion_map,
                                                others=game_state["others"])

        features = np.concatenate([awareness,
                                   danger,
                                   safe_direction,
                                   coin_direction,
                                   crate_direction,
                                   bomb_info,
                                   others_direction])

        return features[None,:]


    if configs.FEATURE_ENGINEERING == "channels+bomb":

        object_map = game_state["field"]

        coin_map = np.zeros_like(game_state["field"])
        for cx, cy in game_state["coins"]: coin_map[cx, cy] = 1

        self_map = np.zeros_like(game_state["field"])
        self_map[game_state["self"][3]] = 1

        explosion_map = get_bomb_map(object_position=game_state["field"], bomb_list=game_state["bombs"],
                                     explosion_position=game_state["explosion_map"])

        other_agents = np.zeros_like(game_state["field"])
        for (_, _, _, (cx, cy)) in game_state["others"]: other_agents[cx, cy] = 1

        # channel for additional features
        additional_info = np.zeros_like(game_state["field"])
        additional_info[0,0] = game_state["self"][2]    # add bomb info

        # create channels based on the field and coin information.
        channels = [object_map, self_map, coin_map, explosion_map, other_agents, additional_info]

        # concatenate them as a feature tensor
        stacked_channels = np.stack(channels)

        # return the channels and whether bomb action is possible
        return stacked_channels[None,:]

    if configs.FEATURE_ENGINEERING == "standard_extended":

        object_map = game_state["field"]

        coin_map = np.zeros_like(game_state["field"])
        for cx, cy in game_state["coins"]: coin_map[cx, cy] = 1

        explosion_map = get_bomb_map(object_position=game_state["field"], bomb_list=game_state["bombs"],
                                     explosion_position=game_state["explosion_map"])

        other_agents = np.zeros_like(game_state["field"])
        for (_, _, _, (cx, cy)) in game_state["others"]: other_agents[cx, cy] = 1

        # now extract the fields surrounding our agent
        self_position = np.array(game_state["self"][3])

        new_features = np.zeros((4, 24))
        # set per default everything to walls
        new_features[0,:] = -1
        # go 3 steps into each direction (via try and except?)
        x_indices = [-3, -2, -1, 0, 1, 2, 3]
        count = 0
        for x_i, j in enumerate([0,1,2,3,2,1,0]):
            for t in range(-j, j+1):
                #print(x_indices[x_i], t)
                test_pos = tuple(self_position + np.array([x_indices[x_i], t]))
                if test_pos == (0,0):
                    continue
                try:
                    new_features[0, count] = object_map[test_pos]
                    new_features[1, count] = coin_map[test_pos]
                    new_features[2, count] = explosion_map[test_pos]
                    new_features[3, count] = other_agents[test_pos]
                except IndexError:
                    pass
                count += 1

        # flatten new features and concatenate to other vector
        new_features = new_features.reshape(-1)

        # 2. 1D array, len = 4:  indicating in which direction lays the closest coin determined by BFS (up, right, down, left)
        coin_direction = get_coin_direction(object_position=game_state["field"], coin_list=game_state["coins"],
                                            self_position=game_state["self"][3])

        # 2D array indicating which area is affected by exploded bombs or by bombs which are about to explode
        explosion_map = get_bomb_map(object_position=game_state["field"], bomb_list=game_state["bombs"],
                                     explosion_position=game_state["explosion_map"])

        # 4. 1D array, len = 5:  indicating in which direction to flee from bomb if immediate danger (current, up, right, down, left),
        # if no immediate danger/current spot is safe returns all zeros, otherwise direction
        safe_direction = get_safe_direction(object_position=game_state["field"], explosion_map=explosion_map,
                                            self_position=game_state["self"][3])

        # 5. 1D array, len = 5:  indicating in which direction lays the most lucrative position for laying a bomb
        # that destroys most crates penalized by the distance as determined by BFS (up, right, down, left)
        crate_direction = get_crate_direction(object_position=game_state["field"], bomb_list=game_state["bombs"],
                                              self_position=game_state["self"][3], explosion_map=explosion_map)

        # 6. 1D array, len = 2: indicating whether bomb can be dropped and survival is possible
        bomb_info = get_bomb_info(object_position=game_state["field"], explosion_map=explosion_map,
                                  self=game_state["self"], bomb_list=game_state["bombs"])

        #others_direction = get_others_direction(object_position=game_state["field"], bomb_list=game_state["bombs"],
        #                                        self_position=game_state["self"][3], explosion_map=explosion_map,
        #                                        others=game_state["others"])

        features = np.concatenate([safe_direction,
                                   coin_direction,
                                   crate_direction,
                                   bomb_info,
                                   new_features])

        return features[None,:]


def state_to_features_pretrain(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the game_state dict before the game begins and after it ends, so features cannot be extracted
    if game_state is None:
        return None

    if configs.PRETRAIN_FEATURES == "channels":

        object_map = game_state["field"]

        coin_map = np.zeros_like(game_state["field"])
        for cx, cy in game_state["coins"]: coin_map[cx, cy] = 1

        self_map = np.zeros_like(game_state["field"])
        self_map[game_state["self"][3]] = 1

        explosion_map = get_bomb_map(object_position=game_state["field"], bomb_list=game_state["bombs"],
                                     explosion_position=game_state["explosion_map"])

        other_agents = np.zeros_like(game_state["field"])
        for (_, _, _, (cx, cy)) in game_state["others"]: other_agents[cx, cy] = 1

        # create channels based on the field and coin information.
        channels = [object_map, self_map, coin_map, explosion_map, other_agents]

        # concatenate them as a feature tensor
        stacked_channels = np.stack(channels)

        # and return them as a vector
        return stacked_channels[None,:]

    if configs.PRETRAIN_FEATURES == "standard":

        # 1. 1D array, len = 4: indicating in which directions the agent can move (up, right, down, left)
        awareness = get_awareness(object_position=game_state["field"], self_position=game_state["self"][3])

        # 2. 1D array, len = 4:  indicating in which direction lays the closest coin determined by BFS (up, right, down, left)
        coin_direction = get_coin_direction(object_position=game_state["field"], coin_list=game_state["coins"],
                                            self_position=game_state["self"][3])

        # 2D array indicating which area is affected by exploded bombs or by bombs which are about to explode
        explosion_map = get_bomb_map(object_position=game_state["field"], bomb_list=game_state["bombs"],
                                     explosion_position=game_state["explosion_map"])

        # 3. 1D array, len = 5:  indicating how dangerous the current field, up, right, down, left are
        danger = get_danger(explosion_map=explosion_map, self_position=game_state["self"][3])

        # 4. 1D array, len = 5:  indicating in which direction to flee from bomb if immediate danger (current, up, right, down, left),
        # if no immediate danger/current spot is safe returns all zeros, otherwise direction
        safe_direction = get_safe_direction(object_position=game_state["field"], explosion_map=explosion_map,
                                            self_position=game_state["self"][3])

        # 5. 1D array, len = 5:  indicating in which direction lays the most lucrative position for laying a bomb
        # that destroys most crates penalized by the distance as determined by BFS (up, right, down, left)
        crate_direction = get_crate_direction(object_position=game_state["field"], bomb_list=game_state["bombs"],
                                              self_position=game_state["self"][3], explosion_map=explosion_map)

        # 6. 1D array, len = 2: indicating whether bomb can be dropped and survival is possible
        bomb_info = get_bomb_info(object_position=game_state["field"], explosion_map=explosion_map,
                                  self=game_state["self"], bomb_list=game_state["bombs"])

        others_direction = get_others_direction(object_position=game_state["field"], bomb_list=game_state["bombs"],
                                                self_position=game_state["self"][3], explosion_map=explosion_map,
                                                others=game_state["others"])

        features = np.concatenate([awareness,
                                   danger,
                                   safe_direction,
                                   coin_direction,
                                   crate_direction,
                                   bomb_info,
                                   others_direction])

        return features[None,:]


    if configs.FEATURE_ENGINEERING == "channels+bomb":

        object_map = game_state["field"]

        coin_map = np.zeros_like(game_state["field"])
        for cx, cy in game_state["coins"]: coin_map[cx, cy] = 1

        self_map = np.zeros_like(game_state["field"])
        self_map[game_state["self"][3]] = 1

        explosion_map = get_bomb_map(object_position=game_state["field"], bomb_list=game_state["bombs"],
                                     explosion_position=game_state["explosion_map"])

        other_agents = np.zeros_like(game_state["field"])
        for (_, _, _, (cx, cy)) in game_state["others"]: other_agents[cx, cy] = 1

        # channel for additional features
        additional_info = np.zeros_like(game_state["field"])
        additional_info[0,0] = game_state["self"][2]    # add bomb info

        # create channels based on the field and coin information.
        channels = [object_map, self_map, coin_map, explosion_map, other_agents, additional_info]

        # concatenate them as a feature tensor
        stacked_channels = np.stack(channels)

        # return the channels and whether bomb action is possible
        return stacked_channels[None,:]


# define simple node class used for BFS
class Node(object):
    def __init__(self, position, parent_position, move, steps=None):
        self.position = position                    # child node position
        self.parent_position = parent_position      # parent node position
        self.move = move                            # how to get from parent to child node
        self.steps = steps                          # number of steps from starting node (not always used)


# define simple queue class that also allows checking for states (which is an attribute of the node)
class Queue(object):
    def __init__(self):
        self.fifo = []  # first in first out

    def put(self, node):
        self.fifo.append(node)

    def contains_state(self, state):
        return any(node.position == state for node in self.fifo)

    def empty(self):
        return len(self.fifo) == 0

    def get(self):
        if self.empty():
            raise Exception("empty queue")
        else:
            node = self.fifo.pop(0)
            return node


def get_neighbors(object_position, position):
    neighbor_info = {"actions": [], "neighbors": []}
    if object_position[position[0] - 1, position[1]] == 0:
        neighbor_info["actions"].append("up")
        neighbor_info["neighbors"].append((position[0] - 1, position[1]))
    if object_position[position[0], position[1] + 1] == 0:
        neighbor_info["actions"].append("right")
        neighbor_info["neighbors"].append((position[0], position[1] + 1))
    if object_position[position[0] + 1, position[1]] == 0:
        neighbor_info["actions"].append("down")
        neighbor_info["neighbors"].append((position[0] + 1, position[1]))
    if object_position[position[0], position[1] - 1] == 0:
        neighbor_info["actions"].append("left")
        neighbor_info["neighbors"].append((position[0], position[1] - 1))
    return neighbor_info


def coin_bfs(object_position, coin_position, self_position):
    """
    Find path to nearest coin via breadth-first search (BFS)
    :param object_position:
    :param coin_position:
    :param self_position:
    :return:
    """
    q = Queue()
    explored = set()
    # add start to the Queue
    q.put(Node(position=self_position, parent_position=None, move=None))
    # loop over the queue as long as it is not empty
    while True:
        # if we cannot reach any revealed coin
        if q.empty(): return None
        # always get first element
        node = q.get()
        # found a coin, trace back to parent, but not if coin is where initial position is
        if node.parent_position is not None and node.position in coin_position:
            actions, cells = [], []
            # Backtracking: From each node grab state and action; and then redefine node as parent node
            while node.parent_position is not None:
                actions.append(node.move)
                cells.append(node.position)
                node = node.parent_position
            actions.reverse()
            cells.reverse()
            return actions, cells
        explored.add(node.position)
        # Add neighbors to fifo
        neighbors = get_neighbors(object_position, node.position)
        for action, neighbor in zip(neighbors["actions"], neighbors["neighbors"]):
            if neighbor not in explored and not q.contains_state(neighbor):
                child = Node(position=neighbor, parent_position=node, move=action)
                q.put(child)


def save_bfs(object_position, explosion_map, self_position):
    """
    Find path to nearest save position via breadth-first search (BFS)
    :param object_position:
    :param explosion_map:
    :param self_position
    :return:
    """
    q = Queue()
    explored = set()
    # add start to the Queue
    q.put(Node(position=self_position, parent_position=None, move=None, steps=0))
    # loop over the queue as long as it is not empty
    while True:
        if q.empty():
            return "dead"
        # always get first element
        node = q.get()
        # found a save position, either not danger or danger is already gone
        if explosion_map[node.position] == 0 or (explosion_map[node.position] + (node.steps * 1/5)) > 1:
            moves = []
            cells = []
            # Backtracking: From each node grab state and action; and then redefine node as parent node
            while node.parent_position is not None:
                moves.append(node.move)
                cells.append(node.position)
                node = node.parent_position
            moves.reverse()
            cells.reverse()
            return moves, cells
        explored.add(node.position)
        # Add neighbors to fifo
        neighbors = get_neighbors(object_position, node.position)
        for action, neighbor in zip(neighbors["actions"], neighbors["neighbors"]):
            # make sure that we do not die on the way to the potential neighbor (states 4/5 and 5/5 are deadly!)
            if neighbor not in explored and not q.contains_state(neighbor) and not 4/5 <= (explosion_map[neighbor] + ((node.steps+1) * 1/5)) <= 1:
                # TODO: Check whether steps is really always + 1, but should make sense in BFS
                child = Node(position=neighbor, parent_position=node, move=action, steps=node.steps+1)
                q.put(child)


# default dist can be reduced to make computations faster
def crate_bfs(object_position, self_position, bomb_list, explosion_map, max_dist=16, distance_discount=0.8):
    """
    Find path to position where bomb destroys most crates via breadth-first search (BFS)
    Thereby, we have to take the distance to the considered positions into consideration (trade-off!)
    :param bomb_list:
    :param distance_discount:
    :param explosion_map:
    :param max_dist:
    :param object_position:
    :param self_position:
    :return:
    """
    q = Queue()
    explored = set()
    # initialize maximum yet
    top_considered_node = None
    top_score = -np.inf
    # add start to the Queue
    q.put(Node(position=self_position, parent_position=None, move=None))
    # loop over the queue as long as it is not empty
    while True:
        # when empty we will trace the path to our top node
        if q.empty():
            # basically no target was found
            if top_considered_node is None:
                return None
            else:
                node = top_considered_node
                moves, cells = [], []
                while node.parent_position is not None:
                    moves.append(node.move)
                    cells.append(node.position)
                    node = node.parent_position
                moves.reverse()
                cells.reverse()
                return moves, cells
        # always get first element
        node = q.get()
        # compute distance to current position
        dist_to_self = np.abs(np.array(node.position) - np.array(self_position)).sum()
        # only consider if escaping is possible at the current position
        if check_survival(object_position=object_position, bomb_list=bomb_list,
                          position=node.position, explosion_map=explosion_map):
            # compute number of destroyed crates
            destroyed_crates = get_destroyed_crates(object_position, node.position)
            # TODO: Think about how to compute this destruction score, taking 1/4 because bomb takes 4 steps to explode
            # combine distance and destroyed crates into a score
            destruction_score = destroyed_crates - distance_discount * dist_to_self
            # found a better position according to our destruction score
            if destruction_score > top_score and destroyed_crates >= 1:
                top_considered_node = node
                top_score = destruction_score
        explored.add(node.position)
        # if too far away do not consider adding the neighbors!
        if dist_to_self <= max_dist:
            neighbors = get_neighbors(object_position, node.position)
            for action, neighbor in zip(neighbors["actions"], neighbors["neighbors"]):
                if neighbor not in explored and not q.contains_state(neighbor):
                    child = Node(position=neighbor, parent_position=node, move=action)
                    q.put(child)


def get_destroyed_crates(object_position, bomb_position):
    """
    Compute the number of destroyed crates for a given bomb position
    :param object_position: numpy 2D array: game_state["field"]
    :param bomb_position: coordinate tuple (x,y)
    :return: int: number of destroyed crates
    """
    destroyed_crates = 0
    # check above
    for up_x in range(bomb_position[0] - 1, bomb_position[0] - 4, -1):
        if 0 <= up_x <= 16:
            if object_position[up_x, bomb_position[1]] == -1:
                break
            else:
                # if a crate is present at the position add to destroyed crates counter
                if object_position[up_x, bomb_position[1]] == 1:
                    destroyed_crates += 1
    # check below
    for down_x in range(bomb_position[0] + 1, bomb_position[0] + 4, 1):
        if 0 <= down_x <= 16:
            if object_position[down_x, bomb_position[1]] == -1:
                break
            else:
                if object_position[down_x, bomb_position[1]] == 1:
                    destroyed_crates += 1
    # check to the left
    for left_y in range(bomb_position[1] - 1, bomb_position[1] - 4, -1):
        if 0 <= left_y <= 16:
            if object_position[bomb_position[0], left_y] == -1:
                break
            else:
                if object_position[bomb_position[0], left_y] == 1:
                    destroyed_crates += 1
    # check to the right
    for right_y in range(bomb_position[1] + 1, bomb_position[1] + 4, 1):
        if 0 <= right_y <= 16:
            if object_position[bomb_position[0], right_y] == -1:
                break
            else:
                if object_position[bomb_position[0], right_y] == 1:
                    destroyed_crates += 1

    return destroyed_crates


def check_survival(object_position, bomb_list, position, explosion_map):
    bomb_list_tmp = bomb_list.copy()
    bomb_list_tmp.append(((position), 3))
    # add information about whether dropping a bomb is suicide
    updated_explosion_map = get_bomb_map(object_position=object_position,
                                         bomb_list=bomb_list_tmp,
                                         explosion_position=explosion_map)
    check = save_bfs(object_position=object_position, explosion_map=updated_explosion_map,
                     self_position=position)
    if check != "dead":
        return True
    else:
        return False


def get_awareness(object_position, self_position):
    return (np.array([
        object_position[self_position[0] - 1, self_position[1]],  # up
        object_position[self_position[0], self_position[1] + 1],  # right
        object_position[self_position[0] + 1, self_position[1]],  # down
        object_position[self_position[0], self_position[1] - 1]   # left
    ]) == 0).astype(float)


def get_coin_direction(object_position, coin_list, self_position):
    # if no coins revealed return all 0
    if len(coin_list) == 0:
        return np.zeros(4)

    # perform BFS to find the nearest coin
    coin_info = coin_bfs(object_position=object_position, coin_position=coin_list,
                         self_position=self_position)

    coin_direction = np.zeros(4)
    if coin_info is not None:  # can we even reach the revealed coins? due to walls/crates it may be impossible
        if coin_info[0][0] == "up":
            coin_direction[0] = 1
        elif coin_info[0][0] == "right":
            coin_direction[1] = 1
        elif coin_info[0][0] == "down":
            coin_direction[2] = 1
        elif coin_info[0][0] == "left":
            coin_direction[3] = 1
    return coin_direction


def get_bomb_map(object_position, bomb_list, explosion_position):
    # get information about affected areas meaning where bombs are about to explode and where it is still dangerous
    explosion_map = explosion_position.copy().astype(float)
    ctd_to_score = lambda ctd: 4/5 - (1/5 * ctd)
    if bomb_list is None:
        return explosion_map
    else:
        bombs = bomb_list.copy()
        for bomb_pos, bomb_ctd in bombs:
            # position of the bomb itself
            explosion_map[bomb_pos] = max(explosion_map[bomb_pos], ctd_to_score(bomb_ctd))  # max ensure not to overwrite the game state explosion map
            # check above
            for up_x in range(bomb_pos[0] - 1, bomb_pos[0] - 4, -1):
                if 0 <= up_x <= 16:
                    if object_position[up_x, bomb_pos[1]] == -1:
                        break
                    else:
                        explosion_map[up_x, bomb_pos[1]] = max(explosion_map[up_x, bomb_pos[1]], ctd_to_score(bomb_ctd))
            # check below
            for down_x in range(bomb_pos[0] + 1, bomb_pos[0] + 4, 1):
                if 0 <= down_x <= 16:
                    if object_position[down_x, bomb_pos[1]] == -1:
                        break
                    else:
                        explosion_map[down_x, bomb_pos[1]] = max(explosion_map[down_x, bomb_pos[1]], ctd_to_score(bomb_ctd))
            # check to the left
            for left_y in range(bomb_pos[1] - 1, bomb_pos[1] - 4, -1):
                if 0 <= left_y <= 16:
                    if object_position[bomb_pos[0], left_y] == -1:
                        break
                    else:
                        explosion_map[bomb_pos[0], left_y] = max(explosion_map[bomb_pos[0], left_y], ctd_to_score(bomb_ctd))
            # check to the right
            for right_y in range(bomb_pos[1] + 1, bomb_pos[1] + 4, 1):
                if 0 <= right_y <= 16:
                    if object_position[bomb_pos[0], right_y] == -1:
                        break
                    else:
                        explosion_map[bomb_pos[0], right_y] = max(explosion_map[bomb_pos[0], right_y], ctd_to_score(bomb_ctd))
        return explosion_map


def get_danger(explosion_map, self_position):
    return np.array([explosion_map[self_position[0], self_position[1]],         # current position
                     explosion_map[self_position[0] - 1, self_position[1]],     # up
                     explosion_map[self_position[0], self_position[1] + 1],     # right
                     explosion_map[self_position[0] + 1, self_position[1]],     # down
                     explosion_map[self_position[0], self_position[1] - 1]])    # left


def get_safe_direction(object_position, explosion_map, self_position):
    if explosion_map[self_position[0], self_position[1]] == 0:
        return np.zeros(5)

    safe_direction = np.zeros(5)
    explosion_info = save_bfs(object_position=object_position, explosion_map=explosion_map,
                              self_position=self_position)
    if explosion_info == "dead":            # no safe position at all detected, sure death
        safe_direction[0] = 1
    #elif explosion_info == ([], []):       # should not happen since we check above for immediate danger
    #    pass
    elif explosion_info[0][0] == "up":      # up is safe
        safe_direction[1] = 1
    elif explosion_info[0][0] == "right":   # right is safe
        safe_direction[2] = 1
    elif explosion_info[0][0] == "down":    # down is safe
        safe_direction[3] = 1
    elif explosion_info[0][0] == "left":    # left is safe
        safe_direction[4] = 1

    return safe_direction


def get_crate_direction(object_position, bomb_list, self_position, explosion_map):
    crate_direction = np.zeros(5)
    crate_info = crate_bfs(object_position=object_position, self_position=self_position,
                           bomb_list=bomb_list, explosion_map=explosion_map)

    if crate_info is not None:  # None is returned if destruction score was negative for all considered positions
        crate_direction[0] = 1              # indicating that a target was found
        if crate_info == ([], []):          # if current position, keep all directional indicators 0
            pass
        elif crate_info[0][0] == "up":
            crate_direction[1] = 1
        elif crate_info[0][0] == "right":
            crate_direction[2] = 1
        elif crate_info[0][0] == "down":
            crate_direction[3] = 1
        elif crate_info[0][0] == "left":
            crate_direction[4] = 1

    return crate_direction


def get_bomb_info(object_position, explosion_map, bomb_list, self):
    """
    Compute vector of length 2 where the positions indicate:
    0: 1 if bomb action possible else 0
    1: 1 if bomb action does NOT lead to suicide else 0
    :param object_position:
    :param explosion_map:
    :param bomb_list:
    :param self:
    :return:
    """
    bomb_info = np.zeros(2)
    # check if bomb action is possible
    if self[2]:
        bomb_info[0] = 1
    if check_survival(object_position=object_position, explosion_map=explosion_map,
                      bomb_list=bomb_list, position=self[3]):
        bomb_info[1] = 1
    return bomb_info


def get_others_direction(object_position, bomb_list, self_position, explosion_map, others):
    others_position = np.zeros_like(object_position)
    for (_, _, _, (cx, cy)) in others: others_position[cx, cy] += 1
    others_direction = np.zeros(5)
    others_info = other_bfs(object_position, bomb_list, self_position, explosion_map, others_position, max_dist=16)

    if others_info is not None:  # None is returned if destruction score was negative for all considered positions
        others_direction[0] = 1              # indicating that a target was found
        if others_info == ([], []):          # if current position, keep all directional indicators 0
            pass
        elif others_info[0][0] == "up":
            others_direction[1] = 1
        elif others_info[0][0] == "right":
            others_direction[2] = 1
        elif others_info[0][0] == "down":
            others_direction[3] = 1
        elif others_info[0][0] == "left":
            others_direction[4] = 1

    return others_direction


def other_bfs(object_position, bomb_list, self_position, explosion_map, others_position, max_dist=16):
    q = Queue()
    explored = set()
    # add start to the Queue
    q.put(Node(position=self_position, parent_position=None, move=None))
    # loop over the queue as long as it is not empty
    while True:
        # if we cannot reach any revealed coin
        if q.empty(): return None
        # always get first element
        node = q.get()
        # found a position to kill enemies and laying a bomb there is no suicide
        if get_destroyed_agents(object_position, node.position, others_position) >= 1 and \
                check_survival(object_position, bomb_list, node.position, explosion_map):
            actions, cells = [], []
            # Backtracking: From each node grab state and action; and then redefine node as parent node
            while node.parent_position is not None:
                actions.append(node.move)
                cells.append(node.position)
                node = node.parent_position
            actions.reverse()
            cells.reverse()
            return actions, cells
        explored.add(node.position)
        # only add neighbors that are closer than the maximal distance
        dist_to_self = np.abs(np.array(node.position) - np.array(self_position)).sum()
        if dist_to_self <= max_dist:
            neighbors = get_neighbors(object_position, node.position)
            for action, neighbor in zip(neighbors["actions"], neighbors["neighbors"]):
                if neighbor not in explored and not q.contains_state(neighbor):
                    child = Node(position=neighbor, parent_position=node, move=action)
                    q.put(child)


def get_destroyed_agents(object_position, bomb_position, other_positions):
    """
    Compute the number of destroyed crates for a given bomb position
    :param object_position: numpy 2D array: game_state["field"]
    :param bomb_position: coordinate tuple (x,y)
    :return: int: number of destroyed crates
    """
    destroyed_others = 0
    # check above
    for up_x in range(bomb_position[0] - 1, bomb_position[0] - 4, -1):
        if 0 <= up_x <= 16:
            if object_position[up_x, bomb_position[1]] == -1:
                break
            else:
                # if a crate is present at the position add to destroyed crates counter
                if other_positions[up_x, bomb_position[1]] == 1:
                    destroyed_others += 1
    # check below
    for down_x in range(bomb_position[0] + 1, bomb_position[0] + 4, 1):
        if 0 <= down_x <= 16:
            if object_position[down_x, bomb_position[1]] == -1:
                break
            else:
                if other_positions[down_x, bomb_position[1]] == 1:
                    destroyed_others += 1
    # check to the left
    for left_y in range(bomb_position[1] - 1, bomb_position[1] - 4, -1):
        if 0 <= left_y <= 16:
            if object_position[bomb_position[0], left_y] == -1:
                break
            else:
                if other_positions[bomb_position[0], left_y] == 1:
                    destroyed_others += 1
    # check to the right
    for right_y in range(bomb_position[1] + 1, bomb_position[1] + 4, 1):
        if 0 <= right_y <= 16:
            if object_position[bomb_position[0], right_y] == -1:
                break
            else:
                if other_positions[bomb_position[0], right_y] == 1:
                    destroyed_others += 1
    return destroyed_others