import os
import pickle
import numpy as np
# from agent_code.nn_agent_v1.nn_model import NNModel
from agent_code.nn_agent_v1.dnn_model import DoubleNNModel
from agent_code.nn_agent_v1.config import configs

# helper lists and dictionaries for actions
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
ACTION_TRANSLATE = {"UP": 0, "RIGHT": 1, "DOWN": 2, "LEFT": 3, "WAIT": 4, "BOMB": 5}
ACTION_TRANSLATE_REV = {val: key for key, val in ACTION_TRANSLATE.items()}

# importing and defining parameters
DEFAULT_PROBS = configs["DEFAULT_PROBS"]
POLICY = configs["POLICY"]
FEAT_ENG = configs["FEATURE_ENGINEERING"]


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
    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        #self.model = NNModel()
        self.model = DoubleNNModel()
    else:
        self.logger.info("Loading model from saved state.")
        #self.model = NNModel()
        self.model = DoubleNNModel()


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """

    # only do exploration if we train, no random moves in tournament
    # reduce exploration in later episodes/games
    episode_n = 0 if game_state is None else game_state["round"]
    if self.train and np.random.rand() <= max(self.epsilon_min, self.epsilon * self.epsilon_reduction ** episode_n):
        self.logger.debug("Choosing action at random due to epsilon-greedy policy")
        return np.random.choice(ACTIONS, p=DEFAULT_PROBS)
    else:
        self.logger.debug("Querying fitted model for action.")

        # array.reshape(1, -1) if it contains a single sample for MultiOutputRegressor
        features = state_to_features(game_state).reshape(1, -1)

        # compute q-values using our fitted model, important to flatten the output again
        q_values = self.model.predict(features).reshape(-1)

        if POLICY == "deterministic":
            return ACTION_TRANSLATE_REV[np.argmax(q_values)]

        elif POLICY == "stochastic":
            # normalize the q_values, take care not to divide by zero (fall back to default probs)
            if any(q_values != 0):
                probs = np.exp(q_values) / np.sum(np.exp(q_values))
            else:
                self.logger.debug("Choosing action at random because q-values are all 0")
                probs = DEFAULT_PROBS

            # using a stochastic policy!
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
    # This is the game_state dict before the game begins and after it ends
    if game_state is None:
        return None

    if FEAT_ENG == "channels":

        object_map = game_state["field"]

        # for the coin challenge we need to know where the agent is, where walls are and where the coins are
        # so, we create a coin map in the same shape as the field
        coin_map = np.zeros_like(game_state["field"])
        for cx, cy in game_state["coins"]:
            coin_map[cx, cy] = 1

        # also adding where one self is on the map
        self_map = np.zeros_like(game_state["field"])
        self_map[game_state["self"][3]] = 1

        explosion_map = get_bomb_map(object_position=game_state["field"], bomb_list=game_state["bombs"],
                                     explosion_position=game_state["explosion_map"])

        # create channels based on the field and coin information.
        channels = [object_map, self_map, coin_map, explosion_map]

        # concatenate them as a feature tensor (they must have the same shape), ...
        stacked_channels = np.stack(channels)
        # and return them as a vector
        return stacked_channels.reshape(-1)

    elif FEAT_ENG == "standard":

        # Situational awareness, indicating in which directions the agent can move
        # Add is bomb action possible?
        awareness = get_awareness(object_position=game_state["field"], self_position=game_state["self"][3])

        # Direction to the closest coin determined by BFS
        coin_direction = get_coin_direction(object_position=game_state["field"], coin_list=game_state["coins"],
                                            self_position=game_state["self"][3])

        # 2D array indicating which area is affected by exploded bombs or by bombs which are about to explode
        explosion_map = get_bomb_map(object_position=game_state["field"], bomb_list=game_state["bombs"],
                                     explosion_position=game_state["explosion_map"])

        # 1D array indicating whether current field, up, right, down, left are dangerous
        danger = get_danger(explosion_map=explosion_map, self_position=game_state["self"][3])

        # 1D array indicating in which direction to flee from bomb if immediate danger
        # if no immediate danger return 5 zeros or current spot is safe returns all zeros, otherwise direction
        safe_direction = get_safe_direction(object_position=game_state["field"], explosion_map=explosion_map,
                                            self_position=game_state["self"][3])

        crate_direction = get_crate_direction(object_position=game_state["field"], bomb_list=game_state["bombs"],
                                              self_position=game_state["self"][3], explosion_map=explosion_map)

        # 1D array indicating whether bomb can be dropped and survival is possible
        bomb_info = get_bomb_info(object_position=game_state["field"], explosion_map=explosion_map,
                                  self=game_state["self"], bomb_list=game_state["bombs"])

        features = np.concatenate([awareness,
                                   danger,
                                   safe_direction,
                                   coin_direction,
                                   crate_direction,
                                   bomb_info])

        return features


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
        if q.empty():
            return None

        # always get first element
        node = q.get()

        # found a coin, trace back to parent, but not if coin is where initial position is
        if node.parent_position is not None and node.position in coin_position:
            actions = []
            cells = []
            # Backtracking: From each node grab state and action; and then redefine node as parent node
            while node.parent_position is not None:
                actions.append(node.move)
                cells.append(node.position)
                node = node.parent_position
            # Reverse is a method for lists that reverses the content
            actions.reverse()
            cells.reverse()
            return actions, cells

        explored.add(node.position)

        # Add neighbors to frontier
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
        if explosion_map[node.position] == 0 or (explosion_map[node.position] + (node.steps * 1/6)) > 1:
            moves = []
            cells = []
            # Backtracking: From each node grab state and action; and then redefine node as parent node
            while node.parent_position is not None:
                moves.append(node.move)
                cells.append(node.position)
                node = node.parent_position
            # Reverse is a method for lists that reverses the order
            moves.reverse()
            cells.reverse()
            return moves, cells

        explored.add(node.position)

        # Add neighbors to frontier
        neighbors = get_neighbors(object_position, node.position)
        for action, neighbor in zip(neighbors["actions"], neighbors["neighbors"]):
            # make sure that we do not die on the way to the potential neighbor
            if neighbor not in explored and not q.contains_state(neighbor) and not 5/6 <= (explosion_map[neighbor] + ((node.steps+1) * 1/6)) <= 1:
                # TODO: Check whether steps is really always + 1, but should make sense in BFS
                child = Node(position=neighbor, parent_position=node, move=action, steps=node.steps+1)
                q.put(child)


# reduce default dist to make computations faster
# TODO: Improve performance here
def crate_bfs(object_position, self_position, bomb_list, explosion_map, max_dist=14, distance_discount=1/4):
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
                moves = []
                cells = []
                # Backtracking: From each node grab state and action; and then redefine node as parent node
                while node.parent_position is not None:
                    moves.append(node.move)
                    cells.append(node.position)
                    node = node.parent_position
                # Reverse is a method for lists that reverses the order
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
        if dist_to_self < max_dist:
            # Add neighbors to queue
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
    bomb_list_tmp.append(((position), 4))
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
    return np.array([
        object_position[self_position[0] - 1, self_position[1]],  # up
        object_position[self_position[0], self_position[1] + 1],  # right
        object_position[self_position[0] + 1, self_position[1]],  # down
        object_position[self_position[0], self_position[1] - 1]   # left
    ]) == 0


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
    ctd_to_score = lambda ctd: 5/6 - (1/6 * ctd)
    if bomb_list is None:
        return explosion_map
    else:
        bombs = bomb_list.copy()
        for bomb_pos, bomb_ctd in bombs:
            # position of the bomb itself
            explosion_map[bomb_pos] = max(explosion_map[bomb_pos], ctd_to_score(bomb_ctd))
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
        if crate_info == ([], []):          # current position
            crate_direction[0] = 1
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
    Compute vector of lenght 2 where the positions indicate:
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


