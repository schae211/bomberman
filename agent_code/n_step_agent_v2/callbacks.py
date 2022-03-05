import os
import pickle
import numpy as np

# import MultiOutputRegressor to create MultiOutputRegressor with LGBMRegressor
from sklearn.multioutput import MultiOutputRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from numba import jit

# helper lists and dictionaries
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
ACTION_TRANSLATE = {
    "UP": 0,
    "RIGHT": 1,
    "DOWN": 2,
    "LEFT": 3,
    "WAIT": 4,
    "BOMB": 5
}
ACTION_TRANSLATE_REV = {
    0: "UP",
    1: "RIGHT",
    2: "DOWN",
    3: "LEFT",
    4: "WAIT",
    5: "BOMB"
}

# DEFAULT_PROBS = [.225, .225, .225, .225, .1, .0] # no bombs
# DEFAULT_PROBS = [.15, .15, .15, .15, .1, .3] # many bombs
DEFAULT_PROBS = [.2, .2, .2, .2, .1, .1]

# Define option for policy: {"stochastic", "deterministic"}
POLICY = "deterministic"

# Define option for feature engineering: {"channels", "standard", "minimal"}
FEAT_ENG = "standard"


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
        # n_jobs=-1 means that all CPUs are used.
        self.model = MultiOutputRegressor(LGBMRegressor(n_estimators=100, n_jobs=-1))
        # self.model = LinearRegression()
        # self.model = MultiOutputRegressor(RandomForestRegressor(n_estimators=10, n_jobs=-1))  # very slow!
    else:
        self.logger.info("Loading model from saved state.")
        # if a model has been trained before, load it again
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)

    # variable necessary to keep track if we have fitted our MultiOutputRegressor(LGBMRegressor(...))
    self.isFit = False


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

    elif self.train and not self.isFit:
        self.logger.debug("Choosing action at random because model is not fitted yet")
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
                probs = (q_values - q_values.min()) / (q_values.max() - q_values.min())  # min-max scaling
                probs = probs / probs.sum()  # normalization
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
        # for the coin challenge we need to know where the agent is, where walls are and where the coins are
        # so, we create a coin map in the same shape as the field
        coin_map = np.zeros_like(game_state["field"])
        for cx, cy in game_state["coins"]:
            coin_map[cx, cy] = 1

        # also adding where one self is on the map
        self_map = np.zeros_like(game_state["field"])
        self_map[game_state["self"][3]] = 1

        # create channels based on the field and coin information.
        channels = [self_map, game_state["field"], coin_map]

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

        crate_direction = get_crate_direction(object_position=game_state["field"], explosion_map=explosion_map,
                                              self_position=game_state["self"][3])

        # 1D array indicating whether bomb can be dropped and survival is possible
        bomb_info = get_bomb_info(object_position=game_state["field"], explosion_map=explosion_map,
                                  self=game_state["self"])

        features = np.concatenate([awareness,
                                   danger,
                                   safe_direction,
                                   coin_direction,
                                   crate_direction,
                                   bomb_info])

        return features

    elif FEAT_ENG == "minimal":

        # self position, basically needed for all features below
        self_position = np.array(game_state["self"][3])

        # Situational awareness, indicating in which directions the agent can move
        awareness = np.array([
            game_state["field"][self_position[0] - 1, self_position[1]],  # up
            game_state["field"][self_position[0], self_position[1] + 1],  # right
            game_state["field"][self_position[0] + 1, self_position[1]],  # down
            game_state["field"][self_position[0], self_position[1] - 1]  # left
        ])

        coins = game_state["coins"]
        # check if there are revealed coins
        # there are 4 possibilities (up, right, down, left), so we need a vector of length 2
        if len(coins) == 0:
            coin_direction = np.zeros(2)
        else:
            # perform BFS to find the nearest coin
            coin_info = coin_bfs(object_position=game_state["field"], coin_position=game_state["coins"],
                                 self_position=game_state["self"][3])

            coin_direction = np.zeros(2)
            if coin_info:  # can we even reach the revealed coins?
                if coin_info[0][0] == "up":
                    coin_direction[0] += 1
                elif coin_info[0][0] == "right":
                    coin_direction[1] -= 1
                elif coin_info[0][0] == "down":
                    coin_direction[0] += 1
                elif coin_info[0][0] == "left":
                    coin_direction[1] -= 1

        # get information about affected areas meaning where bombs are about to explode and where it is still dangerous
        # TODO: Consider countdown information
        explosion_map = game_state["explosion_map"].copy()
        bombs = game_state["bombs"].copy()
        for bomb_pos, bomb_ctd in bombs:
            explosion_map[bomb_pos] += 1
            # check above
            for up_x in range(bomb_pos[0] - 1, bomb_pos[0] - 4, -1):
                if 0 <= up_x <= 16:
                    if game_state["field"][up_x, bomb_pos[1]] == -1:
                        break
                    else:
                        explosion_map[up_x, bomb_pos[1]] += 1
            # check below
            for down_x in range(bomb_pos[0] + 1, bomb_pos[0] + 4, 1):
                if 0 <= down_x <= 16:
                    if game_state["field"][down_x, bomb_pos[1]] == -1:
                        break
                    else:
                        explosion_map[down_x, bomb_pos[1]] += 1
            # check to the left
            for left_y in range(bomb_pos[1] - 1, bomb_pos[1] - 4, -1):
                if 0 <= left_y <= 16:
                    if game_state["field"][bomb_pos[0], left_y] == -1:
                        break
                    else:
                        explosion_map[bomb_pos[0], left_y] += 1
            # check to the right
            for right_y in range(bomb_pos[1] + 1, bomb_pos[1] + 4, 1):
                if 0 <= right_y <= 16:
                    if game_state["field"][bomb_pos[0], right_y] == -1:
                        break
                    else:
                        explosion_map[bomb_pos[0], right_y] += 1
        # normalize explosion map (since two bombs are not more dangerous than one bomb)
        explosion_map = np.where(explosion_map > 0, 1, 0)

        # find the closest save spot using once again BFS, ones indicate save spots
        # we basically remove all the positions that are not returned by BFS indicating they are not the closest escape
        # again we need a vector of length 2,
        explosion_direction = np.zeros(2)
        if np.sum(explosion_map) > 0:
            explosion_info = save_bfs(object_position=game_state["field"], explosion_map=explosion_map,
                                      self_position=game_state["self"][3])
            if explosion_info == "dead":  # no safe position at all detected
                explosion_direction += 1
            elif explosion_info == ([], []):  # current position is safe
                pass
            elif explosion_info[0][0] == "up":  # up is safe
                explosion_direction[0] += 1
            elif explosion_info[0][0] == "right":  # right is safe
                explosion_direction[1] += 1
            elif explosion_info[0][0] == "down":  # down is safe
                explosion_direction[0] -= 1
            elif explosion_info[0][0] == "left":  # left is safe
                explosion_direction[1] -= 1

        # add feature vector indicating whether the self position or the adjacent fields are dangerous
        danger = np.array([explosion_map[self_position[0], self_position[1]],  # current position
                           explosion_map[self_position[0] - 1, self_position[1]],  # up
                           explosion_map[self_position[0], self_position[1] + 1],  # right
                           explosion_map[self_position[0] + 1, self_position[1]],  # down
                           explosion_map[self_position[0], self_position[1] - 1]])  # left

        # add feature vector indicating in which direction to move to destroy the most tiles
        # encoding:
        # TODO: Should we only consider this if we can throw a bomb?
        crate_direction = np.zeros(2)
        crate_info = crate_bfs(game_state["field"], game_state["self"][3], explosion_map)
        if crate_info:  # check whether at all it makes sense to destroy crates now
            if crate_info == ([], []):  # current position
                pass
            elif crate_info[0][0] == "up":
                crate_direction[0] += 1
            elif crate_info[0][0] == "right":
                crate_direction[1] += 1
            elif crate_info[0][0] == "down":
                crate_direction[0] -= 1
            elif crate_info[0][0] == "left":
                crate_direction[1] -= 1

        surviving_bomb = np.array(check_survival(game_state["field"], explosion_map, game_state["self"][3])[0]).reshape(
            -1)

        if explosion_map.sum() > 0:
            break_here = 0  # for debugging purposes, put condition above

        features = np.concatenate([awareness,
                                   coin_direction,
                                   explosion_direction,
                                   danger,
                                   crate_direction,
                                   surviving_bomb])

        return features


# define simple node class used for BFS
class Node(object):
    def __init__(self, position, parent_position, move, steps=None):
        self.position = position
        self.parent_position = parent_position
        self.move = move
        self.steps = steps


# define simple queue class that also allows checking for states (which is an attribute of the node)
class Queue(object):
    def __init__(self):
        self.fifo = []

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


def possible_moves(object_position, position):
    moves = []
    if object_position[position[0] - 1, position[1]] == 0:
        moves.append("up")
    if object_position[position[0], position[1] + 1] == 0:
        moves.append("right")
    if object_position[position[0] + 1, position[1]] == 0:
        moves.append("down")
    if object_position[position[0], position[1] - 1] == 0:
        moves.append("left")
    return moves


def get_neighbors(object_position, position):
    neighbors = []
    possible = possible_moves(object_position, position)
    for move in possible:
        if move == "up":
            neighbors.append((position[0] - 1, position[1]))
        elif move == "right":
            neighbors.append((position[0], position[1] + 1))
        elif move == "down":
            neighbors.append((position[0] + 1, position[1]))
        elif move == "left":
            neighbors.append((position[0], position[1] - 1))
    return {"actions": possible, "neighbors": neighbors}


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

        # found a save position
        if explosion_map[node.position] == 0:
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
            if neighbor not in explored and not q.contains_state(neighbor):
                # TODO: Check whether steps is really always + 1, but should make sense in BFS
                child = Node(position=neighbor, parent_position=node, move=action, steps=node.steps+1)
                q.put(child)


# reduce default dist to make computations faster
# TODO: Improve performance here
def crate_bfs(object_position, self_position, explosion_map, max_dist=10):
    """
    Find path to position where bomb destroys most crates via breadth-first search (BFS)
    Thereby, we have to take the distance to the considered positions into consideration (trade-off!)
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
        if check_survival(object_position, explosion_map, node.position):

            # compute number of destroyed crates
            destroyed_crates = get_destroyed_crates(object_position, node.position)

            # TODO: Think about how to compute this destruction score, taking 1/4 because bomb takes 4 steps to explode
            # combine distance and destroyed crates into a score
            destruction_score = destroyed_crates - 1/4 * dist_to_self

            # found a better position according to our destruction score
            if destruction_score > top_score:
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


def check_survival(object_position, explosion_map, position):
    # add information about whether dropping a bomb is suicide
    updated_explosion_map = explosion_map.copy()
    updated_explosion_map[position] += 1
    # check above
    for up_x in range(position[0] - 1, position[0] - 4, -1):
        if 0 <= up_x <= 16:
            if object_position[up_x, position[1]] == -1:
                break
            else:
                updated_explosion_map[up_x, position[1]] += 1
    # check below
    for down_x in range(position[0] + 1, position[0] + 4, 1):
        if 0 <= down_x <= 16:
            if object_position[down_x, position[1]] == -1:
                break
            else:
                updated_explosion_map[down_x, position[1]] += 1
    # check to the left
    for left_y in range(position[1] - 1, position[1] - 4, -1):
        if 0 <= left_y <= 16:
            if object_position[position[0], left_y] == -1:
                break
            else:
                updated_explosion_map[position[0], left_y] += 1
    # check to the right
    for right_y in range(position[1] + 1, position[1] + 4, 1):
        if 0 <= right_y <= 16:
            if object_position[position[0], right_y] == -1:
                break
            else:
                updated_explosion_map[position[0], right_y] += 1

    # normalize explosion map (since two bombs are not more dangerous than one bomb)
    updated_explosion_map = np.where(updated_explosion_map > 0, 1, 0)
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
    # TODO: Consider adding countdown information
    explosion_map = explosion_position.copy().astype(float)
    ctd_to_score = lambda ctd : 5/6 - (1/6 * ctd)
    if len(bomb_list) > 0:
        break_here = 0
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

    # normalize explosion map (since two bombs are not more dangerous than one bomb)
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


def get_crate_direction(object_position, explosion_map, self_position):
    crate_direction = np.zeros(5)
    crate_info = crate_bfs(object_position, self_position, explosion_map)

    if crate_info is not None:  # None is returned if destruction score was negative for all considered positions
        crate_direction[0] = 1              # indicating that target was found
        if crate_info == ([], []):          # current position
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


def get_bomb_info(object_position, explosion_map, self):
    bomb_info = np.zeros(2)
    # check if bomb action is possible
    if self[2]:
        bomb_info[0] = 1
    if check_survival(object_position, explosion_map, self[3]):
        bomb_info[1] = 1
    return bomb_info


