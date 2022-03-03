
import os
import pickle
import numpy as np

# import MultiOutputRegressor to create MultiOutputRegressor with LGBMRegressor
from sklearn.multioutput import MultiOutputRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

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
# DEFAULT_PROBS = [.2, .2, .2, .2, .1, .1]
DEFAULT_PROBS = [.15, .15, .15, .15, .1, .3]

# Define option for policy: {"stochastic", "deterministic"}
POLICY = "deterministic"

# Define option for feature engineering: {"channels", "minimal"}
FEAT_ENG = "minimal"


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
    if self.train and np.random.rand() <= max(self.epsilon_min, self.epsilon*self.epsilon_reduction**episode_n):
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
                probs = (q_values-q_values.min()) / (q_values.max()-q_values.min())  # min-max scaling
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
    # This is the dict before the game begins and after it ends
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

    elif FEAT_ENG == "minimal":

        # self position
        s = np.array(game_state["self"][3])

        # also adding situational awareness, indicating in which directions the agent could move
        awareness = np.array([
            game_state["field"][s[0] - 1, s[1]],    # up
            game_state["field"][s[0], s[1] + 1],    # right
            game_state["field"][s[0] + 1, s[1]],    # down
            game_state["field"][s[0], s[1] - 1]     # left
        ])
        coins = game_state["coins"]
        if len(coins) == 0:
            coin_direction = np.zeros(4)
        else:
            # perform BFS to find the nearest coin
            coin_info = coin_bfs(object_position=game_state["field"], coin_position=game_state["coins"],
                                 self_position=game_state["self"][3])

            coin_direction = np.zeros(4)
            if coin_info[0][0] == "up":
                coin_direction[0] += 1
            elif coin_info[0][0] == "right":
                coin_direction[1] += 1
            elif coin_info[0][0] == "down":
                coin_direction[2] += 1
            elif coin_info[0][0] == "left":
                coin_direction[3] += 1

        # get information about affected areas
        explosion_map = game_state["explosion_map"].copy()
        bombs = game_state["bombs"].copy()
        for bomb_pos, bomb_ctd in bombs:
            explosion_map[bomb_pos] += 1
            # check above
            for up_x in range(bomb_pos[0]-1, bomb_pos[0]-4, -1):
                if 0 <= up_x <= 16:
                    if game_state["field"][up_x, bomb_pos[1]] == -1:
                        break
                    else:
                        explosion_map[up_x, bomb_pos[1]] += 1
            # check below
            for down_x in range(bomb_pos[0]+1, bomb_pos[0]+4, 1):
                if 0 <= down_x <= 16:
                    if game_state["field"][down_x, bomb_pos[1]] == -1:
                        break
                    else:
                        explosion_map[down_x, bomb_pos[1]] += 1
            # check to the left
            for left_y in range(bomb_pos[1]-1, bomb_pos[1]-4, -1):
                if 0 <= left_y <= 16:
                    if game_state["field"][bomb_pos[0], left_y] == -1:
                        break
                    else:
                        explosion_map[bomb_pos[0], left_y] += 1
            # check to the right
            for right_y in range(bomb_pos[1]+1, bomb_pos[1]+4, 1):
                if 0 <= right_y <= 16:
                    if game_state["field"][bomb_pos[0], right_y] == -1:
                        break
                    else:
                        explosion_map[bomb_pos[0], right_y] += 1
        # normalize explosion map
        explosion_map = np.where(explosion_map > 0, True, False)

        # find the closest save spot
        explosion_direction = np.zeros(4)
        if np.sum(explosion_map) > 0:
            explosion_info = save_bfs(object_position=game_state["field"], explosion_map=explosion_map, self_position=game_state["self"][3])
            if explosion_info == ([], []):  # ([], []) is returned if already in save position
                pass
            elif explosion_info[0][0] == "up":
                explosion_direction[0] += 1
            elif explosion_info[0][0] == "right":
                explosion_direction[1] += 1
            elif explosion_info[0][0] == "down":
                explosion_direction[2] += 1
            elif explosion_info[0][0] == "left":
                explosion_direction[3] += 1

        features = np.concatenate([awareness, coin_direction, explosion_direction])
        return features


# define simple queue object that also allows checking for states
class Queue(object):
    def __init__(self):
        self.fifo = []

    def put(self, node):
        self.fifo.append(node)

    def contains_state(self, state):
        return any(node.state == state for node in self.fifo)

    def empty(self):
        return len(self.fifo) == 0

    def get(self):
        if self.empty():
            raise Exception("empty queue")
        else:
            node = self.fifo.pop(0)
            return node


class Node(object):
    def __init__(self, state, parent, action):
        self.state = state
        self.parent = parent
        self.action = action


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
            neighbors.append((position[0]-1, position[1]))
        elif move == "right":
            neighbors.append((position[0], position[1]+1))
        elif move == "down":
            neighbors.append((position[0]+1, position[1]))
        elif move == "left":
            neighbors.append((position[0], position[1]-1))
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
    q.put(Node(state=self_position, parent=None, action=None))

    # loop over the queue as long as it is not empty
    while True:
        if q.empty():
            raise Exception("no solution")

        # always get first element
        node = q.get()

        # found a coin, trace back to parent, but not if coin is where initial position is
        if node.parent is not None and node.state in coin_position:
            actions = []
            cells = []
            # Backtracking: From each node grab state and action; and then redefine node as parent node
            while node.parent is not None:
                actions.append(node.action)
                cells.append(node.state)
                node = node.parent
            # Reverse is a method for lists that reverses the content
            actions.reverse()
            cells.reverse()
            return actions, cells

        explored.add(node.state)

        # Add neighbors to frontier
        neighbors = get_neighbors(object_position, node.state)
        for action, neighbor in zip(neighbors["actions"], neighbors["neighbors"]):
            if neighbor not in explored and not q.contains_state(neighbor):
                child = Node(state=neighbor, parent=node, action=action)
                q.put(child)


def save_bfs(object_position, explosion_map, self_position):
    """
    Find path to nearest save position via breadth-first search (BFS)
    :param explosion_map:
    :param self_position:
    :return:
    """
    q = Queue()
    explored = set()
    # add start to the Queue
    q.put(Node(state=self_position, parent=None, action=None))

    # loop over the queue as long as it is not empty
    while True:
        if q.empty():
            raise Exception("no solution")  # should not happen

        # always get first element
        node = q.get()

        # found a save position
        if not explosion_map[node.state]:
            actions = []
            cells = []
            # Backtracking: From each node grab state and action; and then redefine node as parent node
            while node.parent is not None:
                actions.append(node.action)
                cells.append(node.state)
                node = node.parent
            # Reverse is a method for lists that reverses the order
            actions.reverse()
            cells.reverse()
            return actions, cells

        explored.add(node.state)

        # Add neighbors to frontier
        neighbors = get_neighbors(object_position, node.state)
        for action, neighbor in zip(neighbors["actions"], neighbors["neighbors"]):
            if neighbor not in explored and not q.contains_state(neighbor):
                child = Node(state=neighbor, parent=node, action=action)
                q.put(child)


