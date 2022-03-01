import os
import pickle
import random
import numpy as np
from collections import deque

# import MultiOutputRegressor to create MultiOutputRegressor with LGBMRegressor
from sklearn.multioutput import MultiOutputRegressor
from lightgbm import LGBMRegressor

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

# Setting up the hyper parameters (is it ok to put the here?
# starting with a simple espilon greedy strategy
EPSILON = 0.1

# for later usage to reduce exploration over time, for simplicity we will start with epsilon-greedy strategy
# EXPLORATION_MAX = 1.0
# we must push minimum exploration rate up to prevent the learner from getting stuck.
# It is likely that the small memory we have will get filled with low quality experiences,
# so we need to keep exploring
# EXPLORATION_MIN = 0.05
# EXPLORATION_DECAY = 0.96


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
        # random initialization of the weight with uniform [0,1)
        # weights = np.random.rand(len(ACTIONS))
        # normalize the probabilities to get sum = 1
        # self.model = weights / weights.sum()
        self.model = MultiOutputRegressor(LGBMRegressor(n_estimators=100, n_jobs=-1))
    else:
        self.logger.info("Loading model from saved state.")
        # if a model has been trained before, load it again
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)

    # to navigate exploration-exploitation tradeoff
    self.epsilon = EPSILON

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
    if self.train and np.random.rand() < EPSILON:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb (are these good initialization?)
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])
    if self.isFit:
        self.logger.debug("Querying model for action.")
        # or array.reshape(1, -1) if it contains a single sample
        features = state_to_features(game_state).reshape(1, -1)
        self.logger.debug(f"Shape of the features: {features.shape}")

        # compute q-values using our fitted model, important to flatten the output again
        q_values = self.model.predict(features).reshape(-1)
        self.logger.debug(f"Shape of the predicted q-values: {q_values.shape}")

        # normalize the q_values, take care not to divide by zero (fall back to default probs)
        if q_values.sum() != 0:
            probs = q_values / q_values.sum()
        else:
            probs = [.2, .2, .2, .2, .1, .1]

        # using a stochastic policy!
        return np.random.choice(ACTIONS, p=probs)
    # if we have not yet fit the model return random action
    else:
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])


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

    # for learning saving the game state if not None:
    # with open("/Users/philipp/game_dict.pt", "wb") as file: pickle.dump(game_state, file)
    # with open("/Users/philipp/game_dict.pt", "rb") as file: game_dict = pickle.load(file)

    # for the coin challenge we need to know where walls are and where the coins are
    # so, we create a coin map in the same shape as the field
    coin_map = np.zeros(game_state["field"].shape)
    for cx, cy in game_state["coins"]:
        coin_map[cx, cy] = 1

    # create channels based on the field and coin information.
    channels = [game_state["field"], coin_map]
    # concatenate them as a feature tensor (they must have the same shape), ...
    stacked_channels = np.stack(channels)
    # and return them as a vector
    return stacked_channels.reshape(-1)
