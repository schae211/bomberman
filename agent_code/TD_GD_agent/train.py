from collections import namedtuple, deque

import numpy as np
import pickle
from typing import List

import events as e
from .callbacks import state_to_features

# a way to structure our code?
Transition = namedtuple("Transition",
                        ("state", "action", "next_state", "reward"))

ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]
ACTION_TRANSLATE = {
    "UP": 0,
    "RIGHT": 1,
    "DOWN": 2,
    "LEFT": 3,
    "WAIT": 4,
    "BOMB": 5
}

# Hyper parameters -- DO modify

# discount rate
GAMMA = 0.95

# learning rate alpha
LEARNING_RATE = 0.001

# memory size "experience buffer"
MEMORY_SIZE = 20

# min size before starting to train? Should I implement this?
#BATCH_SIZE = 20

# what is this for?
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    TODO: Where should I initialize the model: self.model = MultiOutputRegressor(LGBMRegressor(n_estimators=100, n_jobs=-1))?

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Set up an array that will keep track of transition tuples (s, a, r, s')
    self.memory = deque(maxlen=MEMORY_SIZE)

    # the model is already setup in callback.py




def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # fill our memory after each step, state_to_features is defined in callbacks.py and imported above
    self.memory.append(Transition(state_to_features(old_game_state),    # state
                                  self_action,                          # action
                                  state_to_features(new_game_state),    # next_state
                                  reward_from_events(self, events)))    # reward

    # Idea: Add your own events to hand out rewards
    #if ...:
    #    events.append(PLACEHOLDER_EVENT)

def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    # info that episode is over
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')

    # Train the model
    self.logger.debug(f"Starting to train the model (has it been fit before={self.isFit})\n")

    # initialize our X and y which we use for fitting
    x = []
    y = []

    # extract information from each transition tuple (as stored above)
    for state, action, state_next, reward in self.memory:

        # I think I cannot use these for training
        if state is None or state_next is None:
            continue

        # translate action to int
        action = ACTION_TRANSLATE[action]
        self.logger.debug(f"Translated action: {action}")

        # if we have fit the model before q-update is the following
        if self.isFit:

            # compute q update according to the formula from the lecture, also don't forget to reshape here for single instance
            q_update = (reward + GAMMA * np.max(self.model.predict(state_next.reshape(1, -1))))

            self.logger.debug(f"Model has been fit before, computing proper q-update: {q_update}")
        # if we haven't fit the model before the q-update is only the reward
        else:
            q_update = reward

        # if we have fit the model before we predict the q-values, otherwise simply 0
        if self.isFit:
            # again don't forget to reshape
            q_values = self.model.predict(state.reshape(1, -1)).reshape(-1)
            self.logger.debug(f"Model has been fit before, predicting q-values from current model: {q_values}")
        else:
            q_values = np.zeros(len(ACTIONS))

        # for the action that we actually took update the q-value
        q_values[action] = q_update

        # check
        self.logger.debug(f"Shape of the q_values[0]: {q_values}")

        # appending the state to our X list
        # reshaping needed according to sklearn/utils/validation.py
        x.append(state)

        # appending the q-value which we want to predict using X to our target list
        y.append(q_values)

    # fitting the model and setting isFit to true
    # importantly partial fitting is not possible with most methods except for NN (so we fit again to the whole TS)
    # if y has the right size here (len = 6) everything should be fine
    self.logger.debug(f"Trying to fit the model:")
    # checking the shape
    self.logger.debug(f"x shapes: {[elem.shape for elem in x]}")
    self.logger.debug(f"y shapes: {[elem.shape for elem in y]}")
    # reshape our predictors
    x_reshaped = np.stack(x, axis=0)
    y_reshaped = np.stack(y, axis=0)
    self.logger.debug(f"Shape x_reshape: {x_reshaped.shape}")
    self.logger.debug(f"Shape y_reshape: {y_reshaped.shape}")
    self.model.fit(x_reshaped, y_reshaped)
    self.isFit = True

    # TODO: reducing the exploration rate over time to optimize the exploration-exploitation tradeoff.

    # should execute this code chunk before or after training?
    # initially I though that missing the next state makes my training impossible
    self.memory.append(Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))

    # Store the current model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)


# this is where we are going to specify the rewards for certain actions
def reward_from_events(self, events: List[str]) -> int:
    """
    Computing the rewards for our agent in a given step
    """
    game_rewards = {
        e.COIN_COLLECTED: 1
        # for now, I will keep it simply and only use one reward for collecting coins!
        #e.KILLED_OPPONENT: 5,
        #PLACEHOLDER_EVENT: -.1  # idea: the custom event is bad
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum


# TODO: Augment training data by exploiting symmetry (so we have to play less)
