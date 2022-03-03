from collections import namedtuple, deque

import numpy as np
import pickle
from typing import List
import pandas as pd
import random

import events as e
from .callbacks import state_to_features

# a way to structure our code?
Transition = namedtuple("Transition",
                        ("round", "state", "action", "next_state", "reward"))

ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]
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

# Hyper parameters -- DO modify

# discount rate
GAMMA = 0.95

# setting the parameter for epsilon-greedy policy, epsilon is the probability to do random move
EPSILON = 0.2

# reducing epsilon over time
EPSILON_REDUCTION = 0.99

# n-step temporal difference learning
N = 5

# memory size "experience buffer", if I fit the model only after each episode a large memory size should be fine
# i think this parameter effectively determines how fast I can train the model (i.e. how long each round takes)
MEMORY_SIZE = 200

# what batch size should we use, and should we sample randomly or prioritize?
BATCH_SIZE = 200

# what is this for?
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# should training data be augmented? {True, False}
# keep in mind that it only works for proper channels at the moment.
AUGMENT = False

# Needed for augmenting training data
GAME_SIZE = 7

# specify argument whether training statistics should be saved
SAVE_TRAIN = True
SAVE_AFTER = 100
if SAVE_TRAIN:
    step_information = {"round": [], "step": [], "events": [], "reward": []}
    episode_information = {"round": [], "TS_MSE_1": [], "TS_MSE_2": [], "TS_MSE_3": [],
                           "TS_MSE_4": [], "TS_MSE_5": [], "TS_MSE_6": []}


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Set up an array that will keep track of transition tuples (s, a, r, s')
    self.memory = deque(maxlen=MEMORY_SIZE)

    # adding epsilon var to agent
    self.epsilon = EPSILON
    self.epsilon_reduction = EPSILON_REDUCTION


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
    if old_game_state:
        self.memory.append(Transition(old_game_state["round"],
                                      state_to_features(old_game_state),                                    # state
                                      self_action,                                                          # action
                                      state_to_features(new_game_state),                                    # next_state
                                      reward_from_events(self, events, old_game_state, new_game_state)))    # reward

    # use global step information variable
    global step_information
    if old_game_state:
        step_information["round"].append(old_game_state["round"])
        step_information["step"].append(old_game_state["step"])
        step_information["events"].append("| ".join(map(repr, events)))
        step_information["reward"].append(reward_from_events(self, events, old_game_state, new_game_state))

    # if True:
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
    # also adding the last state to the memory (which we are not using anyway)
    self.memory.append(Transition(last_game_state["round"],
                                  state_to_features(last_game_state),
                                  last_action,
                                  None,
                                  reward_from_events(self, events, last_game_state, None)))

    global step_information
    step_information["round"].append(last_game_state["round"])
    step_information["step"].append(last_game_state["step"])
    step_information["events"].append("| ".join(map(repr, events)))
    step_information["reward"].append(reward_from_events(self, events, last_game_state, None))

    if last_game_state["round"] >= SAVE_AFTER:
        pd.DataFrame(step_information).to_csv("game_stats.csv", index=False)

    # Train the model
    self.logger.debug(f"Starting to train the model (has it been fit before={self.isFit})\n")

    # initialize our x and y which we use for fitting later on
    x = []
    y = []

    # get a batch
    if len(self.memory) > BATCH_SIZE:
        #batch = random.sample(self.memory, BATCH_SIZE)
        batch = random.sample(range(len(self.memory)), BATCH_SIZE)
    else:
        batch = range(len(self.memory))

    # extract information from each transition tuple (as stored above)
    for i in batch:

        # get episode
        episode = self.memory[i].round
        state = self.memory[i].state
        immediate_reward = self.memory[i].reward

        # translate action to int
        action = ACTION_TRANSLATE[self.memory[i].action]

        # check whether state is none, which corresponds to the first state
        if self.memory[i].state is None:
            continue

        if self.isFit:

            # get the reward of the N next steps, check that loop does not extend over length of memory
            rewards = []
            loop_until = min(i+N, len(self.memory))  # Fixme: Does this still work with proper batches?
            for t in range(i, loop_until):
                # check whether we are still in the same episode
                if self.memory[t].round == episode:
                    rewards.append(self.memory[t].reward)

            gammas = [GAMMA**t for t in range(0, len(rewards))]

            # now multiply elementwise discounted gammas by the rewards and get the sum
            n_steps_reward = (np.array(rewards) * np.array(gammas)).sum()

            # if we have a terminal state in the next states we cannot predict q-values for it
            if self.memory[len(rewards)-1].next_state is None:
                q_update = n_steps_reward
            else:
                q_update = n_steps_reward + GAMMA**N * \
                           np.amax(self.model.predict(self.memory[len(rewards)-1].next_state.reshape(1, -1)))

            # use the model to predict all the other q_values
            # (below we replace the q_value for the selected action with this q_update)
            q_values = self.model.predict(state.reshape(1, -1)).reshape(-1)

        # if we haven't fit the model before the q-update is only the reward, and we set all other q_values to 0
        else:
            q_update = immediate_reward
            q_values = np.zeros(len(ACTIONS))

        # for the action that we actually took update the q-value according to above
        q_values[action] = q_update

        # append the predictors x=state, and response y=q_values
        x.append(state)
        y.append(q_values)

        if AUGMENT:
            augmented_states, augmented_values = augment_training(state, q_values)
            for s, qval in zip(augmented_states, augmented_values):
                x.append(s)
                y.append(qval)

    # importantly partial fitting is not possible with most methods except for NN (so we fit again to the whole TS)
    self.logger.debug(f"Fitting the model using the input as specified below:")

    # reshape our predictors and checking the shape
    x_reshaped = np.stack(x, axis=0)
    y_reshaped = np.stack(y, axis=0)
    self.logger.debug(f"Shape x_reshape: {x_reshaped.shape}")
    self.logger.debug(f"Shape y_reshape: {y_reshaped.shape}")
    self.model.fit(X=x_reshaped, y=y_reshaped)
    self.isFit = True

    global episode_information
    if last_game_state:
        episode_information["round"].append(last_game_state["round"])
        TS_MSE = ((self.model.predict(x_reshaped) - y_reshaped)**2).mean(axis=0)
        episode_information["TS_MSE_1"].append(TS_MSE[0])
        episode_information["TS_MSE_2"].append(TS_MSE[1])
        episode_information["TS_MSE_3"].append(TS_MSE[2])
        episode_information["TS_MSE_4"].append(TS_MSE[3])
        episode_information["TS_MSE_5"].append(TS_MSE[4])
        episode_information["TS_MSE_6"].append(TS_MSE[5])

    if last_game_state["round"] >= SAVE_AFTER:
        pd.DataFrame(episode_information).to_csv("episode_stats.csv", index=False)

    # Store the current model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)


# this is where we are going to specify the rewards for certain actions
def reward_from_events(self, events: List[str], old_game_state: dict, new_game_state) -> int:
    """
    Computing the rewards for our agent in a given step
    """

    # for the first step nothing is returned, but I need the step argument to discount the coin reward
    if old_game_state is None:
        step = 0
    else:
        step = old_game_state["step"]

    # add reward/penalty based on  whether the agent moved towards/away from the nearest coin (manhatten distance)
    if old_game_state and new_game_state:
        old_self = np.array(old_game_state["self"][3])
        coins = old_game_state["coins"]
        distances = [np.abs(np.array(coin_tup) - old_self).sum() for coin_tup in coins]
        closest_coin = np.argmin(distances)
        original_distance = np.min(distances)
        new_self = np.array(new_game_state["self"][3])
        updated_distance = np.abs(np.array(coins[closest_coin]) - new_self).sum()
        if updated_distance < original_distance:
            events.append(e.MOVE_TO_COIN)
        elif updated_distance > original_distance:
            events.append(e.MOVE_FROM_COIN)

    game_rewards = {
        # TODO: Tune (e.g. reduce) the discount factor
        #e.COIN_COLLECTED: 100 * 0.99**step,  # discount the reward for collecting coins over time
        e.COIN_COLLECTED: 40,
        e.KILLED_SELF: -20,
        e.INVALID_ACTION: -1,
        e.WAITED: -1,
        e.MOVE_TO_COIN: 5,
        e.MOVE_FROM_COIN: -5
        # e.KILLED_OPPONENT: 5 # not useful at the moment
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]

    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum


# TODO: Augment training data by exploiting symmetry (so we have to play less)
def augment_training(state, q_values):
    """
    Exploiting the rotational and mirror symmetry we can augment training data,
    We just need to make sure to adjust all the movement directions accordingly
    So we have 4 rotations: 90, 180, 270, 360 and each can be mirrored => TSx8!
    :return:
    """
    augmented_states = []
    augmented_values = []
    reshape_feature = state.reshape(-1, GAME_SIZE, GAME_SIZE)
    # looping through the rotation angles:
    for i in range(4):
        augmented_states.append(np.rot90(reshape_feature, i, axes=(1, 2)).reshape(-1))
        augmented_values.append(rotated_actions(i, q_values))
    return augmented_states, augmented_values


def rotated_actions(rot, q_values):
    """
    mapping from default action sequence:   ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]
                                               0      1       2        3       4       5
    to 90° rotation:                        ["LEFT", "UP", "RIGHT", "DOWN", "WAIT", "BOMB"]
    to 180° rotation:                       ["DOWN", "LEFT", "UP", "RIGHT", "WAIT", "BOMB"]
    to 270° rotation:                        ["RIGHT", "DOWN", "LEFT", "UP", "WAIT", "BOMB"]
    keep waiting and bomb the same
    :param rot: integer {0,1,2,3}
    :param q_values: np.array with shape = (#actions,)
    :return: q_values: np.array with shape = (#actions,) adjusted according to the rotation
    """
    if rot == 0:
        return q_values
    elif rot == 1:
        return np.array([q_values[3], q_values[0], q_values[1], q_values[2], q_values[4], q_values[5]])
    elif rot == 2:
        return np.array([q_values[2], q_values[3], q_values[0], q_values[1], q_values[4], q_values[5]])
    elif rot == 3:
        return np.array([q_values[1], q_values[2], q_values[3], q_values[0], q_values[4], q_values[5]])

