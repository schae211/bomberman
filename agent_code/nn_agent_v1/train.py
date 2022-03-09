from collections import namedtuple, deque

import numpy as np
import pickle
from typing import List
import pandas as pd
import random
from datetime import datetime

import events as e
from agent_code.nn_agent_v1.callbacks import state_to_features
from agent_code.nn_agent_v1.callbacks import coin_bfs, save_bfs
from agent_code.nn_agent_v1.callbacks import get_bomb_map
from agent_code.nn_agent_v1.config import configs
from agent_code.nn_agent_v1.config import SAVE_KEY, SAVE_TIME


# a way to structure our code?
Transition = namedtuple("Transition",
                        ("round", "state", "action", "next_state", "reward"))

# helper lists and dictionaries for actions
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
ACTION_TRANSLATE = {"UP": 0, "RIGHT": 1, "DOWN": 2, "LEFT": 3, "WAIT": 4, "BOMB": 5}
ACTION_TRANSLATE_REV = {val: key for key, val in ACTION_TRANSLATE.items()}


# Hyper parameters
GAMMA = configs["GAMMA"]
EPSILON = configs["EPSILON"]
EPSILON_REDUCTION = configs["EPSILON_DECAY"]
MIN_EPSILON = configs["EPSILON_MIN"]
N = configs["N_STEPS"]
MEMORY_SIZE = configs["MEMORY_SIZE"]
BATCH_SIZE = configs["BATCH_SIZE"]

# should training data be augmented? {True, False}
# keep in mind that it only works for proper channels at the moment.
AUGMENT = False

# Needed for augmenting training data
GAME_SIZE = 17

# specify argument whether training statistics should be saved
SAVE_TRAIN = True
SAVE_EVERY = 200
INCLUDE_EVERY = 4 # only include every x episode in the saved stats
SAVE_DIR = configs["MODEL_LOC"]
if SAVE_TRAIN:
    step_information = {"round": [], "step": [], "events": [], "reward": []}
    episode_information = {"round": [], "TS_MSE_1": [], "TS_MSE_2": [], "TS_MSE_3": [],
                           "TS_MSE_4": [], "TS_MSE_5": [], "TS_MSE_6": []}

# global variable to store the last states used to shape rewards
LAST_STATES = 5

#
PRIORITIZED_REPLAY = configs.PRIORITIZED_REPLAY


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Set up an array that will keep track of transition tuples (s, a, r, s')
    self.memory = deque(maxlen=MEMORY_SIZE)
    self.last_states = deque(maxlen=LAST_STATES)

    # adding epsilon var to agent
    self.epsilon = EPSILON
    self.epsilon_reduction = EPSILON_REDUCTION
    self.epsilon_min = MIN_EPSILON


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

    # specify that we want to use the global step information variable
    global step_information

    # fill our memory after each step, state_to_features is defined in callbacks.py and imported above
    if old_game_state:
        rewards = reward_from_events(self, events, old_game_state, new_game_state)
        self.memory.append(Transition(old_game_state["round"],
                                      state_to_features(old_game_state),  # state
                                      self_action,  # action
                                      state_to_features(new_game_state),  # next_state
                                      rewards))  # reward


        # use global step information variable
        if (old_game_state["round"] % INCLUDE_EVERY) == 0:
            step_information["round"].append(old_game_state["round"])
            step_information["step"].append(old_game_state["step"])
            step_information["events"].append("| ".join(map(repr, events)))
            step_information["reward"].append(rewards)

        # keep status of last N steps for reward shaping
        self.last_states.append(old_game_state["self"][3])

    # first game state, can still be used by using the round information from the new game state
    elif new_game_state and (new_game_state["round"] % INCLUDE_EVERY) == 0:
        step_information["round"].append(new_game_state["round"])
        step_information["step"].append(0)
        step_information["events"].append("| ".join(map(repr, events)))
        step_information["reward"].append(0)


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
    if (last_game_state["round"] % INCLUDE_EVERY) == 0:
        step_information["round"].append(last_game_state["round"])
        step_information["step"].append(last_game_state["step"])
        step_information["events"].append("| ".join(map(repr, events)))
        step_information["reward"].append(reward_from_events(self, events, last_game_state, None))

    if last_game_state["round"] % SAVE_EVERY == 0:
        pd.DataFrame(step_information).to_csv(f"{SAVE_DIR}/{SAVE_TIME}_{SAVE_KEY}_game_stats.csv", index=False)

    # initialize our x and y which we use for fitting later on
    x, y = [], []

    # TODO: Check if implementation for Prioritized Replay is correct
    if len(self.memory) > BATCH_SIZE:
        # Prioritized Replay: Before building our training dataset, we need to compute the temporal difference error
        # for transitions currently stored in the memory (aka replay buffer).
        priorities = get_priority(self)
        batch = np.random.choice(a=np.arange(0, len(self.memory)), size=BATCH_SIZE, replace=False, p=priorities)
    else:
        batch = range(len(self.memory))

    # extract information from each transition tuple (as stored above)
    for i in batch:

        # get episode
        episode = self.memory[i].round
        state = self.memory[i].state

        # translate action to int
        action = ACTION_TRANSLATE[self.memory[i].action]

        # check whether state is none, which corresponds to the first state
        if self.memory[i].state is None:
            continue

        # get the reward of the N next steps, check that loop does not extend over length of memory
        rewards = []
        loop_until = min(i + N, len(self.memory))  # prevent index out of range error
        for t in range(i, loop_until):
            # check whether we are still in the same episode
            if self.memory[t].round == episode:
                rewards.append(self.memory[t].reward)

        gammas = [GAMMA ** t for t in range(0, len(rewards))]

        # now multiply elementwise discounted gammas by the rewards and get the sum

        n_steps_reward = (np.array(rewards) * np.array(gammas)).sum()

        # standard case: non-terminal state and model is fit
        if self.memory[len(rewards) - 1].next_state is not None:
            q_update = n_steps_reward + GAMMA ** N * \
                       np.amax(self.model.predict(self.memory[len(rewards) - 1].next_state.reshape(1, -1)))
            # use the model to predict all the other q_values
            # (below we replace the q_value for the selected action with this q_update)
            q_values = self.model.predict(state.reshape(1, -1)).reshape(-1)
        # if we have a terminal state in the next states we cannot predict q-values for the terminal state
        else:
            q_update = n_steps_reward
            # use the model to predict all the other q_values
            # (below we replace the q_value for the selected action with this q_update)
            q_values = self.model.predict(state.reshape(1, -1)).reshape(-1)

        # for the action that we actually took update the q-value according to above
        q_values[action] = q_update

        # append the predictors x=state, and response y=q_values
        x.append(state)
        y.append(q_values)

    # importantly partial fitting is not possible with most methods except for NN (so we fit again to the whole TS)
    self.logger.debug(f"Fitting the model using the input as specified below:")

    # reshape our predictors and checking the shape
    x_reshaped = np.stack(x, axis=0)
    y_reshaped = np.stack(y, axis=0)
    self.logger.debug(f"Shape x_reshape: {x_reshaped.shape}")
    self.logger.debug(f"Shape y_reshape: {y_reshaped.shape}")
    self.model.fit(X=x_reshaped, y=y_reshaped)

    global episode_information
    if last_game_state:
        episode_information["round"].append(last_game_state["round"])
        TS_MSE = ((self.model.predict(x_reshaped) - y_reshaped) ** 2).mean(axis=0)
        episode_information["TS_MSE_1"].append(TS_MSE[0])
        episode_information["TS_MSE_2"].append(TS_MSE[1])
        episode_information["TS_MSE_3"].append(TS_MSE[2])
        episode_information["TS_MSE_4"].append(TS_MSE[3])
        episode_information["TS_MSE_5"].append(TS_MSE[4])
        episode_information["TS_MSE_6"].append(TS_MSE[5])

    if last_game_state["round"] % SAVE_EVERY == 0:
        pd.DataFrame(episode_information).to_csv(f"{SAVE_DIR}/{SAVE_TIME}_{SAVE_KEY}_episode_stats.csv", index=False)


# this is where we are going to specify the rewards for certain actions
def reward_from_events(self, events: List[str], old_game_state: dict, new_game_state: dict) -> int:
    """
    Computing the rewards for our agent in a given step
    """

    # for the first step nothing is returned, but I need the step argument to discount the coin reward
    if old_game_state is None:
        step = 0
    else:
        step = old_game_state["step"]

    # add reward/penalty based on whether the agent moved towards/away from the nearest coin (if coins were visible)
    if old_game_state and new_game_state and old_game_state["coins"] != []:
        coin_info = coin_bfs(old_game_state["field"], old_game_state["coins"], old_game_state["self"][3])
        if coin_info is not None:  # check whether we can even reach any revealed coin
            closest_coin = coin_info[1][-1]
            next_move = coin_info[1][0]
            if new_game_state["self"][3] == next_move:  # moved towards coin
                events.append(e.MOVE_TO_COIN)
            elif old_game_state["self"][3] == new_game_state["self"][3]:  # not moved
                pass
            else:  # moved away from coin
                events.append(e.MOVE_FROM_COIN)

    # reward/penalize if escaping/running into bomb
    if old_game_state and new_game_state and old_game_state["bombs"] != []:
        explosion_map = get_bomb_map(old_game_state["field"], old_game_state["bombs"], old_game_state["explosion_map"])
        # check if old position is in danger zone in explosion map
        explosion_info = save_bfs(object_position=old_game_state["field"], explosion_map=explosion_map,
                                  self_position=old_game_state["self"][3])
        if explosion_info != ([], []):  # if the old position is not safe
            closest_save_spot = explosion_info[1][-1]
            next_move = explosion_info[1][0]
            if new_game_state["self"][3] == next_move:  # if we did make the move suggested by bfs
                events.append(e.ESCAPE_FROM_BOMB)
            else:
                events.append(e.STAY_IN_BOMB)

    # check whether agent stayed in the same spot for too long (3 out of 5)
    if new_game_state:
        check_same_pos = np.array([state == new_game_state["self"][3] for state in self.last_states]).sum()
        if check_same_pos >= 3:
            events.append(e.MOVE_IN_CIRCLES)

    game_rewards = {
        e.COIN_COLLECTED: 5,
        e.KILLED_SELF: -20,
        e.INVALID_ACTION: -2,
        e.WAITED: -0.5,
        e.MOVE_TO_COIN: 1,
        e.MOVE_FROM_COIN: -1,
        e.MOVE_IN_CIRCLES: -1,
        e.CRATE_DESTROYED: 2,
        e.COIN_FOUND: 2,
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]

    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum


def get_priority(self):
    """
    Only called when the model is fitted
    Compute priority values for prioritized replay
    :param self:
    :return:
    """
    memory = self.memory
    temporal_differences = np.zeros(len(memory))
    for i in range(len(memory)):
        # get episode and state
        episode, state = memory[i].round, memory[i].state

        # translate action to int
        action = ACTION_TRANSLATE[memory[i].action]

        # starting states have no priority
        if memory[i].state is None:
            temporal_differences[i] = 0

        # get the reward of the N next steps, check that loop does not extend over length of memory
        rewards = []
        loop_until = min(i + N, len(memory))  # prevent index out of range error
        for t in range(i, loop_until):
            # check whether we are still in the same episode
            if memory[t].round == episode:
                rewards.append(memory[t].reward)

        gammas = [GAMMA ** t for t in range(0, len(rewards))]

        # now multiply elementwise discounted gammas by the rewards and get the sum
        n_steps_reward = (np.array(rewards) * np.array(gammas)).sum()

        # standard case: non-terminal state and model is fit
        if self.memory[len(rewards) - 1].next_state is not None:
            q_update = n_steps_reward + GAMMA ** N * \
                       np.amax(self.model.predict(self.memory[len(rewards) - 1].next_state.reshape(1, -1)))
            q_values = self.model.predict(state.reshape(1, -1)).reshape(-1)

        # if we have a terminal state in the next states we cannot predict q-values for the terminal state
        else:
            q_update = n_steps_reward
            q_values = self.model.predict(state.reshape(1, -1)).reshape(-1)

        temporal_differences[i] = np.abs(q_update - q_values[action])

        # adding constant to ensure that no experience has 0 probability to be taken
        const_e = 1  # TODO: How to choose e here?
        temporal_differences += const_e

        # computing the priority values
        const_a = 0.8  # TODO: How to choose e here?
        priorities = (temporal_differences**const_a)/(temporal_differences**const_a).sum()
        return priorities



