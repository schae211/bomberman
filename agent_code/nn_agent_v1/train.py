from collections import namedtuple, deque

import numpy as np
from typing import List
import pandas as pd
from numba import jit

import events as e
from agent_code.nn_agent_v1.callbacks import state_to_features, coin_bfs, save_bfs, get_bomb_map
from agent_code.nn_agent_v1.config import configs, feature_specs, reward_specs, auxiliary_rewards, SAVE_KEY, SAVE_TIME

Transition = namedtuple("Transition", ("round", "state", "action", "next_state", "reward"))

# helper lists and dictionaries for actions (e.g. to translate between index and action string)
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
ACTION_TRANSLATE = {"UP": 0, "RIGHT": 1, "DOWN": 2, "LEFT": 3, "WAIT": 4, "BOMB": 5}
ACTION_TRANSLATE_REV = {val: key for key, val in ACTION_TRANSLATE.items()}


# Saving training statistics
SAVE_EVERY = 100   # write the stats to disk (csv) every x episodes
INCLUDE_EVERY = 4  # only include every x episode in the saved stats
step_information = {"round": [], "step": [], "events": [], "reward": []}
episode_information = {"round": [], "TS_MSE_1": [], "TS_MSE_2": [], "TS_MSE_3": [], "TS_MSE_4": [], "TS_MSE_5": [], "TS_MSE_6": []}


def setup_training(self):
    """
    Initialise self for training purpose.
    This is called after `setup` in callbacks.py.
    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Set up an array that will keep track of transition tuples (s, a, r, s')
    self.memory = deque(maxlen=configs.MEMORY_SIZE)
    self.last_states = deque(maxlen=5)  # used for reward shaping

    # adding epsilon var to agent
    self.epsilon = configs.EPSILON
    self.epsilon_reduction = configs.EPSILON_DECAY
    self.epsilon_min = configs.EPSILON_MIN


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

    # importantly, we will never fill our memory if the state is missing (old_game_state is None)
    if old_game_state:
        rewards = reward_from_events(self, events, old_game_state, new_game_state)
        self.memory.append(Transition(old_game_state["round"],              # round
                                      state_to_features(old_game_state),    # state
                                      self_action,                          # action
                                      state_to_features(new_game_state),    # next_state
                                      rewards))                             # reward


        # use global step information variable
        if (old_game_state["round"] % INCLUDE_EVERY) == 0:
            step_information["round"].append(old_game_state["round"])
            step_information["step"].append(old_game_state["step"])
            step_information["events"].append("| ".join(map(repr, events)))
            step_information["reward"].append(rewards)

        # keep status of last N steps for reward shaping
        self.last_states.append(old_game_state["self"][3])


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.
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

    if last_game_state["round"] % SAVE_EVERY == 0:  # save the TS stats about every x rounds.
        pd.DataFrame(step_information).to_csv(f"{configs.MODEL_LOC}/{SAVE_TIME}_{SAVE_KEY}_game_stats.csv", index=False)

    # updating the target network
    if last_game_state["round"] % configs.UPDATE_FREQ == 0:
        self.model.update_target()

    if configs.PRIORITIZED_REPLAY:
        priorities, states, updated_Qs = compute_priority(self)
        batch = np.random.choice(a=np.arange(0, len(self.memory)), size=min(configs.SAMPLE_SIZE, len(self.memory)),
                                 replace=False, p=priorities)
        X, y = states[batch, :], updated_Qs[batch, :]
    else:
        batch = np.random.choice(a=np.arange(0, len(self.memory)), size=min(configs.SAMPLE_SIZE, len(self.memory)),
                                 replace=False)
        X, y = compute_TS(self, batch)

    if configs.TS_AUGMENTATION:
        X, y = get_augmented_TS(configs.FEATURE_ENGINEERING, X, y)

    # reshape our predictors and checking the shape
    self.logger.debug(f"Shape x: {X.shape}, Shape y: {y.shape}")
    self.model.fit(input_data=X, target=y)

    global episode_information
    if last_game_state:
        episode_information["round"].append(last_game_state["round"])
        TS_MSE = ((self.model.predict_policy(X) - y) ** 2).mean(axis=0)  # compute TS error
        for i in range(6): episode_information[f"TS_MSE_{i+1}"].append(TS_MSE[i])

    if last_game_state["round"] % SAVE_EVERY == 0:
        pd.DataFrame(episode_information).to_csv(f"{configs.MODEL_LOC}/{SAVE_TIME}_{SAVE_KEY}_episode_stats.csv", index=False)


# this is where we are going to specify the rewards for certain actions
def reward_from_events(self, events: List[str], old_game_state: dict, new_game_state: dict) -> int:
    """
    Computing the rewards for our agent in a given step
    """
    # function computes auxiliary events/rewards and appends them to the event list and returns the extended event list
    events = compute_auxiliary_rewards(self, events, old_game_state, new_game_state)

    game_rewards = {
        # original events
        e.MOVED_RIGHT: reward_specs.MOVED_RIGHT,
        e.MOVED_UP: reward_specs.MOVED_UP,
        e.MOVED_DOWN: reward_specs.MOVED_DOWN,
        e.MOVED_LEFT: reward_specs.MOVED_LEFT,
        e.WAITED: reward_specs.WAITED,
        e.INVALID_ACTION: reward_specs.INVALID_ACTION,
        e.BOMB_DROPPED: reward_specs.BOMB_DROPPED,
        e.BOMB_EXPLODED: reward_specs.BOMB_EXPLODED,
        e.CRATE_DESTROYED: reward_specs.CRATE_DESTROYED,
        e.COIN_FOUND: reward_specs.COIN_FOUND,
        e.COIN_COLLECTED: reward_specs.COIN_COLLECTED,
        e.KILLED_OPPONENT: reward_specs.KILLED_OPPONENT,
        e.KILLED_SELF: reward_specs.KILLED_SELF,
        e.GOT_KILLED: reward_specs.GOT_KILLED,
        e.OPPONENT_ELIMINATED: reward_specs.OPPONENT_ELIMINATED,
        e.SURVIVED_ROUND: reward_specs.SURVIVED_ROUND,
        # auxiliary events and rewards:
        e.MOVE_TO_COIN: reward_specs.MOVE_TO_COIN,
        e.MOVE_FROM_COIN: reward_specs.MOVE_FROM_COIN,
        e.MOVE_IN_CIRCLES: reward_specs.MOVE_IN_CIRCLES,
        e.STAY_IN_BOMB: reward_specs.STAY_IN_BOMB,
        e.ESCAPE_FROM_BOMB: reward_specs.ESCAPE_FROM_BOMB
    }

    reward_sum = np.array([game_rewards[event] for event in events if event in game_rewards]).sum()
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum


def prep_memory(self):
    """
    Put our memory dequeue into np.arrays which are suitable to be process by numba jit compiled functions
    :param self:
    :return:
    """
    memory = self.memory
    # translating memory to numpy arrays suitable for processing with numba jit compilation
    episodes = np.zeros(len(memory))
    states = np.zeros([len(memory)] + feature_specs[configs.FEATURE_ENGINEERING].shape)
    actions = np.zeros(len(memory)).astype(np.int)
    next_states = np.zeros([len(memory)] + feature_specs[configs.FEATURE_ENGINEERING].shape)
    rewards = np.zeros(len(memory))
    # initializing boolean arrays indicating whether a given state or next state was present or not
    proper_next_states = np.ones(len(memory)).astype(np.bool)
    # now we need to loop through the dequeue and populate the np arrays.
    for i in range(len(memory)):
        episodes[i] = memory[i].round
        actions[i] = ACTION_TRANSLATE[memory[i].action]
        rewards[i] = self.memory[i].reward
        states[i] = memory[i].state

        if memory[i].next_state is None:
            proper_next_states[i] = False
        else:                                           # if next_state is terminal state, we will just keep all
            next_states[i] = memory[i].next_state       # the values zero! (and take note of it in proper_next_states)
    return episodes, states, actions, next_states, proper_next_states, rewards


@jit(nopython=True)
def compute_n_step_reward(episodes, rewards, N, GAMMA):
    """
    Compute the discounted sum of the next N steps for a given state S_t
    :param episodes:
    :param rewards:
    :return:
    """
    gammas = np.repeat(GAMMA, N)**(np.arange(N))            # gammas already taken to the power of the step
    n_step_rewards = np.zeros(len(episodes))                # initialize array to store the n_step rewards
    loop_untils = np.empty(len(episodes))
    for i in range(len(loop_untils)):                       # making sure to prevent index out of range errors and looping into the next episode
        loop_untils[i] = max([t for t in range(min(i+N+1, len(episodes))) if episodes[t] == episodes[i]])  # so t can maximally be i+N or len(episodes)-1
    for i in range(len(n_step_rewards)):                    # looping through each step
        r = np.zeros(N)
        for idx, t in enumerate(range(i, loop_untils[i])):
            r[idx] = rewards[t]
        r *= gammas
        n_step_rewards[i] = r.sum()
    return n_step_rewards, loop_untils


def compute_priority(self):
    episodes, states, actions, next_states, proper_next_states, rewards = prep_memory(self)
    # n_step rewards contains the sum of the discounted rewards of the 10 next steps
    n_step_rewards, loop_untils = compute_n_step_reward(episodes, rewards, configs.N_STEPS, configs.GAMMA)
    # st_plus_N_Qs contains the maximum action-value for the state s_(t+N) if this state exists, otherwise it should simply be zero (which it will be since we initialized with 0)
    st_plus_N_Qs = np.zeros(len(n_step_rewards))
    # check whether the corresponding next state for a loop until index is present ("proper") meaning not None
    st_plus_N_Qs[proper_next_states[loop_untils.astype(np.int16)]] = \
        np.amax(self.model.predict_target(next_states[proper_next_states[loop_untils.astype(np.int16)],:]), axis=1)
    # adding together the discounted rewards for the next N steps and the maximum action-value of the state s_(t+N)
    q_updates = (n_step_rewards + st_plus_N_Qs)
    # predicting the original action-values for the state s_t
    old_Qs = self.model.predict_policy(states)
    # subsetting only the action-values of the actions that were actually taken
    old_Qs_reduced = np.take_along_axis(arr=old_Qs, indices=actions[:,None], axis=1).reshape(-1)
    # update Q values (np.expand_dims is very useful here)
    updated_Qs = old_Qs.copy()
    np.put_along_axis(arr=updated_Qs, indices=np.expand_dims(actions, axis=1),
                      values=np.expand_dims(q_updates, axis=1), axis=1)
    # computing the absolute temporal difference error
    TDs = np.abs(q_updates - old_Qs_reduced)
    # adding the constant e to ensure that every state has some probability
    TDs += configs.CONST_E
    # computing the probabilities/priorities from the TD errors
    priorities = (TDs ** configs.CONST_A) / (TDs ** configs.CONST_A).sum()
    return priorities, states, updated_Qs


# computing the training sets using the old way since we supposedly have some bugs in the other version
def compute_TS(self, batch):
    X = np.zeros([len(batch)] + feature_specs[configs.FEATURE_ENGINEERING].shape)
    y = np.zeros([len(batch)] + [6])

    # extract information from each transition tuple (as stored above)
    for ts_index, i in enumerate(batch):
        # get episode and state, translate action to integer
        episode, state, action = self.memory[i].round, self.memory[i].state, ACTION_TRANSLATE[self.memory[i].action]

        # check that we loop only over the current episode and also not further than the length of our memory
        loop_until = max(t for t in range(min(i+configs.N_STEPS+1, len(self.memory))) if self.memory[t].round == episode)  # t can maximally be i+N or len(self.memory)-1
        rewards = [self.memory[t].reward for t in range(i, loop_until)] # get the reward of the N next steps
        gammas = [configs.GAMMA ** t for t in range(0, len(rewards))]   # compute discounted gammas
        n_steps_reward = (np.array(rewards) * np.array(gammas)).sum()   # elementwise multiplication with gammas and summation

        # standard case: non-terminal state and model is fit
        if self.memory[loop_until].next_state is not None:
            q_update = n_steps_reward + configs.GAMMA ** configs.N_STEPS * \
                       np.amax(self.model.predict_target(self.memory[loop_until].next_state))
        # we cannot predict q-values for the terminal state (which is none)
        else:
            q_update = n_steps_reward

        q_values = self.model.predict_policy(state).reshape(-1)  # use policy network to predict all q-values
        q_values[action] = q_update                              # replace the q-value of the action that was actually taken

        X[ts_index,:], y[ts_index,:] = state, q_values

    return X, y


def get_augmented_TS(FEAT_ENG, TS_X, TS_y):
    Augmented_X = np.zeros([TS_X.shape[0]*4] + feature_specs[configs.FEATURE_ENGINEERING].shape)
    Augmented_y = np.zeros((TS_X.shape[0]*4, 6))

    if FEAT_ENG == "channels":
        for num_rot in range(4):
            Augmented_X[num_rot*TS_X.shape[0]:(num_rot+1)*TS_X.shape[0],:] = \
                np.rot90(TS_X, num_rot, axes=(2, 3))
            Augmented_y[num_rot*TS_X.shape[0]:(num_rot+1)*TS_X.shape[0],:] = \
                rotated_actions(num_rot, TS_y)
        return Augmented_X, Augmented_y

    elif FEAT_ENG == "standard":
        for num_rot in range(4):
            Augmented_X[num_rot*TS_X.shape[0]:(num_rot+1)*TS_X.shape[0],:] = \
                rotated_standard_features(num_rot, TS_X)
            Augmented_y[num_rot*TS_X.shape[0]:(num_rot+1)*TS_X.shape[0],:] = \
                rotated_actions(num_rot, TS_y)
        return Augmented_X, Augmented_y

    elif FEAT_ENG == "channels+bomb":
        raise NotImplementedError


def rotated_actions(rot, TS_y):
    """
    rotation as defined in the unit circle (so to the left)
    reading from top to bottom, e.g. up (first index) becomes after 90° left (fourth index), and after 180° down (third index)
    mapping from default action sequence:   ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]
                                               0      1       2        3       4       5
    to 90° rotation:                        ["LEFT", "UP", "RIGHT", "DOWN", "WAIT", "BOMB"]
    to 180° rotation:                       ["DOWN", "LEFT", "UP", "RIGHT", "WAIT", "BOMB"]
    to 270° rotation:                       ["RIGHT", "DOWN", "LEFT", "UP", "WAIT", "BOMB"]
    keep waiting and bomb the same
    :param rot: integer {0,1,2,3}
    :param q_values: np.array with shape = (#actions,)
    :return: q_values: np.array with shape = (#actions,) adjusted according to the rotation
    """
    order_orig = [0, 1, 2, 3, 4, 5]
    order_90 =   [1, 2, 3, 0, 4, 5]
    order_180 =  [2, 3, 0, 1, 4, 5]
    order_270 =  [3, 0, 1, 2, 4, 5]

    rotated_y = np.zeros_like(TS_y)
    if rot == 0:
        rotated_y = TS_y.copy()
    elif rot == 1:
        for i, new_i in enumerate(order_90):
            rotated_y[:, i] = TS_y[:, new_i]
    elif rot == 2:
        for i, new_i in enumerate(order_180):
            rotated_y[:, i] = TS_y[:, new_i]
    elif rot == 3:
        for i, new_i in enumerate(order_270):
            rotated_y[:, i] = TS_y[:, new_i]
    return rotated_y


def rotated_standard_features(rot, TS_X):
    """
    rotation as defined in the unit circle (so to the left)
    mapping from default action sequence:   ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]
                                               0      1       2        3       4       5
    to 90° rotation:                        ["LEFT", "UP", "RIGHT", "DOWN", "WAIT", "BOMB"]
    to 180° rotation:                       ["DOWN", "LEFT", "UP", "RIGHT", "WAIT", "BOMB"]
    to 270° rotation:                       ["RIGHT", "DOWN", "LEFT", "UP", "WAIT", "BOMB"]
    keep waiting and bomb the same
    """
    order_orig = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
    order_90 =   [1, 2, 3, 0, 4, 6, 7, 8, 5, 9, 11, 12, 13, 10, 15, 16, 17, 14, 18, 20, 21, 22, 20, 23, 24, 25, 27, 28, 29, 26]
    order_180 =  [2, 3, 0, 1, 4, 7, 8, 5, 6, 9, 12, 13, 10, 11, 16, 17, 14, 15, 18, 21, 22, 19, 20, 23, 24, 25, 28, 29, 26, 27]
    order_270 =  [3, 0, 1, 2, 4, 8, 5, 6, 7, 9, 13, 10, 11, 12, 17, 14, 15, 16, 18, 22, 19, 20, 21, 23, 24, 25, 29, 26, 27, 28]

    rotated_X = np.zeros_like(TS_X)
    if rot == 0:
        rotated_X = TS_X.copy()
    elif rot == 1:
        for i, new_i in enumerate(order_90):
            rotated_X[:, i] = TS_X[:, new_i]
    elif rot == 2:
        for i, new_i in enumerate(order_180):
            rotated_X[:, i] = TS_X[:, new_i]
    elif rot == 3:
        for i, new_i in enumerate(order_270):
            rotated_X[:, i] = TS_X[:, new_i]
    return rotated_X


# define function to compute auxiliary rewards
def compute_auxiliary_rewards(self, events, old_game_state, new_game_state):
    if auxiliary_rewards.MOVE_TO_OR_FROM_COIN:
        # add reward/penalty based on whether the agent moved towards/away from the nearest coin (if coins were visible)
        if old_game_state and new_game_state and old_game_state["coins"] != []:
            coin_info = coin_bfs(old_game_state["field"], old_game_state["coins"], old_game_state["self"][3])
            if coin_info is not None:  # check whether we can even reach any revealed coin
                next_move = coin_info[1][0]
                if new_game_state["self"][3] == next_move:  # moved towards coin
                    events.append(e.MOVE_TO_COIN)
                elif old_game_state["self"][3] == new_game_state["self"][3]:  # not moved
                    pass
                else:  # moved away from coin
                    events.append(e.MOVE_FROM_COIN)
    elif auxiliary_rewards.STAY_OR_ESCAPE_BOMB:
        # reward/penalize if escaping/running into bomb
        if old_game_state and new_game_state and old_game_state["bombs"] != []:
            explosion_map = get_bomb_map(old_game_state["field"], old_game_state["bombs"], old_game_state["explosion_map"])
            # check if old position is in danger zone in explosion map
            explosion_info = save_bfs(object_position=old_game_state["field"], explosion_map=explosion_map,
                                      self_position=old_game_state["self"][3])
            if explosion_info != ([], []):  # if the old position is not safe
                next_move = explosion_info[1][0]
                if new_game_state["self"][3] == next_move:  # if we did make the move suggested by bfs
                    events.append(e.ESCAPE_FROM_BOMB)
                else:
                    events.append(e.STAY_IN_BOMB)
    elif auxiliary_rewards.MOVE_IN_CIRCLES:
        # check whether agent stayed in the same spot for too long (3 out of 5)
        if new_game_state:
            check_same_pos = np.array([state == new_game_state["self"][3] for state in self.last_states]).sum()
            if check_same_pos >= 3:
                events.append(e.MOVE_IN_CIRCLES)
    return events
