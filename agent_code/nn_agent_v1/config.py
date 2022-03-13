
from datetime import datetime
from easydict import EasyDict as edict
import os
import pandas as pd

configs = edict({
    # which agent to use: {"MLP", "CNN"}
    "AGENT": "CNN",
    # epsilon-greedy strategy epsilon parameter = probability to do random move
    "EPSILON": 1,
    # epsilon-greedy strategy decay parameter: epsilon * decay^(#episode)
    "EPSILON_DECAY": 0.9998,
    # epsilon-greedy strategy minimum epsilon: epsilon := max(0.05, epsilon * decay^(#episode))
    "EPSILON_MIN": 0.01,
    # discount factor gamma, which discount future rewards
    "GAMMA": 0.9,
    # N-step temporal difference learning parameter, how many steps to look ahead for computing q-value updates
    "N_STEPS": 10,
    # storing the last x transition as replay buffer fo r training
    "MEMORY_SIZE": 10_000,
    # how many transitions should be sampled from the memory to train the model
    "SAMPLE_SIZE": 128,
    # should we exploit symmetries to augment the training data
    "TS_AUGMENTATION": True,
    # what batch size should be used to train the model
    "BATCH_SIZE": 32,  # 1024,  # 128
    # policy: {"deterministic", "stochastic"}
    "POLICY": "deterministic",
    # default probabilities for the actions [up, right, down, left, wait, bomb]
    "DEFAULT_PROBS": [.2, .2, .2, .2, .1, .1],
    # determines the behavior of the states_to_features function: {"channels", "standard", "channels+bomb"}
    "FEATURE_ENGINEERING": "channels",
    # what loss to use for nn: {mse, huber}
    "LOSS": "huber",
    # learning rate used for gradient descent in nn
    "LEARNING_RATE": 0.0001,
    # should we use prioritized experience replay or sample the batches randomly
    "PRIORITIZED_REPLAY": True,
    # Parameters for prioritized experience replay:
    "CONST_E": 1,
    "CONST_A": 0.8,
    # how often to update the target network
    "UPDATE_FREQ": 10,
    # whether to load a model
    "LOAD": False,
    # where to load the model
    "LOAD_PATH": os.path.expanduser("~/bomberman_stats/11-03-2022-20-51_dnn-v1_1_0.9998_0.05_0.9_10_10000_256_deterministic_channels_huber_0.0001_True_10_model.pt"),
    # where to store and load the model,
    "MODEL_LOC": os.path.expanduser("~/bomberman_stats")
})

auxiliary_rewards = edict({
    "MOVE_IN_CIRCLES": False,
    "MOVE_TO_OR_FROM_COIN": False,
    "STAY_OR_ESCAPE_BOMB": False
})

reward_specs = edict({
    # original events
    "MOVED_RIGHT": 0,
    "MOVED_UP": 0,
    "MOVED_DOWN": 0,
    "MOVED_LEFT": 0,
    "WAITED": -0.5,
    "INVALID_ACTION": -2,
    "BOMB_DROPPED": 0,
    "BOMB_EXPLODED": 0,
    "CRATE_DESTROYED": 2,
    "COIN_FOUND": 2,
    "COIN_COLLECTED": 5,
    "KILLED_OPPONENT": 25,
    "KILLED_SELF": 0,  # setting to 0 because also included in GOT_KILLED
    "GOT_KILLED": -25,
    "OPPONENT_ELIMINATED": 0,
    "SURVIVED_ROUND": 0,
    # auxiliary events to reward shaping
    "MOVE_TO_COIN": 0,
    "MOVE_FROM_COIN": 0,
    "MOVE_IN_CIRCLES": 0,
    "STAY_IN_BOMB": 0,
    "ESCAPE_FROM_BOMB": 0
})

# needed for generating the right shapes
feature_specs = edict({
    "channels": edict({
        "shape": [4, 17, 17]
    }),
    "standard": edict({
        "shape": [25]
    })
})


SAVE_KEY = f'{configs["AGENT"]}_{configs["EPSILON"]}_{configs["EPSILON_DECAY"]}_{configs["EPSILON_MIN"]}_{configs["GAMMA"]}_{configs["N_STEPS"]}_{configs["MEMORY_SIZE"]}_{configs["BATCH_SIZE"]}_{configs["POLICY"]}_{configs["FEATURE_ENGINEERING"]}_{configs["LOSS"]}_{configs["LEARNING_RATE"]}_{configs["PRIORITIZED_REPLAY"]}_{configs["UPDATE_FREQ"]}'
SAVE_TIME = datetime.now().strftime("%d-%m-%Y-%H-%M")

# SAVE all configs as dataframe with the same SAVE_KEY
save_dict = {}
save_dict.update(configs)
save_dict.update(auxiliary_rewards)
save_dict.update(reward_specs)
save_dict["DEFAULT_PROBS"] = str(save_dict["DEFAULT_PROBS"])
pd.DataFrame(save_dict, index=[1]).to_csv(f"{configs.MODEL_LOC}/{SAVE_TIME}_{SAVE_KEY}_configs.csv", index=False)