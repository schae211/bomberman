
from datetime import datetime
from easydict import EasyDict as edict
import os
import pandas as pd

configs = edict({
    # which MLP to use: {"CNN", "MLP", "CNNPlus", "MLPPlus", "MLP_2"}
    "AGENT": "MLPPlus",
    # epsilon-greedy strategy epsilon parameter = probability to do random move
    "EPSILON": 1.0,
    # epsilon-greedy strategy decay parameter: epsilon * decay^(#episode)
    "EPSILON_DECAY": 0.9999,
    # epsilon-greedy strategy minimum epsilon: epsilon := max(0.05, epsilon * decay^(#episode))
    "EPSILON_MIN": 0.001,
    # length of epsilon-greedy decay
    "EPSILON_DECAY_LEN": 20_000,
    # epsilon decay linear or exponential
    "EPSILON_DECAY_LINEAR": True,
    # discount factor gamma, which discount future rewards
    "GAMMA": 0.99,
    # N-step temporal difference learning parameter, how many steps to look ahead for computing q-value updates
    "N_STEPS": 1,
    # storing the last x transition as replay buffer fo r training
    "MEMORY_SIZE": 12_000,
    # how many transitions should be sampled from the memory to train the model
    "SAMPLE_SIZE": 1024,
    # should we exploit symmetries to augment the training data
    "TS_AUGMENTATION": False,
    # what batch size should be used to train the model
    "BATCH_SIZE": 256,
    # policy: {"deterministic", "stochastic"}
    "POLICY": "deterministic",
    # default probabilities for the actions [up, right, down, left, wait, bomb]
    "DEFAULT_PROBS": [.2, .2, .2, .2, .1, .1],
    # determines the behavior of the states_to_features function: {"channels", "standard", "channels+bomb", "standard_extended", "standard_strategy"}
    "FEATURE_ENGINEERING": "standard_extended",
    # what loss to use for nn: {mse, huber}
    "LOSS": "huber",
    # learning rate used for gradient descent in nn
    "LEARNING_RATE": 0.00025,
    # should we use prioritized experience replay or sample the batches randomly
    "PRIORITIZED_REPLAY": True,
    # Parameters for prioritized experience replay:
    "CONST_E": 1,
    "CONST_A": 0.8,
    # how often to update the target network
    "UPDATE_FREQ": 10,
    # whether to load a model
    "LOAD": True,
    # where to load the model
    "LOAD_PATH": os.path.expanduser("~/bomberman_stats/pretrain_models/27-03-2022-20-28_MLPPlus_1.0_0.9999_0.001_0.99_1_12000_256_deterministic_standard_extended_huber_0.00025_True_10_model.pt"),
    # where to store and load the model,
    "MODEL_LOC": os.path.expanduser("~/bomberman_stats"),
    # including some comment
    "COMMENT": "testing extended mlp with new features",
    # include command line call
    "CALL": "python main.py play --n-rounds 500000 --agents DeathPhil_agent rule_based_agent rule_based_agent rule_based_agent --scenario classic --train 1 --no-gui",
    # use other agent to guide the first x episodes (our pretrain method)
    "PRETRAIN": False,
    # number of episodes to use pretraining
    "PRETRAIN_LEN": 0,
    # location of the save pretrain agent model
    "PRETRAIN_LOC":  os.path.expanduser("~/bomberman_stats/pretrain_models/26-03-2022-10-01_MLPPlus_1.0_0.9999_0.001_0.99_1_12000_256_deterministic_standard_extended_huber_0.00025_True_10_model.pt"),
    # pretrain feature engineering: {"channels", "standard", "channels+bomb"}
    "PRETRAIN_FEATURES": "standard",
    # fraction of random moves performed by pretrained agent (so no perfect performance)
    "PRETRAIN_RANDOM": 1.0,
    # decay of the random playing rate for pretraining
    "PRETRAIN_RANDOM_DECAY": 0.9999
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
    "CRATE_DESTROYED": 1,
    "COIN_FOUND": 0,
    "COIN_COLLECTED": 5,
    "KILLED_OPPONENT": 25,
    "KILLED_SELF": 0,  # setting to 0 because also included in GOT_KILLED
    "GOT_KILLED": -25,
    "OPPONENT_ELIMINATED": 0,
    "SURVIVED_ROUND": 0,
    # auxiliary events to reward shaping
    "MOVE_TO_COIN": 1,
    "MOVE_FROM_COIN": -1,
    "MOVE_IN_CIRCLES": -1,
    "STAY_IN_BOMB": -1,
    "ESCAPE_FROM_BOMB": 1
})

# needed for generating the right shapes
feature_specs = edict({
    "channels": edict({
        "shape": [5, 17, 17]
    }),
    "standard": edict({
        "shape": [30]
    }),
    "channels+bomb": edict({
        "shape": [6, 17, 17]
    }),
    "standard_extended": edict({
        "shape": [112]
    }),
    "standard_strategy": edict({
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
save_dict["DEFAULT_PROBS"] = str(save_dict["DEFAULT_PROBS"])  # make list to string
pd.DataFrame(save_dict, index=[1]).to_csv(f"{configs.MODEL_LOC}/{SAVE_TIME}_{SAVE_KEY}_configs.csv", index=False)