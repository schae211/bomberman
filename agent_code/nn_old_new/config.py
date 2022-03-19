
import pandas as pd
from datetime import datetime
from easydict import EasyDict as edict
import os

configs = edict({
    "AGENT": "dnn-v1",
    # epsilon-greedy strategy epsilon parameter = probability to do random move
    "EPSILON": 1,
    # epsilon-greedy strategy decay parameter: epsilon(t) := epsilon(t-1) * decay^(#episode)
    "EPSILON_DECAY": 0.9998,
    # epsilon-greedy strategy minimum epsilon: epsilon(t) := max(0.05, epsilon(t-1) * decay^(#episode))
    "EPSILON_MIN": 0.05,
    # discount factor gamma, which discount future rewards
    "GAMMA": 0.9,
    # N-step temporal difference learning parameter, how many steps to look ahead for computing q-value updates
    "N_STEPS": 10,
    # storing the last x transition as replay buffer fo r training
    "MEMORY_SIZE": 10_000,
    # how many transitions should be sampled from the memory to train the model
    "BATCH_SIZE": 256,  # 1024,  # 128
    # use "deterministic" or "stochastic" policy
    "POLICY": "deterministic",
    # default probabilities for the actions [up, right, down, left, wait, bomb]
    "DEFAULT_PROBS": [.2, .2, .2, .2, .1, .1],
    # determines the behavior of the states_to_features function: {"channels", "standard", "minimal"}
    "FEATURE_ENGINEERING": "channels",
    # what loss to use for nn: {mse, huber}
    "LOSS": "huber",
    # learning rate used for gradient descent in nn
    "LEARNING_RATE": 0.0001,
    # should we use prioritized experience replay or sample the batches randomly
    "PRIORITIZED_REPLAY": True,
    # how often to update the target network
    "UPDATE_FREQ": 10,
    # where to store and load the model,
    "MODEL_LOC": os.path.expanduser("~/bomberman_stats")
})

SAVE_KEY = f'{configs["AGENT"]}_{configs["EPSILON"]}_{configs["EPSILON_DECAY"]}_{configs["EPSILON_MIN"]}_{configs["GAMMA"]}_{configs["N_STEPS"]}_{configs["MEMORY_SIZE"]}_{configs["BATCH_SIZE"]}_{configs["POLICY"]}_{configs["FEATURE_ENGINEERING"]}_{configs["LOSS"]}_{configs["LEARNING_RATE"]}_{configs["PRIORITIZED_REPLAY"]}_{configs["UPDATE_FREQ"]}'
SAVE_TIME = datetime.now().strftime("%d-%m-%Y-%H-%M")

# SAVE all configs as dataframe with the same SAVE_KEY
save_dict = {}
save_dict.update(configs)
#save_dict.update(auxiliary_rewards)
#save_dict.update(reward_specs)
save_dict["DEFAULT_PROBS"] = str(save_dict["DEFAULT_PROBS"])  # make list to string
pd.DataFrame(save_dict, index=[1]).to_csv(f"{configs.MODEL_LOC}/{SAVE_TIME}_{SAVE_KEY}_configs.csv", index=False)

# TODO:
#  Include reward information in the config files, we have to gather all tunable parameters in one file, this way
#  we can ensure that we keep track of the things we already tested in the past.