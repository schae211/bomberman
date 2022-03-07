
from datetime import datetime
import os

configs = {
    "AGENT": "nn-v1",
    # epsilon-greedy strategy epsilon parameter = probability to do random move
    "EPSILON": 0.2,
    # epsilon-greedy strategy decay parameter: epsilon(t) := epsilon(t-1) * decay^(#episode)
    "EPSILON_DECAY": 0.999,
    # epsilon-greedy strategy minimum epsilon: epsilon(t) := max(0.05, epsilon(t-1) * decay^(#episode))
    "EPSILON_MIN": 0.05,
    # discount factor gamma, which discount future rewards
    "GAMMA": 0.9,
    # N-step temporal difference learning parameter, how many steps to look ahead for computing q-value updates
    "N_STEPS": 10,
    # storing the last x transition as replay buffer fo r training
    "MEMORY_SIZE": 10_000,
    # how many transitions should be sampled from the memory to train the model
    "BATCH_SIZE": 1024,  # 128
    # use "deterministic" or "stochastic" policy
    "POLICY": "deterministic",
    # default probabilities for the actions [up, right, down, left, wait, bomb]
    "DEFAULT_PROBS": [.2, .2, .2, .2, .1, .1],
    # determines the behavior of the states_to_features function: {"channels", "standard", "minimal"}
    "FEATURE_ENGINEERING": "standard",
    # where to store and load the model,
    "MODEL_LOC": os.path.expanduser("~/bomberman_stats")
}

SAVE_KEY = f'{configs["AGENT"]}_{configs["EPSILON"]}_{configs["EPSILON_DECAY"]}_{configs["EPSILON_MIN"]}_{configs["GAMMA"]}_{configs["N_STEPS"]}_{configs["MEMORY_SIZE"]}_{configs["BATCH_SIZE"]}_{configs["POLICY"]}_{configs["FEATURE_ENGINEERING"]}'
SAVE_TIME = datetime.now().strftime("%d-%m-%Y-%H-%M")

