
configs = {
    "AGENT": "linear-sgd",
    # epsilon-greedy strategy epsilon parameter = probability to do random move
    "EPSILON": 0.2,
    # epsilon-greedy strategy decay parameter: epsilon(t) := epsilon(t-1) * decay^(#episode)
    "EPSILON_DECAY": 0.95,
    # epsilon-greedy strategy minimum epsilon: epsilon(t) := max(0.05, epsilon(t-1) * decay^(#episode))
    "EPSILON_MIN": 0.05,
    # discount factor gamma, which discount future rewards
    "GAMMA": 0.9,
    # N-step temporal difference learning parameter, how many steps to look ahead for computing q-value updates
    "N_STEPS": 10,
    # storing the last x transition as replay buffer for training
    "MEMORY_SIZE": 10_000,
    # how many transitions should be sampled from the memory to train the model
    "BATCH_SIZE": 1_000,
    # use "deterministic" or "stochastic" policy
    "POLICY": "stochastic",
    # default probabilities for the actions [up, right, down, left, wait, bomb]
    "DEFAULT_PROBS": [.2, .2, .2, .2, .1, .1],
    # determines the behavior of the states_to_features function: {"channels", "standard", "minimal"}
    "FEATURE_ENGINEERING": "channels"
}