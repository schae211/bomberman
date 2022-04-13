
import torch
import torch.nn as nn

import numpy as np
from agent_code.nn_agent_v2.config import configs

# Set device (either cpu or cuda)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


class PretrainedModel(nn.Module):
    """
    Loading NN to get predictions for pretraining.
    """
    def __init__(self):
        super(PretrainedModel, self).__init__()
        print("LOAD MODEL")
        self.policy_net = torch.load(configs.PRETRAIN_LOC,
                                     map_location=torch.device(device))
        self.policy_net.to(device)
        self.policy_net.eval()

    def predict(self, state: np.ndarray) -> np.ndarray:
        self.policy_net.eval()
        return self.policy_net(torch.Tensor(state).to(device)).detach().cpu().numpy()




