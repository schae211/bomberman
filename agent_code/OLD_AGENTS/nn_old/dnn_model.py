import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Dataset
import numpy as np
from agent_code.OLD_AGENTS.nn_old.config import configs, SAVE_KEY, SAVE_TIME


# Hyperparameters
BATCH_SIZE = configs.BATCH_SIZE
if configs.LOSS == "huber":
    LOSS_FUNCTION = nn.HuberLoss()
elif configs.LOSS == "mse":
    LOSS_FUNCTION = nn.MSELoss()
LEARNING_RATE = configs.LEARNING_RATE
LOAD = True


# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


class DoubleNNModel(nn.Module):
    """
    Implementation of target network, which is used to compute the expected Q values and value/policy network,
    which is updated in each learning iteration. The "older" target network is updated with the
    "value network" every x steps to keep it current
    """
    def __init__(self):
        super(DoubleNNModel, self).__init__()
        if LOAD:
            print("LOAD MODEL")
            self.policy_net = torch.load("/Users/philipp/Python_projects/bomberman_rl/agent_code/nn_agent_v1/model.pt",
                                    map_location=torch.device('cpu'))
            self.target_net = torch.load("/Users/philipp/Python_projects/bomberman_rl/agent_code/nn_agent_v1/model.pt",
                                    map_location=torch.device('cpu'))
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_net.eval()
        else:
            print("INITIALIZE MODEL")
            self.policy_net = MLP(input_size=25, output_size=6, hidden_size=128)
            self.target_net = MLP(input_size=25, output_size=6, hidden_size=128)
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.optimizer = torch.optim.AdamW(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.policy_net.to(device)
        self.target_net.to(device)
        self.target_net.eval()

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def predict_policy(self, state: np.ndarray) -> np.ndarray:
        self.policy_net.eval()
        return self.policy_net(torch.Tensor(state).to(device)).detach().cpu().numpy()

    def predict_target(self, state: np.ndarray) -> np.ndarray:
        self.target_net.eval()
        return self.target_net(torch.Tensor(state).to(device)).detach().cpu().numpy()

    def fit(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Trains the model on a batch of data
        Args:
            X: stack states as numpy array [size, input_size]
            y: stack Q-Values as numPy array [size, num_actions]

        Returns:

        """
        dataset = StateValueDataset(states=X, values=y)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        size = len(dataset)
        self.policy_net.train()
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            # Compute prediction error
            pred = self.policy_net(X)
            loss = LOSS_FUNCTION(pred, y)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        torch.save(obj=self.policy_net, f=f'{configs["MODEL_LOC"]}/{SAVE_TIME}_{SAVE_KEY}_model.pt')

        return loss


class MLP(nn.Module):
    def __init__(self, input_size=25, output_size=6, hidden_size=128):
        super(MLP, self).__init__()
        self.net = nn.Sequential(nn.Linear(input_size, hidden_size),
                                 nn.ReLU(),
                                 nn.Linear(hidden_size, hidden_size),
                                 nn.ReLU(),
                                 nn.Linear(hidden_size, output_size))

    def forward(self, x):
        """
        Forward pass of the neural network.
        Args:
            x: [batch_size, input_size] tensor

        Returns: [batch_size, output_size] tensor
        """
        return self.net(x)


class StateValueDataset(Dataset):
    def __init__(self, states: np.ndarray, values: np.ndarray):
        self.input_data = states.astype(np.float32)
        self.target = values.astype(np.float32)

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.input_data[idx], self.target[idx]
