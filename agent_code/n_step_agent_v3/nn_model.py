import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Dataset
import numpy as np


# Hyperparameters
BATCH_SIZE = 32
LOSS_FUNCTION = nn.MSELoss()
LEARNING_RATE = 0.0001
# play --no-gui --n-rounds 10000 --agents n_step_agent_v3 --train 1 --scenario crate_heaven

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


class NNModel(nn.Module):
    def __init__(self):
        super(NNModel, self).__init__()
        self.model = MLP(input_size=25, output_size=6, hidden_size=128)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=LEARNING_RATE)
        # self.model = torch.load("/Users/maxschafer/PycharmProjects/bomberman/agent_code/n_step_agent_v3/best_my-saved-model.pt")
        # self.model.eval()
        print("LOAD MODEL")

    def predict(self, state: np.ndarray) -> np.ndarray:
        self.model.eval()
        return self.model(torch.Tensor(state).to(device)).detach().cpu().numpy()

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
        self.model.train()
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            # Compute prediction error
            pred = self.model(X)
            loss = LOSS_FUNCTION(pred, y)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        torch.save(self.model, "/Users/maxschafer/PycharmProjects/bomberman/"
                               "agent_code/n_step_agent_v3/my-saved-model.pt")
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
