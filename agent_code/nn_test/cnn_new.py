
from collections import namedtuple, deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from agent_code.nn_test.config import configs
import numpy as np
import random
Transition = namedtuple("Transition", ("round", "state", "action", "next_state", "reward"))

device = "cuda" if torch.cuda.is_available() else "cpu"

class TestModel(nn.Module):
    """
    Implementation of target network, which is used to compute the expected Q values and value/policy network,
    which is updated in each learning iteration. The "older" target network is updated with the
    "value network" every x steps to keep it current
    """
    def __init__(self):
        super(TestModel, self).__init__()
        if configs.LOAD:
            print("LOAD MODEL")
            self.policy_net = torch.load(configs.LOAD_PATH,
                                    map_location=torch.device(device))  # map_location=torch.device("cpu"))
            self.target_net = torch.load(configs.LOAD_PATH,
                                    map_location=torch.device(device))  # map_location=torch.device("cpu"))
            self.optimizer = optim.RMSprop(self.policy_net.parameters())
        else:
            print("INITIALIZE MODEL")
            self.policy_net = DQN(17, 17, 6).to(device)
            self.target_net = DQN(17, 17, 6).to(device)
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.optimizer = optim.RMSprop(self.policy_net.parameters())

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def predict_policy(self, state: np.ndarray) -> np.ndarray:
        self.policy_net.eval()
        return self.policy_net(torch.Tensor(state).to(device)).detach().cpu()

    def predict_target(self, state: np.ndarray) -> np.ndarray:
        self.target_net.eval()
        return self.target_net(torch.Tensor(state).to(device)).detach().cpu()

    def fit(self, memory):
        if len(memory) < configs.BATCH_SIZE:
            return
        transitions = random.sample(memory, configs.BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch[:,None])

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(configs.BATCH_SIZE, device=device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * configs.GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()


class DQN(nn.Module):

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(5, 16, kernel_size=(3,3), stride=(2,2))
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3,3), stride=(2,2))
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=(3,3), stride=(2,2))
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 3, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))

