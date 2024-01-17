import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Actor(nn.Module):
    """
    implementation of actor network mu
    2-layer mlp with tanh output layer
    """

    def __init__(self, state_dim, action_dim, hidden_size1, hidden_size2, ctrl_range):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, action_dim)

        self.ctrl_range = nn.Parameter(torch.Tensor(ctrl_range), requires_grad=False)

    def forward(self, state):

        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = self.ctrl_range * x

        return x

class Critic_Orig(nn.Module):
    """
    implementation of critic network Q(s, a)
    2 layer mlp
    """
    def __init__(self, state_dim, action_dim, acctuator, device):
        super(Critic_Orig, self).__init__()
        hidden_size1 = 256
        hidden_size2 = 256
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

class Critic_NN1(nn.Module):
    """
    Implementation of critic network Q(s, a)
    """
    def __init__(self, state_dim, action_dim, acctuator, device):
        super(Critic_NN1, self).__init__()
        self.device = device
        self.acctuator =  acctuator
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.layer1 = nn.Sequential(
            nn.Conv1d(2, 4, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm1d(4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size = 2, stride = 2))
        self.fc = nn.Linear(200, 128)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(128, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)

    def forward(self, state, action):
        acc = torch.matmul(action[:,[0]],torch.tensor(self.acctuator[0,:], dtype=torch.float).unsqueeze(0).to(self.device))
        for i in range(1,len(self.acctuator)):
          acc = acc + torch.matmul(action[:,[i]],torch.tensor(self.acctuator[i,:], dtype=torch.float).unsqueeze(0).to(self.device))        
        x = torch.stack((state,acc),1)
        out = self.layer1(x)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)

        return out

class Critic_NN2(nn.Module):
    """
    Classical implementation of LeNet fitted to batch_size,2,33,33, one channel for state and control.
    """
    def __init__(self, state_dim, action_dim, acctuator, resortIndex, device):
        super(Critic_NN2, self).__init__()
        self.device = device
        self.acctuator =  acctuator
        self.resortIndex = torch.tensor(resortIndex).view(1, state_dim)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.layer1 = nn.Sequential(
            nn.Conv2d(2, 6, kernel_size=6, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.fc = nn.Linear(400, 120)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(84, 1)

    def forward(self, state, action):
        acc = torch.matmul(action[:,[0]],torch.tensor(self.acctuator[0,:], dtype=torch.float).unsqueeze(0).to(self.device))
        for i in range(1,len(self.acctuator)):
          acc = acc + torch.matmul(action[:,[i]],torch.tensor(self.acctuator[i,:], dtype=torch.float).unsqueeze(0).to(self.device))
        y = torch.reshape(state[:,self.resortIndex], (state.size(0),int(np.sqrt(self.state_dim)), int(np.sqrt(self.state_dim))))
        u = torch.reshape(acc[:,self.resortIndex], (state.size(0),int(np.sqrt(self.state_dim)), int(np.sqrt(self.state_dim))))
        x = torch.stack((y,u),1)
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out