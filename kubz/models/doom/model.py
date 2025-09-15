import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DoomAgent(nn.Module):
    def __init__(self, n_actions=7):
        super(DoomAgent, self).__init__()

        self.conv1 = nn.Conv2d(4, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)

        self.fc_input_dim = 32 * 7 * 7

        self.fc1 = nn.Linear(self.fc_input_dim, 256)
        self.fc2 = nn.Linear(256, n_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.reshape(-1, self.fc_input_dim)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def act(self, state, epsilon=0.0):
        if np.random.random() < epsilon:
            return np.random.randint(7)

        with torch.no_grad():
            state_tensor = (
                torch.FloatTensor(state).unsqueeze(0).to(next(self.parameters()).device)
            )
            q_values = self.forward(state_tensor)
            return q_values.argmax().item()


# TODO: Generate this on the fly (Using whole agent model) - or ask for it to be provided- model slicer
class Conv1Segment(nn.Module):
    def __init__(self):
        super(Conv1Segment, self).__init__()
        self.conv1 = nn.Conv2d(4, 16, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        return F.relu(self.conv1(x))

class Conv2Segment(nn.Module):
    def __init__(self):
        super(Conv2Segment, self).__init__()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return F.relu(self.conv2(x))

class Conv3Segment(nn.Module):
    def __init__(self):
        super(Conv3Segment, self).__init__()
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return F.relu(self.conv3(x))

class FC1Segment(nn.Module):
    def __init__(self):
        super(FC1Segment, self).__init__()
        self.fc_input_dim = 32 * 7 * 7
        self.fc1 = nn.Linear(self.fc_input_dim, 256)

    def forward(self, x):
        x = x.reshape(-1, self.fc_input_dim)
        return F.relu(self.fc1(x))

class FC2Segment(nn.Module):
    def __init__(self, n_actions=7):
        super(FC2Segment, self).__init__()
        self.fc2 = nn.Linear(256, n_actions)

    def forward(self, x):
        return self.fc2(x)

