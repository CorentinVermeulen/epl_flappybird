import math
import random
import time
from collections import namedtuple, deque
from itertools import count
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter  # tensorboard --logdir /Users/corentinvrmln/Desktop/memoire/flappybird/repo/DQN/runs/DQN
from flappy_bird_gym.envs import FlappyBirdEnvRGB
from DQN_agent_simple import DQNAgent_simple, ReplayMemory

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

import pickle


def show_state(out, title):
    plt.title(title)
    plt.imshow(out)
    plt.show()

def processFrame(frame):
    frame = frame[55:288,0:400] #crop image
    frame = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY) #convert image to black and white
    frame = cv2.resize(frame, (30, 30))
    frame = cv2.filter2D(frame, -1, np.array([[-1,-1,-1], [-1, 9,-1],[-1,-1,-1]]))
    frame = frame.astype(np.float64)/255.0
    return torch.tensor(frame, dtype=torch.float32).unsqueeze(0).unsqueeze(0)


class DQN_rgb(nn.Module):

    def __init__(self, n_actions):
        super(DQN_rgb, self).__init__()

        self.conv_part = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear_part = nn.Sequential(
            nn.Linear(64, 512),
            nn.ReLU(),
            nn.Linear(512,  n_actions)
        )

    def test(self,x):
        print(x.shape)
        x = self.conv_part(x)
        print(x.shape)
        x = x.view(-1, 64)
        print(x.shape)
        x = self.linear_part(x)
        print(x.shape)
        return x

    def forward(self, x):
        x = self.conv_part(x)
        x = x.view(-1, 32)
        x = self.linear_part(x)
        return x

    def log_weights(self, writer, epoch):
        for name, param in self.named_parameters():
            writer.add_histogram(name, param, epoch)

class DQN_agent_rgb(DQNAgent_simple):
    def __init__(self, env, hyperparameters):
        super(DQN_agent_rgb, self).__init__(env, hyperparameters)
        self.type="rgb"

    def reset(self):
        # Policy and Target net
        self.policy_net = DQN_rgb(self.n_actions)
        self.target_net = DQN_rgb(self.n_actions)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        # Memory
        self.memory = ReplayMemory(self.MEMORY_SIZE)
        # Optimizer
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.LR, amsgrad=True)
        # Training variables
        self.step_done = 0
        self.eps_threshold = self.EPS_START

    def set_nets(self, path):
        # Create nets
        self.policy_net = DQN_rgb(self.n_actions)
        self.target_net = DQN_rgb(self.n_actions)

        # Load nets
        self.policy_net.load_state_dict(torch.load(path + "/policy_net.pt"))
        self.target_net.load_state_dict(self.policy_net.state_dict())
        # Self.optimizer
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.LR, amsgrad=True)

    def _process_state(self, state):
        return processFrame(state)



t=time.perf_counter()
env = FlappyBirdEnvRGB()

hparams = {"EPOCHS": 1500, "BATCH_SIZE": 64, "EPS_DECAY": 4000}
qdnAgent = DQN_agent_rgb(env, hparams)
qdnAgent.train()

# batch_size = 128
# image_size = (30, 30)
# test_images = np.random.rand(batch_size, 1, image_size[0], image_size[1])
# test_batch = torch.tensor(test_images, dtype=torch.float32)
# qdnAgent.policy_net.test(test_batch)

# # Test forward pass
# print("Test forward pass")
# state = processFrame(env.reset())
# print(state.shape)
# qdnAgent.policy_net(state).max(1)[1].view(1, 1)