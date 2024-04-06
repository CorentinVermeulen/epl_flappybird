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
from agent_simple import AgentSimple, ReplayMemory
from utils import get_kpi, running_mean, HParams

from torchsummary import summary

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
    frame = cv2.resize(frame, (84, 84))
    #frame = cv2.filter2D(frame, -1, np.array([[-1,-1,-1], [-1, 9,-1],[-1,-1,-1]]))
    frame = frame.astype(np.float64)/255.0
    return torch.tensor(frame, dtype=torch.float32).unsqueeze(0).unsqueeze(0)


class DQN_rgb(nn.Module):

    def __init__(self, n_channels, n_actions):
        super(DQN_rgb, self).__init__()

        self.conv_part = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.linear_part = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512,  n_actions)
        )

    def test(self,x):

        print(f"Input shape: {x.shape}")
        x = self.conv_part(x)
        print(f"After conv part: {x.shape}")
        x = self.linear_part(x)
        print(f"Output shape: {x.shape}")
        return x

    def forward(self, x):
        x = self.conv_part(x)
        x = self.linear_part(x)
        return x

    def log_weights(self, writer, epoch):
        for name, param in self.named_parameters():
            writer.add_histogram(name, param, epoch)

class AgentRGB(AgentSimple):
    def __init__(self, env, hyperparameters):
        super(AgentRGB, self).__init__(env, hyperparameters)
        self.type="rgb"

    def reset(self, name='Net'):
        # Policy and Target net
        self.policy_net = DQN_rgb(self.n_observations, self.n_actions).to(
            self.device)
        self.target_net = DQN_rgb(self.n_observations, self.n_actions).to(
            self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # Memory
        self.memory = ReplayMemory(self.hparams.MEMORY_SIZE)

        # Optimizer
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.hparams.LR, amsgrad=True)

        # Training variables
        self.step_done = 0
        self.eps_threshold = self.hparams.EPS_START

    def _process_state(self, state):
        return processFrame(state)


if __name__ == "__main__":
    baseline_HP = {"EPOCHS": 1000,
                   "BATCH_SIZE": 256,
                   "LR": 1e-4,
                   "MEMORY_SIZE": 100000,
                   "GAMMA": 0.95,
                   "EPS_START": 0.9,
                   "EPS_END": 0.001,
                   "EPS_DECAY": 2000,
                   "TAU": 0.01,
                   "LAYER_SIZES": [256, 256, 256, 256],
                   "UPDATE_TARGETNET_RATE": 3}
    hparams = HParams(baseline_HP)

    env = FlappyBirdEnvRGB()
    agent = AgentRGB(env, hparams)
    agent.train(show_progress=True)

