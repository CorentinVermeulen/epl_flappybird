import csv
import math
import random
import os
import time

import matplotlib.pyplot as plt
import numpy as np
from IPython import display

from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from flappy_bird_gym.envs.custom_env_simple import CustomEnvSimple as FlappyBirdEnv
from AbstractAgent import AbstractAgent

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

import pickle


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
        self.capacity = capacity

    def __len__(self):
        return len(self.memory)

    def __str__(self):
        return f"ReplayMemory with capacity {self.capacity} ({len(self.memory)} used elements)"
    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def save(self, file):
        with open(file + '.pkl', 'wb') as fichier:
            pickle.dump(self.memory, fichier)

    def load(self, file):
        with open(file + '.pkl', 'rb') as fichier:
            self.memory = pickle.load(fichier)


class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 64)
        self.layer2 = nn.Linear(64, 128)
        self.layer3 = nn.Linear(128, 256)
        self.layer4 = nn.Linear(256, 256)
        self.output = nn.Linear(256, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[nothing0exp,jump0exp]...]).
    def forward(self, x):
        x = F.relu6(self.layer1(x))
        x = F.relu6(self.layer2(x))
        x = F.relu6(self.layer3(x))
        x = F.relu6(self.layer4(x))
        return self.output(x)

    def log_weights(self, writer, epoch):
        for name, param in self.named_parameters():
            writer.add_histogram(name, param, epoch)


class DQNAgent(AbstractAgent):
    def __init__(self, env, hyperparameters):
        self.env = env
        self.n_actions = env.action_space.n  # Get number of actions from gym action space
        self.n_observations = len(env.reset())  # Get the number of state observations
        self.device = 'mps'
        self.set_hyperparameters(hyperparameters)

        self.policy_net = DQN(self.n_observations, self.n_actions)
        self.target_net = DQN(self.n_observations, self.n_actions)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.LR, amsgrad=True)

        self.memory = ReplayMemory(self.MEMORY_SIZE)

        self.step_done = 0
        self.eps_threshold = self.EPS_START

        self.writer = None


    def set_hyperparameters(self, hyperparameters):
        self.EPOCHS = hyperparameters.get('EPOCHS', 2000)
        self.BATCH_SIZE = hyperparameters.get('BATCH_SIZE', 256)
        self.LR = hyperparameters.get('LR', 1e-4)  # the learning rate of the ``AdamW`` optimizer

        self.MEMORY_SIZE = hyperparameters.get('MEMORY_SIZE', 100000)
        self.GAMMA = hyperparameters.get('GAMMA', 0.99)  # the discount factor as mentioned in the previous section
        self.EPS_START = hyperparameters.get('EPS_STAR', 0.9)  # the starting value of epsilon
        self.EPS_END = hyperparameters.get('EPS_END', 0.01)  # the final value of epsilon
        self.EPS_DECAY = hyperparameters.get('EPS_DECAY',2000)  # higher means a slower decay
        self.TAU = hyperparameters.get('TAU', 0.005)  # the update rate of the target network

    def train(self):
        id_training = 'runs/DQN/' + time.strftime("%d%m-%H%M%S")
        self.writer = SummaryWriter(id_training)
        desc = ""
        for k, v in self.__dict__.items():
            desc += f"{k}: {v}\n"
        self.writer.add_text('Hyperparameters', desc)
        self.writer.add_graph(self.policy_net, torch.zeros(1, self.n_observations))

        best_score = 0
        best_duration = 0
        scores = []
        durations = []

        for i_episode in range(self.EPOCHS):
            # Initialize the environment and state
            total_reward = 0
            state = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

            # Start playing and learning
            for t in count():
                action = self._select_action(state)
                observation, reward, terminated, _ = self.env.step(action.item())
                reward = torch.tensor([reward])
                total_reward += reward.item()

                done = terminated
                next_state = None if terminated else torch.tensor(observation, dtype=torch.float32).unsqueeze(0)

                # Store transition in memory
                self.memory.push(state, action, next_state, reward)

                state = next_state

                self._optimize_model()

                # Soft update of the target network's weights (θ′ ← τ θ + (1 −τ )θ′)
                target_state_dict = self.target_net.state_dict()
                policy_state_dict = self.policy_net.state_dict()

                for key in policy_state_dict:
                    target_state_dict[key] = self.TAU * policy_state_dict[key] + (1 - self.TAU) * target_state_dict[key]
                self.target_net.load_state_dict(target_state_dict)

                if done:
                    train_score = self.env._game.score
                    scores.append(train_score)
                    durations.append(t+1)
                    dic = {'memory_size': len(self.memory), 'train_score': train_score, 'train_reward': total_reward, 'duration': t+1}

                    best_score = max(train_score, best_score)
                    if t+1 > best_duration:
                        best_duration = t+1
                        self.policy_net.log_weights(self.writer, i_episode)

                        if train_score > 2:
                            name = id_training + f"/E{i_episode}_D{t+1}_S{train_score}"
                            torch.save(self.policy_net.state_dict(), name+'.pt')
                            self.memory.save(name+'_memory')

                    self._write_stats(i_episode, **dic)

                    print(f"\r{i_episode+1}/{self.EPOCHS} - D*: {best_duration} S*: {best_score} "
                          f"\tcD: {t+1} cS: {train_score} "
                          f"\tEPS:{self.eps_threshold}"
                          ,end='')

                    break

        end_dic = self.__dict__
        end_dic['best_score'] = best_score
        end_dic['best_duration'] = best_duration

        self._log_agent(id_training, **end_dic)
        self.writer.close()
        self._make_plot(scores, durations, id_training + '/plot.png')
    def retrain(self, **kwargs):
        pass

    def test(self):
        with torch.no_grad():
            state = self.env().reset()
            for t in count():
                action = self.policy_net(state).max(1)[1].view(1, 1)
                observation, reward, done, _ = self.env().step(action.item())
                state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
                if done:
                    next_state = None
                    return {'score': self.env().score, 'duration': t+1}
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
                state = next_state

    def _select_action(self, state):
        sample = random.random()
        if self.eps_threshold > self.EPS_END:
            self.eps_threshold = round(self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * self.step_done / self.EPS_DECAY), 3)

        self.step_done += 1
        # Best Action
        if sample > self.eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        # Random Action
        else:
            return torch.tensor([[self.env.action_space.sample()]], dtype=torch.long)

    def _optimize_model(self):
        if len(self.memory) < self.BATCH_SIZE:
            return

        transitions = self.memory.sample(self.BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Actual Q(s_t,a)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch) # Sort un tensor contenant Q pour l'action choisie dans action_batch = Q(s_t,a)

        # Compute V(s_{t+1}) for all next states.
        next_state_values = torch.zeros(self.BATCH_SIZE)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
        # Expected Q(s_t,a)
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # Compute loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()


    def _write_stats(self, i_episode, **kwargs):
        for key, value in kwargs.items():
            self.writer.add_scalar(key, value, i_episode)

    def _log_agent(self, name, **kwargs):
        with open(name+"/param_log.txt", 'w') as f:
            for k,v in kwargs.items():
                f.write(f"{k}: {v}\n")

    def _make_plot(self, scores, durations, name):
        plt.figure(figsize=(10, 5))
        plt.subplot(2, 1, 1)
        plt.plot(scores)
        plt.title('Scores')

        plt.subplot(2, 1, 2)
        plt.plot(durations)
        plt.title('Durations')
        plt.xlabel('Epoch')

        plt.savefig(name)
        plt.close()

OBS_VAR = ['player_x', 'player_y', 'pipe_center_x', 'pipe_center_y', 'v_dist', 'h_dist', 'player_vel_y']
env = FlappyBirdEnv(obs_var=OBS_VAR)
qdnAgent = DQNAgent(env, {"EPOCHS":2000, "BATCH_SIZE": 64})
qdnAgent.train()







