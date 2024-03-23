import math
import random
import time
import datetime
from pathlib import Path
import pickle
from collections import namedtuple, deque
from itertools import count
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import \
    SummaryWriter  # tensorboard --logdir /Users/corentinvrmln/Desktop/memoire/flappybird/repo/DQN/runs/DQN
from agents.utils import get_kpi, running_mean

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

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
        """file must be a .pkl file"""

        with open(file, 'rb') as fichier:
            self.memory = pickle.load(fichier)

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions, sizes=[1,2,3,4], name=None):
        self.name = name
        super(DQN, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(n_observations, sizes[0]),
            nn.ReLU(),
            nn.Linear(sizes[0], sizes[1]),
            nn.ReLU(),
            nn.Linear(sizes[1], sizes[2]),
            nn.ReLU(),
            nn.Linear(sizes[2], sizes[3]),
            nn.ReLU(),
            nn.Linear(sizes[3], n_actions)
        )

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[nothing0exp,jump0exp]...]).
    def forward(self, x):
        x = self.layers(x)
        return x

    def log_weights(self, writer, epoch):
        for name, param in self.named_parameters():
            writer.add_histogram(name, param, epoch)

class DQNAgent_simple_cuda():
    def __init__(self, env, hyperparameters, root_path= "runs/default/"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self.device = torch.device("mps")
        self.env = env
        self.n_actions = env.action_space.n  # Get number of actions from gym action space
        self.n_observations = len(env.reset())  # Get the number of state observations
        self.set_hyperparameters(hyperparameters)
        self.reset()
        self.root_path = root_path
        self.training_path = None

    def print_device(self):
        print(f"Running on {self.device} device")

    def reset(self, name='network'):
        # Policy and Target net
        self.policy_net = DQN(self.n_observations, self.n_actions, self.LAYER_SIZES, name).to(self.device)
        self.target_net = DQN(self.n_observations, self.n_actions, self.LAYER_SIZES, name+"_target").to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # Memory
        self.memory = ReplayMemory(self.MEMORY_SIZE)

        # Optimizer
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.LR, amsgrad=True)

        # Training variables
        self.step_done = 0
        self.eps_threshold = self.EPS_START

    def set_hyperparameters(self, hyperparameters):
        self.EPOCHS = hyperparameters.get('EPOCHS', 1000)
        self.BATCH_SIZE = hyperparameters.get('BATCH_SIZE', 256)
        self.LR = hyperparameters.get('LR', 1e-4)  # the learning rate of the ``AdamW`` optimizer

        self.MEMORY_SIZE = hyperparameters.get('MEMORY_SIZE', 100000)
        self.GAMMA = hyperparameters.get('GAMMA', 0.95)  # the discount factor as mentioned in the previous section
        self.EPS_START = hyperparameters.get('EPS_STAR', 0.9)  # the starting value of epsilon
        self.EPS_END = hyperparameters.get('EPS_END', 0.001)  # the final value of epsilon
        self.EPS_DECAY = hyperparameters.get('EPS_DECAY', 2000)  # higher means a slower decay
        self.TAU = hyperparameters.get('TAU', 0.01)  # the update rate of the target network
        self.LAYER_SIZES = hyperparameters.get('layer_sizes', [256, 256, 256, 256])
        self.UPDATE_TARGETNET_RATE = hyperparameters.get('UPDATE_TARGETNET_RATE', 3)

    def train(self, name=None, print_progress=False):

        self._set_training_id(name)

        best_score = 0
        best_duration = 0
        scores = []
        durations = []
        losses = []

        for i_episode in range(self.EPOCHS):
            episode_reward = 0
            episode_loss = 0
            state = self.env.reset()
            state = self._process_state(state)
            for t in count():
                # Select and perform an action
                action = self._select_action(state)
                # Observe new state and reward
                observation, reward, done, info = self.env.step(action.item())
                observation = self._process_state(observation)
                reward = torch.tensor([reward], device=self.device)
                episode_reward += reward.item()
                next_state = None if done else observation
                # Push to memory
                self.memory.push(state, action.to(self.device), next_state, reward)
                # Move to the next state
                state = next_state
                # Perform one step of the optimization
                loss = self._optimize_model()
                if loss:
                    episode_loss += loss.item()

                if done:
                    train_score = self.env._game.score
                    scores.append(train_score)
                    durations.append(t + 1)
                    losses.append(episode_loss)

                    best_score = max(train_score, best_score)

                    if t+1 >= best_duration:
                        best_duration = t+1
                        self._save_agent()

                        if train_score > 1:
                            self._save_agent()
                    if print_progress:
                        print(f"\r{name if name else ''} - "
                              f"[{i_episode + 1}/{self.EPOCHS}]"
                              f"\tD: {t + 1} (D* {best_duration})"
                              f"\tS: {train_score} (S* {best_score})"
                              f"\tEPS:{self.eps_threshold} , last 100 d_mean {np.mean(durations[-100:]):.2f}",end='')
                    break
            if (i_episode % 200 == 0 and i_episode > 0) or i_episode == self.EPOCHS - 1:
                self._make_end_plot(durations, losses)
                self._save_results(name, scores, durations, losses)

        return scores, durations

    def _process_state(self, state):
        ### TO CHNAGE FOR RGB ###
        return torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

    def _select_action(self, state):
        # update eps
        if self.eps_threshold > self.EPS_END:
            self.eps_threshold = round(
                self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * self.step_done / self.EPS_DECAY), 3)

        self.step_done += 1
        # Select actions
        sample = random.random()
        # best Action
        if sample > self.eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        # random Action
        else:
            return torch.tensor([[self.env.action_space.sample()]], dtype=torch.long, device=self.device)

    def _optimize_model(self):
        if len(self.memory) < self.BATCH_SIZE:
            return

        transitions = self.memory.sample(self.BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        # Actual Q(s_t,a)
        state_batch = torch.cat([s for s in batch.state]).to(self.device)
        action_batch = torch.cat([a for a in batch.action]).to(self.device)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)  # Sort un tensor contenant Q pour l'action choisie dans action_batch = Q(s_t,a)

        # Compute V(s_{t+1}) for all next states. (Target Q value)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,batch.next_state)), dtype=torch.bool, device=self.device)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).to(self.device)
        reward_batch = torch.cat([b for b in batch.reward]).to(self.device)

        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
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

        # Soft update of the target network's weights (θ′ ← τ θ + (1 −τ )θ′)
        if self.step_done % self.UPDATE_TARGETNET_RATE == 0:
            target_state_dict = self.target_net.state_dict()
            policy_state_dict = self.policy_net.state_dict()

            for key in policy_state_dict:
                target_state_dict[key] = self.TAU * policy_state_dict[key] + (1 - self.TAU) * target_state_dict[key]
            self.target_net.load_state_dict(target_state_dict)

        return loss

    def _save_agent(self):
        name = self.training_path + "/policy_net"
        torch.save(self.policy_net.state_dict(), name + '.pt')

    def _make_end_plot(self, durations, losses):
        N = 50

        plt.figure(figsize=(12, 12))

        plt.subplot(3, 1, 1)
        plt.plot(durations, alpha=0.5, label='Durations', color='blue')
        plt.plot(running_mean(durations, N), label=f"Running mean ({N})", color="red")
        plt.ylabel("Durations [frames]")
        plt.title(f"Durations (mean: {np.mean(durations):.2f})")

        plt.subplot(3, 1, 2)
        plt.plot(losses, alpha=0.5, label='Durations', color='blue')
        plt.plot(running_mean(losses, N), label=f"Running mean ({N})", color="red")
        plt.ylabel("Loss [Delta Q value]")
        plt.title(f"Losses")

        avg_dur = np.cumsum(durations) / np.arange(1, len(durations) + 1)
        plt.subplot(3, 1, 3)
        plt.plot(avg_dur, label='Average durations', color='blue')
        plt.ylabel("Durations [frames]")
        plt.title(f"Avg durations")
        plt.xlabel("Game played")

        plt.legend()
        plt.tight_layout()
        plt.savefig(self.training_path + f"/plot.png")
        plt.close()

    def _save_results(self, name, scores, durations, loss):
       pd.DataFrame({"scores": scores,
                     "durations": durations,
                     "loss": loss}).to_csv(self.training_path + f"/results_{name}.csv")

       end_dic = self.__dict__
       if scores and durations:
           end_dic['n_episodes'] = len(scores)
           end_dic['best_score'] = np.max(scores)
           end_dic['best_duration'] = np.max(durations)
           end_dic['n_to_max'] = scores.index(max(scores))
           kpi = get_kpi(scores, durations)
           end_dic.update(kpi)

       with open(self.training_path + "/param_log.txt", 'w') as f:
           for k, v in end_dic.items():
               f.write(f"{k}: {v}\n")

    def _set_training_id(self, name=None):
        id = datetime.datetime.now().strftime("%d%m_%H%M%S")
        if name:
            id = f"{name}_{id}/"
        self.training_path = self.root_path + id + '/'
        Path(self.training_path).mkdir(parents=True)

    def update_env(self, dic):
        self.env.update_params(dic)
