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
from torchsummary import summary

from torch.utils.tensorboard import \
    SummaryWriter  # tensorboard --logdir /Users/corentinvrmln/Desktop/memoire/flappybird/repo/DQN/runs/DQN
from utils import get_kpi, running_mean, HParams

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

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions, sizes=None, name=None):
        super(DQN, self).__init__()
        if sizes is None:
            sizes = [1, 2, 3, 4]

        self.name = name
        self.n_observations = n_observations
        self.n_actions = n_actions

        layers = [n_observations] + sizes

        layer_list = []
        for i in range(1, len(layers)):
            layer_list.append(nn.Linear(layers[i - 1], layers[i]))
            layer_list.append(nn.ReLU())
        layer_list.append(nn.Linear(layers[-1], n_actions))

        self.layers = nn.Sequential(*layer_list)

    def forward(self, x):
        x = self.layers(x)
        return x

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def __str__(self):
        s = super(DQN, self).__str__()
        layer_sizes = []
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                layer_sizes.append((layer.in_features, layer.out_features))
        layer_str = ' -> '.join(f'{in_f}' for in_f, out_f in layer_sizes)
        layer_str += f' -> {self.n_actions}'
        s += f"\nLayers summary: {layer_str}"
        s += f"\nTrainable parameters: {self.count_parameters():,d}"
        return s

class AgentSimple():
    def __init__(self, env, hyperparameters, root_path="runs/default/"):
        self.training_time = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("mps")
        self.env = env
        self.n_actions = env.action_space.n  # Get number of actions from gym action space
        self.n_observations = len(env.reset())  # Get the number of state observations
        self.hparams = hyperparameters
        self.reset()
        self.root_path = root_path
        self.training_path = None

    def print_device(self):
        print(f"Running on : {self.device} device")

    def reset(self, name='Net'):
        # Policy and Target net
        self.policy_net = DQN(self.n_observations, self.n_actions, self.hparams.LAYER_SIZES, name+ "_policy").to(self.device)
        self.target_net = DQN(self.n_observations, self.n_actions, self.hparams.LAYER_SIZES, name + "_target").to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # Memory
        self.memory = ReplayMemory(self.hparams.MEMORY_SIZE)

        # Optimizer
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.hparams.LR, amsgrad=True)

        # Training variables
        self.step_done = 0
        self.eps_threshold = self.hparams.EPS_START

    def update_hyperparameters(self, hyperparameters):
        self.hparams = hyperparameters
        print(f"Hyperparameters updated : {self.hparams}")

    def train(self, name=None, show_progress=False):
        t = time.perf_counter()
        self.set_training_id(name)

        best_score = 0
        best_duration = 0
        scores = []
        durations = []
        losses = []

        for i_episode in range(self.hparams.EPOCHS):
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

                    if t + 1 >= best_duration:
                        best_duration = t + 1
                        self._save_agent()

                        if train_score > 1:
                            self._save_agent()
                    if show_progress:
                        print(f"\r{name if name else ''} - "
                              f"[{i_episode + 1}/{self.hparams.EPOCHS}]"
                              f"\tD: {t + 1} (D* {best_duration})"
                              f"\tS: {train_score} (S* {best_score})"
                              f"\tEPS:{self.eps_threshold} , last 100 d_mean {np.mean(durations[-100:]):.2f}",end='')
                    break
            if (i_episode % 200 == 0 and i_episode > 0) or i_episode == self.hparams.EPOCHS - 1:
                self._make_end_plot(durations, losses, name)
                time_final = time.perf_counter() - t
                self.training_time = f"{divmod(time_final, 60)[0]:02}:{divmod(time_final, 60)[1]:02}"
                self._save_results(name, scores, durations, losses)

        return scores, durations

    def _select_action(self, state):
        # update eps
        if self.eps_threshold > self.hparams.EPS_END:
            self.eps_threshold = round(
                self.hparams.EPS_END + (self.hparams.EPS_START - self.hparams.EPS_END) * math.exp(
                    -1. * self.step_done / self.hparams.EPS_DECAY), 3)

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
        if len(self.memory) < self.hparams.BATCH_SIZE:
            return

        transitions = self.memory.sample(self.hparams.BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        # Actual Q(s_t,a)
        state_batch = torch.cat([s for s in batch.state]).to(self.device)
        action_batch = torch.cat([a for a in batch.action]).to(self.device)
        state_action_values = self.policy_net(state_batch).gather(1,
                                                                  action_batch)  # Sort un tensor contenant Q pour l'action choisie dans action_batch = Q(s_t,a)

        # Compute V(s_{t+1}) for all next states. (Target Q value)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool,
                                      device=self.device)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).to(self.device)
        reward_batch = torch.cat([b for b in batch.reward]).to(self.device)

        next_state_values = torch.zeros(self.hparams.BATCH_SIZE, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
        # Expected Q(s_t,a)
        expected_state_action_values = (next_state_values * self.hparams.GAMMA) + reward_batch

        # Compute loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        # optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        # Soft update of the target network's weights (θ′ ← τ θ + (1 −τ )θ′)
        if self.step_done % self.hparams.UPDATE_TARGETNET_RATE == 0:
            target_state_dict = self.target_net.state_dict()
            policy_state_dict = self.policy_net.state_dict()

            for key in policy_state_dict:
                target_state_dict[key] = self.hparams.TAU * policy_state_dict[key] + (1 - self.hparams.TAU) * \
                                         target_state_dict[key]
            self.target_net.load_state_dict(target_state_dict)

        return loss

    def _save_agent(self):
        name = self.training_path + "/policy_net"
        torch.save(self.policy_net.state_dict(), name + '.pt')

    def _make_end_plot(self, durations, losses, name):
        N = 50

        plt.figure(figsize=(12, 12))
        plt.suptitle(name if name else "")

        plt.subplot(3, 1, 1)
        plt.plot(durations, alpha=0.5, label='Durations', color='blue')
        plt.plot(running_mean(durations, N), label=f"Running mean ({N})", color="red")
        plt.ylabel("Durations [frames]")
        plt.title(f"Durations (mean: {np.mean(durations):.2f})")
        plt.legend()

        avg_dur = np.cumsum(durations) / np.arange(1, len(durations) + 1)
        plt.subplot(3, 1, 2)
        plt.plot(avg_dur, label='Average durations', color='blue')
        plt.ylabel("Durations [frames]")
        plt.title(f"Avg durations")
        plt.legend()

        plt.subplot(3, 1, 3)
        plt.plot(losses, alpha=0.3, label='Loss', color='blue')
        plt.plot(running_mean(losses, N), label=f"Running mean ({N})", color="red")
        plt.ylabel("Loss [Delta Q value]")
        plt.title(f"Losses")


        plt.xlabel("Game played")
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.training_path + f"/plot.png")
        plt.close()

    def _save_results(self, name, scores, durations, loss):
        pd.DataFrame({"scores": scores,
                      "durations": durations,
                      "loss": loss}).to_csv(self.training_path + f"/results_({name}).csv")

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

    def set_training_id(self, name=None):
        id = datetime.datetime.now().strftime("%d%m_%H%M%S")
        if name:
            id = f"{name}_{id}/"
        self.training_path = self.root_path + id + '/'
        Path(self.training_path).mkdir(parents=True)

    def _process_state(self, state):
        ### TO CHNAGE FOR RGB ###
        return torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

    def update_env(self, dic):
        self.env.update_params(dic)

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

    net = DQN(8, 2, [256, 256, 256, 256])
    print(net.count_parameters())

    net = DQN(8, 2, [128, 128, 128, 128, 128, 128], name='2')
    print(net.count_parameters())
    print(net)
