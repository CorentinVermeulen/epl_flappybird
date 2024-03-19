import random, datetime, os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import math
from itertools import count

import pandas as pd
import torch
from torch import nn
# from torchvision import transforms as T
from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage
# from collections import deque
# from gym.wrappers import FrameStack
from agents.utils import running_mean

from flappy_bird_gym.envs.custom_env_simple import CustomEnvSimple as FlappyBirdEnv


class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.policy = self._buid_model(input_dim, output_dim)

        # Target Network is a frozen copy of the policy network
        self.target = self._buid_model(input_dim, output_dim)
        self.target.load_state_dict(self.policy.state_dict())
        for p in self.target.parameters():
            p.requires_grad = False

    def __str__(self):
        s = 'QNetworks (Policy and Target networks):\n'
        for layer in self.policy:
            s += f'\t{layer}\n'
        return s

    def forward(self, x, model: str = "policy"):
        if model == "policy":
            return self.policy(x)
        elif model == "target":
            return self.target(x)
        else:
            raise ValueError(f"Unknown model {model}")

    def _buid_model(self, input_dim, output_dim):
        return nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )


class Agent():
    def __init__(self, env, hyperparameters={}, root_path=None):
        self.env = env
        self.n_actions = env.action_space.n  # Get number of actions from gym action space
        self.n_observations = len(env.reset())  # Get the number of state observations
        self.root_path = root_path if root_path else "default/"
        self.device = 'cpu'
        # self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.device = 'mps'
        # Set nets and optimizer
        self.qnet = QNetwork(self.n_observations, self.n_actions)
        self.qnet = self.qnet.to(device=self.device)
        self.optimizer = torch.optim.Adam(self.qnet.parameters(), lr=0.00025)
        self.loss_fn = torch.nn.SmoothL1Loss()

        # Set hyperparameters
        self.set_hyperparameters(hyperparameters)

        # Reset
        self.reset()

    def set_hyperparameters(self, hyperparameters):
        self.eps_threshold = 1
        self.EPS_DECAY = 2000
        self.EPS_START = 1
        self.EPS_END = 0.001

        self.save_every = 5e5  # no. of experiences between saving Mario Net

        self.MEMMORY_SIZE = 100_000
        self.BATCH_SIZE = 256
        self.GAMES_TO_PLAY = 1000

        self.TAU = 0.01
        self.GAMMA = 0.95
        self.BURNING = 0  # min. experiences before training
        self.learn_every = 1  # no. of experiences between updates to Q_online
        self.sync_every = 3  # no. of experiences between Q_target & Q_online sync

        for param_name, param_value in hyperparameters.items():
            if hasattr(self, param_name):
                setattr(self, param_name, param_value)
            else:
                print(f"Warning: Attribute '{param_name}' does not exist in this object.")

    def reset(self):
        self.step_done = 0
        self.memory = TensorDictReplayBuffer(storage=LazyMemmapStorage(self.MEMMORY_SIZE, device=torch.device("cpu")))

    def select_action(self, state):
        """ Select an action from the input state """
        # Exploitation = Best action
        if random.random() > self.eps_threshold:
            state = state[0].__array__() if isinstance(state, tuple) else state.__array__()
            state = torch.tensor(state, device=self.device).unsqueeze(0)
            with torch.no_grad():
                action = self.qnet(state, "policy").max(1)[1].view(1, 1)

        # Exploration = Random action
        else:
            action = torch.tensor([[self.env.action_space.sample()]], dtype=torch.long, device=self.device)

        # Update exploration rate
        if self.eps_threshold > self.EPS_END:
            self.eps_threshold = round(
                self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * self.step_done / self.EPS_DECAY), 3)
        # Update step
        self.step_done += 1

        return action

    def optimize_model(self):
        """ Optimize the policy network with batch of experience"""
        if self.step_done % self.sync_every == 0:
            self.sync_target()

        if self.step_done % self.save_every == 0:
            self.save()

        if self.step_done < self.BURNING:
            return None, None

        if self.step_done % self.learn_every != 0:
            return None, None

        state, action, reward, next_state, done = self.recall()
        q_policy_est = self.q_estimated(state, action)
        q_target_est = self.q_target(reward, next_state, done).unsqueeze(1)
        loss = self.update_Q_policy(q_policy_est, q_target_est)
        return (q_policy_est.mean().item(), loss)

    def cache(self, state, action, reward, next_state, done):
        """ Cache the experience in the replay buffer"""

        state = torch.tensor(state)
        action = torch.tensor([action])
        reward = torch.tensor([reward])
        next_state = torch.tensor(next_state)
        done = torch.tensor([done])

        self.memory.add(TensorDict({"state": state, "action": action,
                                    "reward": reward, "next_state": next_state, "done": done},
                                   batch_size=[]
                                   )
                        )

    def recall(self):
        """ Sample batch from replay buffer"""
        batch = self.memory.sample(self.BATCH_SIZE).to(self.device)
        state, next_state, action, reward, done = (batch.get(key) for key in
                                                   ("state", "next_state", "action", "reward", "done"))
        return state, action, reward.squeeze(), next_state, done.squeeze()

    def q_estimated(self, states_batch, actions_batch):
        # Sort un tensor contenant Q pour l'action choisie dans action_batch = Q(s_t,a)
        current_Q = self.qnet(states_batch, "policy").gather(1, actions_batch)
        return current_Q

    @torch.no_grad()
    def q_target(self, reward_batch, next_state_batch, done_batch):

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, next_state_batch)), dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in tuple(s.unsqueeze(0) for s in next_state_batch) if s is not None])
        # Compute V(s_{t+1}) for all next states.
        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.qnet(non_final_next_states.to(self.device), "target").max(1)[0]
        # Expected Q(s_t,a)
        return (next_state_values * self.GAMMA) + reward_batch

    def update_Q_policy(self, q_policy, q_target):
        loss = self.loss_fn(q_policy, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_target(self):
        target_state_dict = self.qnet.policy.state_dict()
        policy_state_dict = self.qnet.target.state_dict()

        for key in policy_state_dict:
            target_state_dict[key] = self.TAU * policy_state_dict[key] + (1 - self.TAU) * target_state_dict[key]
            self.qnet.target.load_state_dict(target_state_dict)

    def save(self, name=None):
        name = "model" if not name else "model_" + name
        path = self.training_path + f"{name}.chkpt"
        torch.save(self.qnet.policy.state_dict(), path)
        # print(f"Model saved to {path} at step {self.step_done}")

    def set_training_id(self, name=None):
        id = datetime.datetime.now().strftime("%d%m_%H%M%S")
        if name:
            id = f"{name}_{id}/"
        self.training_path = self.root_path + id + '/'
        Path(self.training_path).mkdir(parents=True)

    def train(self, name=None):
        self.set_training_id(name)
        best_score = 0
        scores = []
        durations = []
        losses = []
        qs = []
        for i in range(self.GAMES_TO_PLAY):
            state = self.env.reset()
            total_loss = 0
            total_qs = 0
            for t in count():
                action = self.select_action(state)
                next_state, reward, done, info = self.env.step(action)
                self.cache(state, action, reward, next_state, done)
                q, loss = self.optimize_model()

                if loss:
                    total_loss += loss
                if q:
                    total_qs += q

                state = next_state
                if done:
                    scores.append(info["score"])
                    durations.append(t + 1)
                    losses.append(total_loss)
                    qs.append(total_qs)

                    if info["score"] >= best_score:
                        best_score = info["score"]
                        if info['score'] > 2:
                            self.save()

                    print(f"\r[{i + 1:^4}/{self.GAMES_TO_PLAY}] d:{t} "
                          f"\tS*: {best_score} "
                          f"\teps: {self.eps_threshold}", end="")
                    break

            if i % 100 == 0:
                self.make_plot(durations, losses, qs)

        self.make_plot(durations, losses, qs)
        pd.DataFrame({"scores": scores,
                      "durations": durations,
                      "losses": losses,
                      "qs": qs}).to_csv(self.training_path + "training.csv")

    def make_plot(self, durations, losses, qs):
        plt.figure(figsize=(12, 12))
        plt.subplot(3, 1, 1)
        plt.plot(durations, alpha=0.5, color='blue')
        plt.plot(running_mean(durations, 50), color="red")
        plt.ylabel('duration')
        plt.title("Durations")

        plt.subplot(3, 1, 2)
        plt.plot(losses, alpha=0.5, color='blue')
        plt.plot(running_mean(losses, 50), color="red")
        plt.ylabel('Loss')
        plt.title("Losses")

        plt.subplot(3, 1, 3)
        plt.plot(qs, alpha=0.5, color='blue')
        plt.plot(running_mean(qs, 50), color="red")
        plt.title("Qs")
        plt.ylabel('Q')

        plt.xlabel('Game played')
        plt.tight_layout()

        plt.savefig(self.training_path + "plot.png")
        plt.close()

    def update_env(self, dic):
        self.env.update_params(dic)


env = FlappyBirdEnv()
env.obs_var = ['player_x', 'player_y', 'pipe_center_x', 'pipe_center_y', 'v_dist', 'h_dist', 'player_vel_y']

hparams = {'GAMES_TO_PLAY': 1000}
for rd in [True, False]:
    env_params = {"PLAYER_FLAP_ACC": -6, "PLAYER_ACC_Y": 1, "pipes_are_random": rd}
    for i in range(10):
        agent = Agent(env=env, hyperparameters=hparams)
        agent.update_env(env_params)
        agent.train(name=f"{rd}_{i}")
