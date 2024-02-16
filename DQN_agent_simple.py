import math
import random
import time
import pickle
from collections import namedtuple, deque
from itertools import count
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import \
    SummaryWriter  # tensorboard --logdir /Users/corentinvrmln/Desktop/memoire/flappybird/repo/DQN/runs/DQN

UPDATE_TARGETNET_RATE = 10

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

    def __init__(self, n_observations, n_actions, sizes=[64, 128, 256, 256]):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, sizes[0])
        self.layer2 = nn.Linear(sizes[0], sizes[1])
        self.layer3 = nn.Linear(sizes[1], sizes[2])
        self.layer4 = nn.Linear(sizes[2], sizes[3])
        self.output = nn.Linear(sizes[3], n_actions)

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


class DQNAgent_simple():
    def __init__(self, env, hyperparameters):
        self.env = env
        self.n_actions = env.action_space.n  # Get number of actions from gym action space
        self.n_observations = len(env.reset())  # Get the number of state observations
        self.set_hyperparameters(hyperparameters)
        self.reset()
        self.writer = None
        self.type="simple"
        self.root_path = "runs/Comp/"
        self.training_path = None

    def reset(self):
        # Policy and Target net
        self.policy_net = DQN(self.n_observations, self.n_actions, self.LAYER_SIZES)
        self.target_net = DQN(self.n_observations, self.n_actions, self.LAYER_SIZES)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        # Memory
        self.memory = ReplayMemory(self.MEMORY_SIZE)
        # Optimizer
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.LR, amsgrad=True)
        # Training variables
        self.step_done = 0
        self.eps_threshold = self.EPS_START

    def set_root_path(self, path):
        self.root_path = path

    def set_nets(self, path):
        # create nets
        self.policy_net = DQN(self.n_observations, self.n_actions, self.LAYER_SIZES)
        self.target_net = DQN(self.n_observations, self.n_actions, self.LAYER_SIZES)

        # Load nets
        self.policy_net.load_state_dict(torch.load(path + "/policy_net.pt"))
        self.target_net.load_state_dict(self.policy_net.state_dict())
        # Self.optimizer
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.LR, amsgrad=True)

    def set_memory(self, path):
        self.memory = ReplayMemory(self.MEMORY_SIZE)
        self.memory.load(path + "/policy_net_memory.pkl")

    def set_hyperparameters(self, hyperparameters):
        self.EPOCHS = hyperparameters.get('EPOCHS', 2000)
        self.BATCH_SIZE = hyperparameters.get('BATCH_SIZE', 256)
        self.LR = hyperparameters.get('LR', 1e-4)  # the learning rate of the ``AdamW`` optimizer

        self.MEMORY_SIZE = hyperparameters.get('MEMORY_SIZE', 100000)
        self.GAMMA = hyperparameters.get('GAMMA', 0.99)  # the discount factor as mentioned in the previous section
        self.EPS_START = hyperparameters.get('EPS_STAR', 0.9)  # the starting value of epsilon
        self.EPS_END = hyperparameters.get('EPS_END', 0.01)  # the final value of epsilon
        self.EPS_DECAY = hyperparameters.get('EPS_DECAY', 2000)  # higher means a slower decay
        self.TAU = hyperparameters.get('TAU', 0.005)  # the update rate of the target network
        self.LAYER_SIZES = hyperparameters.get('layer_sizes', [64, 128, 256, 256])

    def create_training_path(self):
        t = time.strftime("%d%m-%H%M%S")
        return self.root_path + f"{self.type}_{t}"

    def _process_state(self, state):
        ### TO CHNAGE FOR RGB ###
        return torch.tensor(state, dtype=torch.float32).unsqueeze(0)

    def _select_action(self, state):
        sample = random.random()
        if self.eps_threshold > self.EPS_END:
            self.eps_threshold = round(
                self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * self.step_done / self.EPS_DECAY), 3)

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
        state_action_values = self.policy_net(state_batch).gather(1,
                                                                  action_batch)  # Sort un tensor contenant Q pour l'action choisie dans action_batch = Q(s_t,a)

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

    def _log_agent(self, **kwargs):
        with open(self.training_path + "/param_log.txt", 'w') as f:
            for k, v in kwargs.items():
                f.write(f"{k}: {v}\n")

    def _write_stats(self, i_episode, **kwargs):
        for key, value in kwargs.items():
            self.writer.add_scalar(key, value, i_episode)

    def _save_agent(self):
        name = self.training_path + "/policy_net"
        torch.save(self.policy_net.state_dict(), name + '.pt')
        self.memory.save(name + '_memory')

    def _make_plot(self, scores, durations):
        plt.figure(figsize=(10, 5))
        plt.subplot(2, 1, 1)
        plt.plot(scores)
        plt.title(f"Scores (max: {np.max(scores)})")

        plt.subplot(2, 1, 2)
        plt.plot(durations)
        plt.title(f"Durations (mean: {np.mean(durations)})")
        plt.xlabel('Epoch')

        plt.savefig(self.training_path + f"/plot_S{max(scores)}.png")
        plt.close()

    def train(self, retrain_path=None, retrain_epochs=None, name=None):
        if retrain_path:
            self.training_path = retrain_path + f"{'_retrain_' + name if name else '_retrain'}"
        else:
            self.training_path = self.create_training_path()

        self.writer = SummaryWriter(self.training_path)

        best_score = 0
        best_duration = 0
        scores = []
        durations = []

        n_to_30 = 0

        for i_episode in range(retrain_epochs if retrain_epochs else self.EPOCHS):
            episode_reward = 0
            state = self.env.reset()
            state = self._process_state(state)

            for t in count():
                action = self._select_action(state)
                observation, reward, done, info = self.env.step(action.item())
                observation = self._process_state(observation)
                reward = torch.tensor([reward])
                episode_reward += reward.item()
                next_state = None if done else observation
                self.memory.push(state, action, next_state, reward)

                state = next_state
                self._optimize_model()

                # Soft update of the target network's weights (θ′ ← τ θ + (1 −τ )θ′)
                if self.step_done % UPDATE_TARGETNET_RATE == 0:
                    target_state_dict = self.target_net.state_dict()
                    policy_state_dict = self.policy_net.state_dict()

                    for key in policy_state_dict:
                        target_state_dict[key] = self.TAU * policy_state_dict[key] + (1 - self.TAU) * target_state_dict[key]
                    self.target_net.load_state_dict(target_state_dict)

                if done:
                    train_score = self.env._game.score
                    scores.append(train_score)
                    durations.append(t + 1)
                    dic = {'memory_size': len(self.memory),
                           'train_score': train_score,
                           'train_reward': episode_reward,
                           'duration': t + 1,
                           }
                    self._write_stats(i_episode, **dic)

                    best_score = max(train_score, best_score)

                    if t + 1 > best_duration:
                        best_duration = t + 1
                        self.policy_net.log_weights(self.writer, i_episode)

                        if train_score > 2:
                            self._save_agent()

                    print(f"\r{i_episode + 1}/{self.EPOCHS}"
                          f"- D: {t + 1} (D* {best_duration}) "
                          f"\tS: {train_score} (S* {best_score}) "
                          f"\tEPS:{self.eps_threshold} , mean duration {np.mean(durations):.2f}",end='')

                    break
            # Stop when score is above 30
            if self.env._game.score >= 30:
                n_to_30 = i_episode
                break

        end_dic = self.__dict__
        end_dic['best_score'] = best_score
        end_dic['best_duration'] = best_duration
        end_dic['n_to_30'] = n_to_30

        self._log_agent(**end_dic)
        self._make_plot(scores, durations)
        self.writer.close()
        print(" ")
        return scores, durations, end_dic

    def retrain(self, training_path, name=None, load_network=False, load_memory=False, eps_start=0.01, epochs=1500):

        self.reset()
        if load_network:
            self.set_nets(training_path)
        if load_memory:
            self.set_memory(training_path)

        self.eps_threshold = eps_start

        return self.train(retrain_path=training_path, retrain_epochs=epochs, name=name)

    def test(self):
        with torch.no_grad():
            state = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            for t in count():
                action = self.policy_net(state).max(1)[1].view(1, 1)
                observation, reward, done, _ = self.env.step(action.item())
                state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
                if done:
                    next_state = None
                    return {'score': self.env._game.score, 'duration': t + 1}
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
                state = next_state

    def show_game(self, agent_vision=False, fps=60):
        if agent_vision:
            self.env.set_custom_render()


        sleep_time = 1 / fps

        total_reward = torch.tensor([0])
        state = self.env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        for t in count():
            action = self.policy_net(state).max(1)[1].view(1, 1)
            observation, reward, terminated, info = self.env.step(action.item())

            q_0 = self.policy_net(state)[0][0].item()
            q_sum = q_0 + self.policy_net(state)[0][1].item()
            stats = q_0 / q_sum  # dont need q_1/q_sum because it's 1 - q_0/q_sum

            self.env.render(stats=stats)
            time.sleep(sleep_time)

            reward = torch.tensor([reward])
            total_reward = total_reward + reward
            done = terminated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)

            # Move to the next state
            state = next_state

            if done:
                print(f"score: {self.env._game.score}, total reward: {total_reward.item()}.\nInfo= {info}")
                break
        time.sleep(1)

    def update_env(self, dic):
        self.env.update_params(dic)
