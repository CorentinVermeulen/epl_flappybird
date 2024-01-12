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

# from flappy_bird_gym.envs.flappy_bird_env_simple import FlappyBirdEnvSimple as FlappyBirdEnv


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
        self.capacity = capacity

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    def isfull(self):
        return len(self.memory) == self.capacity

    def write_to_csv(self, filename):
        with open(filename, 'w') as fichier_csv:
            # Créer un objet writer
            writer = csv.writer(fichier_csv)

            # Écrire les données du deque dans le fichier CSV
            for element in self.memory:
                writer.writerow([element])


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


def test_model_game(policy_net):
    with torch.no_grad():
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        for t in count():
            action = policy_net(state).max(1)[1].view(1, 1)
            observation, reward, terminated, _ = env.step(action.item())
            done = terminated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)

            # Move to the next state
            state = next_state

            if done:
                return env._game.score



if __name__ == "__main__":

    def select_action(state):
        global steps_done
        global eps_threshold  # For plotting
        sample = random.random()
        if eps_threshold > EPS_END:
            eps_threshold = round(EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY),3)

        steps_done += 1

        # Best action
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return policy_net(state).max(1)[1].view(1, 1)
        # Random action
        else:
            return torch.tensor([[env.action_space.sample()]], dtype=torch.long)


    def optimize_model():
        if len(memory) < BATCH_SIZE:
            return
        transitions = memory.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE)
        with torch.no_grad():
            next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
        optimizer.step()



    ###########################
    ''' SOURCE:
    Environment https://github.com/Talendar/flappy-bird-gym/tree/main 
    Reinforcement Learning: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html '''
    ###########################

    torch.device("mps")  # Multi-Processing-Sync

    MEMORY_SIZE = 100000
    EPOCHS = 2000

    # Training parameters
    BATCH_SIZE = 256  # the number of transitions sampled from the replay buffer
    GAMMA = 0.99  # the discount factor as mentioned in the previous section
    EPS_START = 0.9  # the starting value of epsilon
    EPS_END = 0.01  # the final value of epsilon
    EPS_DECAY = 2000  # controls the rate of exponential decay of epsilon, higher means a slower decay
    # EPS_DECAY = EPOCHS / 0.2  # controls the rate of exponential decay of epsilon, higher means a slower decay
    TAU = 0.005  # the update rate of the target network
    LR = 1e-4  # the learning rate of the ``AdamW`` optimizer
    num_episodes = EPOCHS  # the number of episodes for training

    OBS_VAR = ['player_x', 'player_y', 'pipe_center_x', 'pipe_center_y', 'v_dist', 'h_dist']
    env = FlappyBirdEnv(obs_var=OBS_VAR)

    n_actions = env.action_space.n  # Get number of actions from gym action space
    n_observations = len(env.reset())  # Get the number of state observations
    run_name = time.strftime("%m_%d_%H_%M", time.localtime())
    filename = "./dqn_log/" + run_name
    os.makedirs(filename)

    print(f"\n===================\nEnvironment({env})\n==================="
          f"Action space: {env.action_space}\n"
          f"Observation space: {env.observation_space}\n"
          f"Reward range: {env.reward_range}\n"
          f"Env obs variables: {env.obs_var}\n"
          f"Env obs values: {env.reset()}\n"
          f"=========================================================\n")

    policy_net = DQN(n_observations, n_actions)
    target_net = DQN(n_observations, n_actions)
    #policy_net.load_state_dict(torch.load("/Users/corentinvrmln/Desktop/memoire/flappybird/repo/DQN/dqn_log/01_10_16_32/E1444_S50.pt"))
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(MEMORY_SIZE)
    memory_full = None

    # Log hyper parameters
    with open(f"{filename}/parameters.txt", "w") as f:
        f.write(f"MEMORY_SIZE = {MEMORY_SIZE}\n")
        f.write(f"EPOCHS = {EPOCHS}\n")
        f.write(f"BATCH_SIZE = {BATCH_SIZE}\n")
        f.write(f"GAMMA = {GAMMA}\n")
        f.write(f"EPS_START = {EPS_START}\n")
        f.write(f"EPS_END = {EPS_END}\n")
        f.write(f"EPS_DECAY = {EPS_DECAY}\n")
        f.write(f"TAU = {TAU}\n")
        f.write(f"LR = {LR}\n")
        f.write(f"num_episodes = {num_episodes}\n")
        f.write(f"n_actions = {n_actions}\n")
        f.write(f"n_observations = {n_observations}\n")
        f.write(f"obs_var = {env.obs_var}\n")
        f.write(f"\nModel structure:\n")
        for name, layer in policy_net.named_children():
                f.write(f"{name}: {str(layer)}\n")

    eps_threshold = EPS_START

    steps_done = 0
    best_reward = 0
    best_score = 0
    best_duration = 0
    n_train_50 = 0
    n_test_50 = 0

    writer = SummaryWriter(f'runs/DQN/{run_name}')
    def write_stats(writer, epoch, total_reward, train_score, duration, eps_threshold, test_score=None):
        d = {'train': train_score, 'reward': total_reward, 'eps': eps_threshold, 'duration': duration}
        if test_score:
            d['test'] = test_score

        writer.add_scalars('score', d, epoch)

    # Log the model
    obs = {cle: float(valeur) for cle, valeur in env.dict_obs.items() if cle in env.obs_var}
    writer.add_graph(policy_net, torch.tensor(list(obs.values())))

    print("---- START TRAINING ---- ")
    s = time.perf_counter()
    for i_episode in range(num_episodes):
        total_reward = torch.tensor([0])
        # Initialize the environment and get it's state
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        for t in count():
            action = select_action(state)  # Random or best action
            observation, reward, terminated, _ = env.step(action.item())
            reward = torch.tensor([reward])
            total_reward = total_reward + reward
            done = terminated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)

            # Store the transition in memory
            memory.push(state, action, next_state, reward)
            # Log when memory is full
            if not memory_full and memory.isfull():
                memory_full = i_episode

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            optimize_model()

            # Soft update of the target network's weights (θ′ ← τ θ + (1 −τ )θ′)
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
            target_net.load_state_dict(target_net_state_dict)

            if done:
                print(f"\r{i_episode}/{num_episodes}: dur {t+1}\t({round(time.perf_counter() - s, 2)} sec.) eps: {round(eps_threshold, 3)}",end="")
                """ Variables of performance:
                Duration
                Reward
                Score
                """
                # Log the performance
                game_score = env._game.score
                if game_score ==50:
                    n_train_50 += 1
                test_score = test_model_game(policy_net)  # 1 if deterministic
                if test_score == 50:
                    n_test_50 += 1
                duration = t + 1
                write_stats(writer, i_episode, total_reward, game_score, duration, eps_threshold, test_score)
                policy_net.log_weights(writer, i_episode)

                # If best duration, save the model
                if t+1 > best_duration:
                    best_duration = t+1
                    print(f"\r\tNEW BP: i{i_episode}: durations: {t + 1}. \t score: {game_score}\t Eps: {eps_threshold}")

                    if best_score >= 2:
                        torch.save(policy_net.state_dict(), f"{filename}/E{i_episode}_S{game_score}.pt")
                        memory.write_to_csv(f"{filename}/E{i_episode}_S{game_score}_memory.csv")
                    policy_net.log_weights(writer, i_episode)


                best_reward = max(total_reward, best_score)
                best_score = max(game_score, best_score)

                break

    torch.save(policy_net.state_dict(), f"{filename}/E{num_episodes}.pt")
    memory.write_to_csv(f"{filename}/E{num_episodes}__memory.csv")
    policy_net.log_weights(writer, num_episodes)
    target_net.log_weights(writer, num_episodes)
    writer.close()
    with open(f"{filename}/parameters.txt", "a") as f:
        f.write("\nEnd training Results:\n")
        f.write(f"Best duration: {best_duration}\n")
        f.write(f"Best reward: {best_reward}\n")
        f.write(f"Best score: {best_score}\n")
        f.write(f"n_train_50: {n_train_50}\n")
        f.write(f"n_test_50: {n_train_50}\n")
        f.write(f"Memory full at: {memory_full}\n")



    print("\n---- END TRAINING ----\n ")

    ### Save the model
    '''
    Finally, we can save our model:
    '''

    # file_dqn = "../flappy-bird-gym/fb_dqn_at_9000.pt"
    # torch.save(policy_net.state_dict(), file_dqn)

    ### Load the model
    '''
    To load the model, we can instantiate a new object of the same class, and load the parameters using load_state_dict().
    '''
    # policy_net = DQN(n_observations, n_actions)
    # policy_net.load_state_dict(torch.load(file_dqn))
    # policy_net.eval()

    ### Show the model
    # print("---- SHOWING RESULTS ---- ")
    # show_model_game(policy_net)
