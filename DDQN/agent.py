'''
agent.py

the meat of the reinforcement learning program

create the 'agent' to interact with the environment

takes actions, and trains the model

'''

import torch
import model
import memory
import numpy as np
import random
from itertools import count
import pygame as pg
import matplotlib.pyplot as plt
import torch.optim as optim
import math
from torch.utils.tensorboard import SummaryWriter

class Agent():
    def __init__(self, BATCH_SIZE,
                 MEMORY_SIZE,
                 GAMMA,
                 input_dim, output_dim, action_dim, action_dict,
                 EPS_START, EPS_END, EPS_DECAY_VALUE,
                 lr, TAU,
                 network_type='DQN') -> None:
        # Set all the values up
        self.BATCH_SIZE = BATCH_SIZE
        self.GAMMA = GAMMA
        self.MEMORY_SIZE = MEMORY_SIZE
        self.action_dim = action_dim
        self.action_dict = action_dict
        self.EPS_START = EPS_START
        self.EPS_END = EPS_END
        self.EPS_DECAY_VALUE = EPS_DECAY_VALUE
        self.eps = EPS_START
        self.TAU = TAU

        self.UPDATE_TARGET_FREQUENCY = 100

        # MPS Apple M2 GPU
        self.device = "cpu" #mps
        torch.set_default_device(self.device)
        self.episode_durations = []
        # Create the cache recall memory
        self.cache_recall = memory.MemoryRecall(memory_size=MEMORY_SIZE)
        self.network_type = network_type
        # Create the dual Q networks - target and policy nets
        self.policy_net = model.DQN(input_dim=input_dim, output_dim=output_dim, network_type=network_type).to(
            self.device)
        self.target_net = model.DQN(input_dim=input_dim, output_dim=output_dim, network_type=network_type).to(
            self.device)
        # No need to calculate gradients for target net parameters as these are periodically copied from the policy net
        for param in self.target_net.parameters():
            param.requires_grad = False
        # Copy the initial parameters from the policy net to the target net to align them at the start
        # Diff
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr, amsgrad=True)
        self.steps_done = 0
        self.best_duration = 0

        self.writer = SummaryWriter()
        self.policy_net.tensorboard_model(self.writer)
        self.target_net.tensorboard_model(self.writer)


    # We want to use no gradient computation, as we do not update the networks paramaters
    @torch.no_grad()
    def take_action(self, state):
        # Decay and cap the epsilon value
        self.eps = self.eps * self.EPS_DECAY_VALUE
        self.eps = max(self.eps, self.EPS_END)
        # self.eps = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * self.steps_done / self.EPS_DECAY_VALUE)
        # Take a random action
        if self.eps < np.random.rand():
            state = state[None, :]
            action_idx = torch.argmax(self.policy_net(state), dim=1).item()
        # Else take a greedy action
        else:
            action_idx = np.random.randint(0, self.action_dim - 1)
        self.steps_done += 1
        return action_idx

    def plot_durations(self):
        plt.figure(1)
        plt.clf()
        durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
        plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        # Plot the durations
        plt.plot(durations_t.cpu().numpy())
        # Take 100 episode averages of the durations and plot them too, to show a running average on the graph
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.cpu().numpy())
        plt.pause(0.001)  # pause a bit so that plots are updated
        plt.savefig(self.network_type + '_training.png')

    # Function to copy the policy net parameters to the target net
    # def update_target_network(self):
    #     self.target_net.load_state_dict(self.policy_net.state_dict())

    def update_target_network(self):
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        # Update the parameters in the target network
        for key in policy_net_state_dict:
            # target_net_state_dict[key] = policy_net_state_dict[key] * self.TAU + target_net_state_dict[key] * (
            #             1 - self.TAU)
            target_net_state_dict[key].mul_(self.TAU).add_(policy_net_state_dict[key], alpha=1 - self.TAU)
        self.target_net.load_state_dict(target_net_state_dict)

    def optimize_model(self):
        # Only pop the data if we have enough for the network
        if len(self.cache_recall) < self.BATCH_SIZE:
            return
        # Grab the batch
        batch = self.cache_recall.recall(self.BATCH_SIZE)
        # reform the batch so we can grab the states, actions etc easily
        batch = [*zip(*batch)]
        state = torch.stack(batch[0])
        # batch[1] gives us the next_state after the action, we want to create a mask to filter out the states where we end the run (i.e. the flappy bird dies). the end of the run will give a state of None which cannot be inputted into the network
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch[1])), device=self.device,
                                      dtype=torch.bool)
        # Grab the next states that are not final states
        non_final_next_states = torch.stack([s for s in batch[1] if s is not None])
        # Grab the action and the reward
        action = torch.stack(batch[2])
        reward = torch.cat(batch[3])
        next_state_action_values = torch.zeros(self.BATCH_SIZE, dtype=torch.float32, device=self.device)
        # Get the Q values from the policy network and then get the Q value for the given action taken
        # [32,1]
        state_action_values = self.policy_net(state).gather(1, action)
        # print(self.target_net(non_final_next_states).max(1, keepdim=True)[0].size())
        # Use the target network to get the maximum Q for the next state across all the actions
        with torch.no_grad():
            next_state_action_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
        # Calcuate the expected state action values as the per equation we use
        expected_state_action_values = (next_state_action_values * self.GAMMA) + reward
        # Using the L1 Loss, calculate the difference between the expected and predicted values and optimize the policy network only
        loss_fn = torch.nn.SmoothL1Loss()
        # print(state_action_values.size())
        # print(expected_state_action_values.size())
        loss = loss_fn(state_action_values, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self, episodes, env):
        self.steps_done = 0
        for episode in range(episodes):
            torch.mps.empty_cache()
            env.reset_game()
            state = env.getGameState()
            state = torch.tensor(list(state.values()), dtype=torch.float32, device=self.device)
            # Inf count functiom
            for c in count():
                # Choose an action, get back the reward and the next state as a result of taking the action
                action = self.take_action(state)
                reward = env.act(self.action_dict[action])
                reward = torch.tensor([reward], device=self.device)
                action = torch.tensor([action], device=self.device)
                next_state = env.getGameState()
                next_state = torch.tensor(list(next_state.values()), dtype=torch.float32, device=self.device)
                done = env.game_over()
                # if game is over, next state is None
                if done:
                    next_state = None
                # Cache a tuple of the date
                self.cache_recall.cache((state, next_state, action, reward, done))
                # Set the state to the next state
                state = next_state
                # Optimize the model and update the target network
                self.optimize_model()

                if c % self.UPDATE_TARGET_FREQUENCY == 0:
                    self.update_target_network()
                #pg.display.update()
                if episode % 10 == 0:
                    pass
                    #env.render()

                if done:
                    # Track duratiop and plot
                    # self.episode_durations.append(c + 1)
                    # self.plot_durations()

                    print(f"\r i{episode:04d} D:{c+1}", end="")
                    if episode % 10 == 0:
                        print(f"\ri{episode:04d}: score: {env.score()}\t Durations: {c + 1}.\t Eps: {self.eps}")

                    self.writer.add_scalar("score", env.score(), episode)
                    self.writer.add_scalar("duration", c + 1, episode)

                    if c+1 > self.best_duration:
                        self.best_duration = c+1
                        print(f"\r\tNEW BP: i{episode:04d}: score: {env.score()}\t Durations: {c + 1}.\t Eps: {self.eps}")
                        torch.save(self.target_net.state_dict(), self.network_type + '_target_net_best.pt')
                        torch.save(self.policy_net.state_dict(), self.network_type + '_policy_net_best.pt')

                    # Start a new episode
                    break