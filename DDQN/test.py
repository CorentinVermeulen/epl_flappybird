import os
import sys
sys.path.append("./PyGame-Learning-Environment")
import pygame as pg
from ple import PLE
from ple.games.flappybird import FlappyBird
from ple import PLE
import agent
import torch
device = torch.device('cpu')
from flappy_bird_gym.envs.custom_env_simple import CustomPleEnv

# PLEnv
game = FlappyBird(width=256, height=256)
p = PLE(game, display_screen=False)
p.init()
p_actions = p.getActionSet()
p_action_dict = {0: p_actions[1], 1: p_actions[0]}
p_state = p.getGameState()
print(p_state)
len_state = len(p_state)
#____
p_state = p.getGameState()
state = torch.tensor(list(p_state.values()), dtype=torch.float32, device=device)
p_action = 0
p_reward = p.act(p_action_dict[p_action])
p_reward = torch.tensor([p_reward], device=device)
p_action = torch.tensor([p_action], device=device)


# gym Env
env = CustomPleEnv()
env.init()
env_actions = env.getActionSet()
env_action_dict = {0: env_actions[1], 1: env_actions[0]}
env_state = env.getGameState()
print(env_state)
len_state = len(env_state)
#____
env_state = env.getGameState()
state = torch.tensor(list(env_state.values()), dtype=torch.float32, device=device)
env_action = 0
env_reward = env.act(env_action_dict[env_action])
env_reward = torch.tensor([env_reward], device=device)
env_action = torch.tensor([env_action], device=device)
