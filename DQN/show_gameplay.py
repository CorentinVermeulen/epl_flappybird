from flappy_bird_gym.envs.custom_env_simple import CustomEnvSimple as FlappyBirdEnv
from DQN_training import DQN, show_model_game

import torch


env = FlappyBirdEnv()
# Get number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state observations
state = env.reset()
n_observations = len(state)


file_dqn = "/Users/corentinvrmln/Desktop/memoire/flappybird/DQN/dqn_log/01_05_16_47/E585_S25.pt"

policy_net = DQN(n_observations, n_actions)
policy_net.load_state_dict(torch.load(file_dqn))
policy_net.eval()


show_model_game(env,policy_net, num_episodes=1, wait=False)