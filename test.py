from repo.agents.DQN_agent_simple import DQNAgent_simple
from flappy_bird_gym.envs import CustomEnvSimple as FlappyBirdEnv

env = FlappyBirdEnv()
env.obs_var = ['player_y', 'pipe_center_x', 'pipe_center_y', 'player_vel_y'] #['player_x', 'player_y', 'pipe_center_x', 'pipe_center_y', 'v_dist', 'h_dist', 'player_vel_y']
env.reset()
n_obs = env.observation_space.shape[0]
n_actions = env.action_space.n

hparams = {"layer_sizes": [256, 512, 512, 512],
           "EPOCHS": 1500,
           "BATCH_SIZE": 512,
           "EPS_DECAY": 2000}
# Normal game
# agent = DQNAgent_simple(env, hparams, root_path="runs/JumpForce/")
# agent.update_env({"PLAYER_FLAP_ACC": -9, "PLAYER_ACC_Y": 1})
# agent.train()

# Easier game
agent = DQNAgent_simple(env, hparams, root_path="runs/Iterative/")
agent.update_env({"PLAYER_FLAP_ACC": -6, "PLAYER_ACC_Y": 1, "pipes_are_random":True})
agent.train()


