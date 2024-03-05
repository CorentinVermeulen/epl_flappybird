from repo.agents.DQN_agent_simple import DQNAgent_simple
from flappy_bird_gym.envs import CustomEnvSimple as FlappyBirdEnv

# most simple agent

env = FlappyBirdEnv()
env.obs_var = ['player_y', 'pipe_center_x', 'pipe_center_y', 'h_dist', 'player_vel_y'] #['player_x', 'player_y', 'pipe_center_x', 'pipe_center_y', 'v_dist', 'h_dist', 'player_vel_y']
env.reset()
n_obs = env.observation_space.shape[0]
n_actions = env.action_space.n

hparams = {"EPOCHS": 2000,
           "BATCH_SIZE": 128,
           "EPS_DECAY": 2000,
           "layer_sizes": [128,128, 128, 128]}

root = "runs/SimpleAgent/"
dqnAgent = DQNAgent_simple(env, hparams, root_path=root)
dqnAgent.update_env({"PLAYER_FLAP_ACC": -6, "PLAYER_ACC_Y": 1}) # Make game easier
scores, durations, end_dic = dqnAgent.train()
test_dic = dqnAgent.test()





