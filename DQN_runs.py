import pandas as pd
import time
import numpy as np
from DQN_agent_simple import DQNAgent_simple
from flappy_bird_gym.envs import CustomEnvSimple as FlappyBirdEnv

def log_df(df, name, scores, durations, end_dic, test_dic, t):
    df.loc[len(df)] = {'Name': name,
                       'n_to_30': end_dic['n_to_30'],
                       'mean_duration': np.mean(durations),
                       'max_score': max(scores),
                       'test_score': test_dic['score'],
                       'test_duration': test_dic['duration'],
                       'total_time': time.perf_counter() - t
                       }
df = pd.DataFrame( columns=['Name', 'n_to_30', 'mean_duration', 'max_score', 'test_score', 'test_duration', 'total_time'])



# most simple agent

env = FlappyBirdEnv()
env.obs_var = ['player_y', 'pipe_center_x', 'pipe_center_y', 'player_vel_y'] #['player_x', 'player_y', 'pipe_center_x', 'pipe_center_y', 'v_dist', 'h_dist', 'player_vel_y']
env.rewards = {"alive": 0.1, "pass_pipe": 1, "dead": -1, 'score': 0}
env.reset()
n_obs = env.observation_space.shape[0]
n_actions = env.action_space.n

hparams = {"EPOCHS": 4000,
           "BATCH_SIZE": 64,
           "EPS_DECAY": 16000,
           "layer_sizes": [n_obs*4, n_obs*8, n_obs*16, n_obs*32]}

root = "runs/SimpleAgent/"
dqnAgent = DQNAgent_simple(env, hparams, root_path=root)
scores, durations, end_dic = dqnAgent.train()
test_dic = dqnAgent.test()
















# ti = time.perf_counter()
# dqnAgent.reset()
# hparams = {"EPOCHS": 2000, "BATCH_SIZE": 256, "EPS_DECAY": 8000}
# dqnAgent.set_hyperparameters(hparams)
# scores, durations, end_dic = dqnAgent.train()
# test_dic = dqnAgent.test()
# log_df(df, str(hparams), scores, durations, end_dic, test_dic, ti)
# df.to_csv(f"{root}Batch_size_exploration.csv")

