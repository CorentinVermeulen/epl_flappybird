import time
import numpy as np
import pandas as pd
from DQN_agent_simple import DQNAgent_simple
from flappy_bird_gym.envs import CustomEnvSimple as FlappyBirdEnv

"""
PLAN:

- Learn with default parameters
    - Train until score >= 30
        - Save nbr of epochs to reach >= 30
    - Save model
    - Test model
- Change parameter
    - Retrain
        - Save nbr of epochs to reach >= 30
    - Test model
"""

def log_df(df, name, scores, durations, end_dic, test_dic, t):
    df.loc[len(df)] = {'Name': name,
                       'n_to_30': end_dic['n_to_30'],
                       'mean_duration': np.mean(durations),
                       'max_score': max(scores),
                       'test_score': test_dic['score'],
                       'test_duration': test_dic['duration'],
                       'total_time': time.perf_counter() - t
                       }


env = FlappyBirdEnv()
env.obs_var = ['player_x', 'player_y', 'pipe_center_x', 'pipe_center_y', 'v_dist', 'h_dist', 'player_vel_y']
env.rewards = {"alive": 0.1, "pass_pipe": 1, "dead": -1, 'score': 0}

hparams = {"EPOCHS": 1500,
           "BATCH_SIZE": 64,
           "EPS_DECAY": 4000,
           "layer_sizes": [64, 128, 256, 256]}

root = "runs/Change_parms/"
dqnAgent = DQNAgent_simple(env, hparams, root_path=root)

params = [{"PLAYER_FLAP_ACC": -6},
          {"PLAYER_FLAP_ACC": -6},
          {"PLAYER_ACC_Y ": 0.3},
          {"PLAYER_ACC_Y ": 1.7},
          {"pipes_are_random": True}]
df = pd.DataFrame(columns=['Name', 'n_to_30', 'mean_duration', 'max_score', 'test_score', 'test_duration', 'total_time'])


# Learn with default parameters
print("\n\tTRAINING ...")
t = time.perf_counter()
dqnAgent.reset()
scores, durations, end_dic = dqnAgent.train()
training_path = dqnAgent.training_path
print(f"\n\tTESTING ...")
test_dic = dqnAgent.test()

log_df(df, "Initial training", scores, durations, end_dic, test_dic, t)

# Test model
print(f"\n\tTESTING ... {dqnAgent.test()}")

# Change parameter
for param in params:
    print(f"\n\tCHANGING PARAMETERS... {param}")

    env.update_params(param)
    name = str(param) + 'NF_MF'
    t = time.perf_counter()
    dqnAgent.reset()
    scores, durations, end_dic = dqnAgent.retrain(training_path, name=name + 'NF_MF', load_network=False,
                                                  load_memory=False, eps_start=0.99, epochs=1500)
    test_dic = dqnAgent.test()
    log_df(df, name, scores, durations, end_dic, test_dic, t)


    name = str(param) + 'NT_MT'
    t = time.perf_counter()
    dqnAgent.reset()
    scores, durations, end_dic = dqnAgent.retrain(training_path, name=name + 'NT_MT', load_network=True,
                                                  load_memory=True, eps_start=0.5, epochs=1500)

    test_dic = dqnAgent.test()
    log_df(df, name, scores, durations, end_dic, test_dic, t)

print(df)
df.to_csv(f"{root}df.csv")
