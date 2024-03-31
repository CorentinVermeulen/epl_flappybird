import time
import pandas as pd
import numpy as np
import torch.cuda
import os
from utils import HParams, make_experiment_plot
from agent_simple import AgentSimple
from flappy_bird_gym.envs import FlappyBirdEnvSimpleFast as FlappyBirdEnv

baseline_HP = {"EPOCHS": 750,
               "MEMORY_SIZE": 100000,
               "EPS_START": 0.9,
               "EPS_END": 0.001,
               "EPS_DECAY": 2500,
               "TAU": 0.01,
               "LAYER_SIZES": [256, 256, 256, 256],
               "GAMMA": 0.99,
               "UPDATE_TARGETNET_RATE": 3,
               "BATCH_SIZE": 256,
               "LR": 1e-4,
               }
""" EXP 1 : GAMMA ? 
LAYER SIZE: [256, 256, 256, 256] or [64, 128, 256, 512, 256, 128] ? Equivalent but less params in the first one
GAMMA: 0.99 or 0.999 ? 

"""

game_context = {'PLAYER_FLAP_ACC': -5, 'PLAYER_ACC_Y': 1, 'pipes_are_random': False}

## LEARNING PARAMETERS
root = '../../experiments/gamma_99_999/'
gamma = [0.99, 0.999]
iters = 5

print(f"Python script root: {os.getcwd()}")
print(f"Starting {len(gamma)*iters} experiments at {root}")
print("Device cuda? ", torch.cuda.is_available())

for j in range(len(gamma)):
    current_hp = baseline_HP.copy()
    current_hp.update({"GAMMA": gamma[j]})
    for rep in range(iters):
        t = time.perf_counter()
        env = FlappyBirdEnv()
        agent = AgentSimple(FlappyBirdEnv(), HParams(current_hp), root_path=root)
        agent.update_env(game_context)
        scores, durations = agent.train(show_progress=False, name=f'G_{str(gamma[j])}_{rep}')
        HS = np.max(scores)
        MD = np.mean(durations)
        MD_last = np.mean(durations[-250:])
        te = time.perf_counter() - t
        print(
            f"{j+1} {rep+1} - G:{gamma[j]}\n"
            f"\tS* {HS:<4.0f} - E[D] {MD:<5.0f} - E[D]_250 {MD_last:<5.0f} "
            f"- Time {int(te // 60):02}:{int(te % 60):02}"
        )

#make_experiment_plot(root)
