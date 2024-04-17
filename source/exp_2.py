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
               "UPDATE_TARGETNET_RATE": 1,
               "BATCH_SIZE": 256,
               "LR": 1e-4,
               }

## ENVIRONMENT CONTEXT
game_context = {'PLAYER_FLAP_ACC': -5, 'PLAYER_ACC_Y': 1, 'pipes_are_random': False}

## LEARNING PARAMETERS
root = '../../experiments/C_or_tau/'

iters = 5
set = ([50, 0.99], [100, 1], )
n = len(set)

print(f"Python script root: {os.getcwd()}")
print(f"Starting {n*iters} experiments at {root}")
print("Device cuda? ", torch.cuda.is_available())

for update_rate, tau in set:
        current_hp = baseline_HP.copy()
        current_hp.update({"TAU": tau,
                            "UPDATE_TARGETNET_RATE": update_rate}
                          )
        for rep in range(iters):
            t = time.perf_counter()
            env = FlappyBirdEnv()
            agent = AgentSimple(FlappyBirdEnv(), HParams(current_hp), root_path=root)
            agent.update_env(game_context)
            scores, durations = agent.train(show_progress=False, name=f'T{tau}_U{update_rate}_R{rep}')
            HD = np.max(durations)
            MD = np.mean(durations)
            MD_last = np.mean(durations[-250:])
            te = time.perf_counter() - t
            print(
                f"T{tau}_U{update_rate}_R{rep}\n"
                f"\tD* {HD:<4.0f} - E[D] {MD:<5.0f} - E[D]_250 {MD_last:<5.0f} "
                f"- Time {int(te // 60):02}:{int(te % 60):02}"
            )


#make_experiment_plot(root)
