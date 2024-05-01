import time
import pandas as pd
import numpy as np
import torch.cuda
import os
from utils import HParams
from agent_simple import AgentSimple
from flappy_bird_gym.envs import FlappyBirdEnvSimpleFast as FlappyBirdEnv

baseline_HP = {"EPOCHS": 1000,
               "MEMORY_SIZE": 100000,
               "EPS_START": 0.9,
               "EPS_END": 0.001,
               "EPS_DECAY": 2000,
               "TAU": 0.01,
               "LAYER_SIZES": [256, 256, 256, 256],
               "GAMMA": 0.99,
               "UPDATE_TARGETNET_RATE": 1,
               "BATCH_SIZE": 256,
               "LR": 1e-4,
               }

"""
EXPERIENCE 0: Baseline
"""

## ENVIRONMENT CONTEXT
game_context = {'PLAYER_FLAP_ACC': -5,
                'PLAYER_ACC_Y': 1,
                'pipes_are_random': False}

## LEARNING PARAMETERS
root = '../../exps/exp_0/'

iters = 5

lrs = [1e-05]
n = iters * len(lrs)
print(f"Python script root: {os.getcwd()}")
print(f"Starting {n} experiments at {root}")
print("Device cuda? ", torch.cuda.is_available())

for lr in lrs:
    current_hp = baseline_HP.copy()
    current_hp.update({"LR": lr})
    for rep in range(iters):
        t = time.perf_counter()

        env = FlappyBirdEnv()
        env.obs_jumpforce = False
        env.obs_gravity = False

        agent = AgentSimple(env, HParams(current_hp), root_path=root)
        agent.update_env(game_context)

        name = f'Baseline_LR{lr}_R{rep}'

        scores, durations = agent.train(show_progress=False, name=name)
        HD = np.max(durations)
        MD = np.mean(durations)
        MD_last = np.mean(durations[-250:])
        te = time.perf_counter() - t
        print(
            f"{name}\n"
            f"\tD* {HD:<4.0f} - E[D] {MD:<5.0f} - E[D]_250 {MD_last:<5.0f} "
            f"- Time {int(te // 60):02}:{int(te % 60):02}"
        )
print("end_exp_0.py")
