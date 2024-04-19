import time
import pandas as pd
import numpy as np
import torch.cuda
import os
from utils import HParams, make_experiment_plot
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
EXPERIENCE 4: Jump Force + Gravity impact
"""

## ENVIRONMENT CONTEXT
game_context = {'PLAYER_FLAP_ACC': -5, 'PLAYER_ACC_Y': 1, 'pipes_are_random': True}

## LEARNING PARAMETERS
root = '../../exps/exp_4/'

iters = 5
params = [0, 0.5, 1.0, 1.5, 1.75, 2, 2.25, 3]  # Jump Force
p_name = 'PLAYER_FLAP_ACC_VARIANCE'
p_short = 'JF'
params2 = [0, 0.25, 0.5, 0.75, 1.0, 1.5]  # Gravity
p_name2 = 'GRAVITY_VARIANCE'
p_short2 = 'GR'
lrs = [1e-5] # [1e-4, 1e-5]
obss = [True, False]
n = iters * len(params) * len(lrs) * len(obss) * len(params2)
print(f"Python script root: {os.getcwd()}")
print(f"Starting {n} experiments at {root}")
print("Device cuda? ", torch.cuda.is_available())

for obs in obss:
    for lr in lrs:
        for param in params:
            for param2 in params2:
                current_hp = baseline_HP.copy()
                current_hp.update({"LR": lr})
                game_context.update({p_name: param, p_name2: param2})
                for rep in range(iters):
                    t = time.perf_counter()

                    env = FlappyBirdEnv()
                    env.obs_jumpforce = obs
                    env.obs_gravity = obs

                    agent = AgentSimple(env, HParams(current_hp), root_path=root)
                    agent.update_env(game_context)

                    name = f'{p_short}{param}_{p_short2}{param2}_LR{lr}_Obs{obs * 1}_R{rep}'

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
print("end_exp_4.py")