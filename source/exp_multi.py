import time
import pandas as pd
import numpy as np
import torch.cuda
import os
from utils import HParams, make_experiment_plot
from agent_multi import AgentSimple
from flappy_bird_gym.envs import FlappyBirdEnvSimpleMulti as FlappyBirdEnv
import random
baseline_HP = {"EPOCHS": 2500,
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
game_context = {'PLAYER_ACC_Y': 1,
                'PLAYER_FLAP_ACC': -9,
                'pipes_are_random': True, }

## LEARNING PARAMETERS
root = '../../exps/multi_316/'
iters = 10
param = [16,3]

print(f"Python script root: {os.getcwd()}")
print(f"Starting {len(param)*iters} experiments at {root}")
print("Device cuda? ", torch.cuda.is_available())

for i in range(len(param)):
    current_hp = baseline_HP.copy()
    for rep in range(iters):
        t = time.perf_counter()
        env = FlappyBirdEnv(n_actions=param[i])

        agent = AgentSimple(env, HParams(current_hp), root_path=root)
        agent.update_env(game_context)

        sed = random.randrange(0,1000)
        scores, durations = agent.train(show_progress=False, name=f'jf{param[i]}_{i+1}_{rep+1}_{sed}')
        HD = np.max(durations)
        MD = np.mean(durations)
        MD_last = np.mean(durations[-250:])
        te = time.perf_counter() - t
        print(
            f"Param_{param[i]}_{rep+1}_{sed}\n"
            f"\tD* {HD:<4.0f} - E[D] {MD:<5.0f} - E[D]_250 {MD_last:<5.0f} "
            f"- Time {int(te // 60):02}:{int(te % 60):02}"
        )
