import time
import pandas as pd
import numpy as np
import torch.cuda
import os
from utils import HParams, make_experiment_plot
from agent_simple import AgentSimple
from flappy_bird_gym.envs import FlappyBirdEnvSimpleFast as FlappyBirdEnv
import random
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

""" Now we have best hyperparameters for the model,
We can start learning in random environment

First: Random pipes and we will see the effect of having random pipes or fixed pipes

Second: Random jumpforce, this can be more realistic since a player can jump with different forces or can be affected by wind

Third: Random gravity, this can be seen as vertical wind effect on the player

Fourth: 

"""

## ENVIRONMENT CONTEXT
game_context = {'PLAYER_FLAP_ACC': -5, 'PLAYER_ACC_Y': 1, 'pipes_are_random': False}

## LEARNING PARAMETERS
root = '../../experiments/rd_pipes/'
iters = 10
param = [True, False]

print(f"Python script root: {os.getcwd()}")
print(f"Starting {len(param)*iters} experiments at {root}")
print("Device cuda? ", torch.cuda.is_available())

for j in range(len(param)):
    current_hp = baseline_HP.copy()
    game_context.update({'pipes_are_random': param[j]})
    for rep in range(iters):
        t = time.perf_counter()
        env = FlappyBirdEnv()
        agent = AgentSimple(FlappyBirdEnv(), HParams(current_hp), root_path=root)
        agent.update_env(game_context)
        sed = random.randrange(0,1000)
        scores, durations = agent.train(show_progress=False, name=f'{j+1}_{rep+1}_{sed}')
        HS = np.max(scores)
        MD = np.mean(durations)
        MD_last = np.mean(durations[-250:])
        te = time.perf_counter() - t
        print(
            f"{j+1} {rep+1}\n"
            f"\tS* {HS:<4.0f} - E[D] {MD:<5.0f} - E[D]_250 {MD_last:<5.0f} "
            f"- Time {int(te // 60):02}:{int(te % 60):02}"
        )
