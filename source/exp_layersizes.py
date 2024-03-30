import time
import pandas as pd
import numpy as np
import torch.cuda
import os
from utils import HParams, make_experiment_plot
from agent_simple import AgentSimple
from flappy_bird_gym.envs import FlappyBirdEnvSimpleFast as FlappyBirdEnv

baseline_HP = {"EPOCHS": 750,
               "BATCH_SIZE": 256,
               "LR": 1e-4,
               "MEMORY_SIZE": 100000,
               "GAMMA": 0.95,
               "EPS_START": 0.9,
               "EPS_END": 0.001,
               "EPS_DECAY": 2500,
               "TAU": 0.01,
               "LAYER_SIZES": [1024, 1024],
               "UPDATE_TARGETNET_RATE": 3}

game_context = {'PLAYER_FLAP_ACC': -5, 'PLAYER_ACC_Y': 1, 'pipes_are_random': False}

layers = [[256, 256, 256, 256],
          [64, 128, 256, 512, 256, 128],
          ]

gammas = [0.999]
root = '../../experiments/layer_size_duel/'

print(f"Python script root: {os.getcwd()}")
print(f"Starting {len(layers)*len(gammas)*5} experiments at {root}")
print("Device cuda? ", torch.cuda.is_available())

for i in range(len(layers)):
    for j in range(len(gammas)):
        current_hp = baseline_HP.copy()
        current_hp.update({"LAYER_SIZES": layers[i], "GAMMA": gammas[j]})
        for rep in range(1,6):
            t = time.perf_counter()
            env = FlappyBirdEnv()
            agent = AgentSimple(FlappyBirdEnv(), HParams(current_hp), root_path=root)
            agent.update_env(game_context)
            scores, durations = agent.train(show_progress=False, name=f'layer_size_({str(layers[i])}-{rep})')
            HS = np.max(scores)
            MD = np.mean(durations)
            MD_last = np.mean(durations[-250:])
            te = time.perf_counter() - t
            print(
                f"{i + 1} {j+1} {rep} - G:{gammas[j]} LS:{layers[i]}\n"
                f"\tS* {HS:<4.0f} - E[D] {MD:<5.0f} - E[D]_250 {MD_last:<5.0f} "
                f"- Time {int(te // 60):02}:{int(te % 60):02}"
            )

make_experiment_plot(root)
