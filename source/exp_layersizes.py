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
               "EPS_DECAY": 2000,
               "TAU": 0.01,
               "LAYER_SIZES": [1024, 1024],
               "UPDATE_TARGETNET_RATE": 3}


layers = [[2048],
          [512, 1024],
          [64, 1024, 64],
          [64, 128, 256, 512],
          [128, 256, 512, 128],
          [256, 256, 256, 256],
          [512, 512, 512, 512],
          [64, 128, 256, 512, 256, 128]
          ]
root = '../../experiments/layer_size/'

print(f"Python script root: {os.getcwd()}")
print(f"Starting {len(layers)} experiments at {root}")
print("Device cuda? ", torch.cuda.is_available())

for i in range(len(layers)):
    t = time.perf_counter()
    env = FlappyBirdEnv()
    current_hp = baseline_HP.copy()
    current_hp.update({"LAYER_SIZES": layers[i]})
    agent = AgentSimple(FlappyBirdEnv(), HParams(current_hp) , root_path=root)
    scores, durations = agent.train(show_progress=False, name=f'layer_size_({str(layers[i])})')
    HS = np.max(scores)
    MD = np.mean(durations)
    MD_last = np.mean(durations[-250:])
    te = time.perf_counter() - t
    print(f"{i+1} - Layer size {layers[i]}\n\tS* {HS:<4.0f} - E[D] {MD:<5.0f} - E[D]_250 {MD_last:<5.0f} - Time {int(te//60):02}:{int(te%60):02}")

make_experiment_plot(root)
