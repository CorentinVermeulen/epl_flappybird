import time
from agents.DQN_agent_ceci import DQNAgent_simple_cuda as DQNAgent_simple
from flappy_bird_gym.envs import FlappyBirdEnvSimpleFast as FlappyBirdEnv
import pandas as pd
import numpy as np

root = 'TEST_lowdecay_local/'

hp = {"layer_sizes": [256, 256, 256, 256],
      "EPOCHS": 750,
      "BATCH_SIZE": 256,
      "EPS_DECAY": 1000}

env = FlappyBirdEnv()
env.reset()

n_obs = env.observation_space.shape[0]
n_actions = env.action_space.n

agent = DQNAgent_simple(FlappyBirdEnv(), hp, root_path=root)
agent.print_device()

for i in range(10):
    for rd in [False, True]:
        t = time.perf_counter()
        game_conditions = {"PLAYER_FLAP_ACC": -9,
                           "PLAYER_ACC_Y": 1,
                           "pipes_are_random": rd}
        agent = DQNAgent_simple(FlappyBirdEnv(), hp, root_path=root)
        agent.update_env(game_conditions)
        scores, durations = agent.train(name=f"{rd}_{i}", print_progress=True)
        print(f"{rd} {i} : D_250:{np.mean(durations[-250:]):.3f} S*: {np.max(scores)} T: {int((time.perf_counter()-t)//60):02}:{int((time.perf_counter()-t)%60):02}")

