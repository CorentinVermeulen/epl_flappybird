from agent_RGB import AgentRGB
from utils import HParams, make_experiment_plot
from flappy_bird_gym.envs import CustomEnvRGB as FlappyBirdEnv

env = FlappyBirdEnv()
baseline_HP = {"EPOCHS": 50,
               "MEMORY_SIZE": 100000,
               "EPS_START": 0.9,
               "EPS_END": 0.001,
               "EPS_DECAY": 3000,
               "LAYER_SIZES": [256, 256, 256, 256],
               "GAMMA": 0.999,
               "UPDATE_TARGETNET_RATE": 1,
               "BATCH_SIZE": 256,
               "LR": 1e-5,
               "TAU": 0.01,
               }

hparams = HParams(baseline_HP)
agent = AgentRGB(env, hparams)

print(agent.__dict__)