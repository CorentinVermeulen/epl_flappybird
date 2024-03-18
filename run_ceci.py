from agents.DQN_agent_ceci import DQNAgent_simple_cuda as DQNAgent_simple
from flappy_bird_gym.envs import CustomEnvSimple as FlappyBirdEnv
import pandas as pd

root = 'TEST/'

hp = {"layer_sizes": [256, 256, 256, 256],
      "EPOCHS": 1000,
      "BATCH_SIZE": 256,
      "EPS_DECAY": 2000}

conditions = {"PLAYER_FLAP_ACC": -6,
              "PLAYER_ACC_Y": 1,
              "pipes_are_random":True}

env = FlappyBirdEnv()
env.obs_var = ['player_x', 'player_y', 'pipe_center_x', 'pipe_center_y', 'v_dist', 'h_dist', 'player_vel_y']
env.reset()

n_obs = env.observation_space.shape[0]
n_actions = env.action_space.n

agent = DQNAgent_simple(FlappyBirdEnv(), hp, root_path=root)
agent.print_device()

for rd in [False, True]:
    conditions = {"PLAYER_FLAP_ACC": -6,
                  "PLAYER_ACC_Y": 1,
                  "pipes_are_random": rd}
    log_scores = {}
    log_durations = {}

    for i in range(10):
        agent = DQNAgent_simple(FlappyBirdEnv(), hp, root_path=root)
        agent.update_env(conditions)
        scores, durations = agent.train(name=f"rd{rd*1}_{i}")
        log_scores[f"training{rd*1}_{i}"] = scores
        log_durations[f"training{rd*1}_{i}"] = durations
    print('\rLoop with rd =', rd, 'done')

    pd.DataFrame(log_scores).to_csv(f"{root}scores_rd{rd*1}.csv")
    pd.DataFrame(log_durations).to_csv(f"{root}durations_rd{rd*1}.csv")
