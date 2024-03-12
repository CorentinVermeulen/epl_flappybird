from repo.agents.DQN_agent_simple import DQNAgent_simple
from flappy_bird_gym.envs import CustomEnvSimple as FlappyBirdEnv
import pandas as pd
import numpy as np

# ENV SETUP
env = FlappyBirdEnv()
env.obs_var = ['player_x', 'player_y', 'pipe_center_x', 'pipe_center_y', 'v_dist', 'h_dist', 'player_vel_y']
n_obs = env.observation_space.shape[0]
n_actions = env.action_space.n
hparams = {"layer_sizes": [256, 256, 256, 256],
           "EPOCHS": 1000,
           "BATCH_SIZE": 256,
           "EPS_DECAY": 2000}

root = "runs/rand_pregame/"
for i in range(2):
    # Easier game Not random
    agent = DQNAgent_simple(env, hparams, root_path=root)
    agent.update_env({"PLAYER_FLAP_ACC": -6, "PLAYER_ACC_Y": 1, "pipes_are_random":False})
    scores, durations, info = agent.train(name="vpos_not_random")
    path = agent.training_path
    df = pd.DataFrame({"scores": scores, "durations": durations})
    df.to_csv(path + "/results.csv")

    Nsm = np.sum(np.array(scores[-333:]) == 20)
    Dm = np.mean(durations[-333:])
    cumsum = list(np.cumsum(scores))
    Ntosm = cumsum.index(10) if 10 in cumsum else 0

    print(f"\rNot Random & {Nsm} & {Dm:.2f} & {Ntosm:} \\\\")


    # Easier game random vpos
    agent = DQNAgent_simple(env, hparams, root_path=root)
    agent.update_env({"PLAYER_FLAP_ACC": -6, "PLAYER_ACC_Y": 1, "pipes_are_random":True})
    scores, durations, info = agent.train(name="vpos_random")
    path = agent.training_path
    df = pd.DataFrame({"scores": scores, "durations": durations})
    df.to_csv(path + "/results.csv")

    Nsm = np.sum(np.array(scores[-333:]) == 20)
    Dm = np.mean(durations[-333:])
    cumsum = list(np.cumsum(scores))
    Ntosm = cumsum.index(10) if 10 in cumsum else 0

    print(f"\rRandom & {Nsm} & {Dm:.2f} & {Ntosm:} \\\\")


