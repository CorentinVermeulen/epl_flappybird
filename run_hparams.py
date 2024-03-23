import numpy as np
from agents.DQN_agent_simple import DQNAgent_simple
from flappy_bird_gym.envs import CustomEnvSimple as FlappyBirdEnv
from torch.utils.tensorboard import SummaryWriter

# most simple agent

root = "runs/Hparams/"
writer_h = SummaryWriter(root)

env = FlappyBirdEnv()
env.obs_var = ['player_y', 'pipe_center_x', 'pipe_center_y', 'h_dist',
               'player_vel_y']  # ['player_x', 'player_y', 'pipe_center_x', 'pipe_center_y', 'v_dist', 'h_dist', 'player_vel_y']
env.rewards = {"alive": 0.1, "pass_pipe": 1, "dead": -1, 'score': 0}
env.reset()
n_obs = env.observation_space.shape[0]
n_actions = env.action_space.n

# Hyperparameters I want to test
batch_sizes = [64, 128, 256, 512]
taus = [0.01, 0.05, 0.1, 0.15, 0.5]
gammas = [0.999, 0.99, 0.95, 0.9, 0.5]
update_freqs = [1, 3, 7, 10, 15]


for batch_size in [128, 64]:
    for tau in [0.05, 0.1, 0.15]:
        for gamma in [0.99, 0.95, 0.9]:
            for eps_decay in [2000,4000]:
                hparams_cst = {"EPOCHS": 1501,
                               "layer_sizes": [64, 128, 128, 128],
                               "BATCH_SIZE": batch_size,
                               "EPS_DECAY": eps_decay,
                               "TAU": tau,
                               "GAMMA": gamma,
                               }

                dqnAgent = DQNAgent_simple(env, hparams_cst, root_path=root)
                dqnAgent.update_env({"PLAYER_FLAP_ACC": -6, "PLAYER_ACC_Y": 1})  # Make game easier
                scores, durations, end_dic = dqnAgent.train()
                test_d = dqnAgent.test()["duration"]
                hparams_cst['layer_sizes'] = str(hparams_cst['layer_sizes'])
                writer_h.add_hparams(hparams_cst, {"mean_duration": float(np.mean(durations)),
                                                   "max_score": max(scores),
                                                   "n_max_score": int(np.sum(np.array(scores) == 20)),
                                                   "test_d": test_d})
