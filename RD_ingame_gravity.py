import pandas as pd
from repo.agents.DQN_agent_simple import DQNAgent_simple
from flappy_bird_gym.envs import CustomEnvSimple as FlappyBirdEnv
import numpy as np

root = "runs/Rand_ingame/gravity_sin/"

env = FlappyBirdEnv()
env.obs_var = ['player_x', 'player_y',
               'pipe_center_x', 'pipe_center_y',
               'v_dist', 'h_dist',
               'player_vel_y',
               'gravity']
n_obs = env.observation_space.shape[0]
n_actions = env.action_space.n
hparams = {"layer_sizes": [256, 256, 256, 256],
           "EPOCHS": 1000,
           "BATCH_SIZE": 256,
           "EPS_DECAY": 2000}

print(f"Period & Amplitude & Nsm & Dm & Ntosm \\\\")

#Random jump
for period in [100]:
    for amplitude in [0.5, 0.75]:

        env.reset()
        agent = DQNAgent_simple(env, hparams, root_path=root)
        agent.update_env({"pipes_are_random": True,
                          "GRAVITY_SINUS_AMPLITUDE": amplitude,
                          "GRAVITY_SINUS_PERIOD": period
                          })
        scores, durations, dic = agent.train(name=f"R1_p{period}_a{amplitude}")
        path = agent.training_path
        df = pd.DataFrame({"scores": scores, "durations": durations})
        df.to_csv(path + "/results.csv")

        Nsm = np.sum(np.array(scores[-333:]) == 20)
        Dm = np.mean(durations[-333:])
        cumsum = list(np.cumsum(scores))
        Ntosm = cumsum.index(10) if 10 in cumsum else 0

        print(f"\r{period} & {amplitude} &{Nsm} & {Dm:.2f} & {Ntosm:} \\\\")

# Reference test
# amplitude = 0
# period = 1
# env.reset()
# agent = DQNAgent_simple(env, hparams, root_path=root)
# agent.update_env({"pipes_are_random": True,
#                   "GRAVITY_SINUS_AMPLITUDE": amplitude,
#                   "GRAVITY_SINUS_PERIOD": period
#                   })
# scores, durations, dic = agent.train(name=f"R1_p{period}_a{amplitude}")
# path = agent.training_path
# df = pd.DataFrame({"scores": scores, "durations": durations})
# df.to_csv(path + "/results.csv")
#
# Nsm = np.sum(np.array(scores[-333:]) == 20)
# Dm = np.mean(durations[-333:])
# cumsum = list(np.cumsum(scores))
# Ntosm = cumsum.index(10) if 10 in cumsum else 0
#
# print(f"\r{period} & {amplitude} &{Nsm} & {Dm:.2f} & {Ntosm:} \\\\")



## SHOW GAME ##
# env.reset()
# agent = DQNAgent_simple(env, hparams, root_path=root)
# agent.set_nets("runs/Rand_ingame/gravity_sin/R1_p1_a0_simple_1103-125956")
# agent.update_env({"pipes_are_random": True,
#                   "GRAVITY_SINUS_AMPLITUDE": 0,
#                   "GRAVITY_SINUS_PERIOD": 100,
#                   })
# agent.show_game()

