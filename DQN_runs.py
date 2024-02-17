import pandas as pd
import time
import numpy as np
from DQN_agent_simple import DQNAgent_simple
from flappy_bird_gym.envs import CustomEnvSimple as FlappyBirdEnv

t = time.perf_counter()
env = FlappyBirdEnv()
env.obs_var = ['player_x', 'player_y', 'pipe_center_x', 'pipe_center_y', 'v_dist', 'h_dist', 'player_vel_y']
env.rewards = {"alive": 0.1, "pass_pipe": 1, "dead": -1, 'score': 0}
hparams = {"EPOCHS": 1500,
           "BATCH_SIZE": 64,
           "EPS_DECAY": 4000,
           "layer_sizes": [64, 128, 256, 256]}
root = "runs/Update_rate/"
dqnAgent = DQNAgent_simple(env, hparams, root_path="runs/Update_rate/")

df = pd.DataFrame(
    columns=['Name', 'n_to_30', 'mean_duration', 'max_score', 'test_score', 'test_duration', 'total_time'])


def log_df(df, name, scores, durations, end_dic, test_dic, t):
    df.loc[len(df)] = {'Name': name,
                       'n_to_30': end_dic['n_to_30'],
                       'mean_duration': np.mean(durations),
                       'max_score': max(scores),
                       'test_score': test_dic['score'],
                       'test_duration': test_dic['duration'],
                       'total_time': time.perf_counter() - t
                       }


for update_rate in [1, 10, 25]:
    for i in range(3):
        ti = time.perf_counter()
        dqnAgent.reset()
        print(f"\nUpdate rate: {update_rate}")
        hparams["UPDATE_TARGETNET_RATE"] = update_rate
        dqnAgent.set_hyperparameters(hparams)

        scores, durations, end_dic = dqnAgent.train()
        test_dic = dqnAgent.test()

        name = f"U{update_rate}_{i}"
        t = time.perf_counter() - ti
        log_df(df, name, scores, durations, end_dic, test_dic, t)
        df.to_csv(f"{root}Update_rate.csv")

print(df)
df.to_csv(f"{root}Update_rate.csv")

# dqnAgent.train()
b = 0
if b:
    # Training
    print("\n\tTRAINING ...")
    scores, durations, end_dic = dqnAgent.train()
    training_path = dqnAgent.training_path
    print(f"Mean training duration: {np.mean(durations)}")

    # Testing (only possible if score is > 2)
    if max(scores) > 2:
        print("\n\tTESTING ...")
        dqnAgent.reset()
        dqnAgent.set_nets(training_path)
        print("Testing agent after training" + str(dqnAgent.test()))

        print("\n\tCHANGING PARAMETERS...")
        d = {"PLAYER_FLAP_ACC": -6,
             "pipes_are_random": True}
        dqnAgent.update_env(d)
        print(f"New parameters: {d}")

        print("\n\tDIRECT TEST  ...")
        dqnAgent.reset()
        dqnAgent.set_nets(training_path)
        print("Testing agent before retraining" + str(dqnAgent.test()))

        print("\n\tRE-TRAINING ...")
        dqnAgent.reset()
        dqnAgent.retrain(training_path, name="testing", load_network=True, load_memory=True, eps_start=0.5, epochs=1500)
        retraining_path = dqnAgent.training_path

        print("\n\tFINAL TEST  ...")
        dqnAgent.reset()
        dqnAgent.set_nets(retraining_path)
        print("Testing agent after retraining" + str(dqnAgent.test()))

# Show game
a = 0
if a:
    # training_path = dqnAgent.training_path
    training_path = "runs/Comp/simple_1602-155157_retrain_testing"
    dqnAgent.set_nets(training_path)
    dqnAgent.show_game(agent_vision=True, fps=60)

    d = {"PLAYER_FLAP_ACC": -9,
         "PLAYER_ACC_Y": 1,
         "pipes_are_random": True}

    dqnAgent.update_env(d)

    dqnAgent.show_game(agent_vision=True, fps=60)
