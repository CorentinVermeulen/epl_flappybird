
import time
import numpy as np
from DQN_agent_simple import DQNAgent_simple
from flappy_bird_gym.envs import CustomEnvSimple as FlappyBirdEnv

t=time.perf_counter()
env = FlappyBirdEnv()
env.obs_var = ['player_x', 'player_y', 'pipe_center_x', 'pipe_center_y', 'v_dist', 'h_dist', 'player_vel_y']
env.rewards = {"alive": 0.1, "pass_pipe": 1, "dead": -1, 'score': 0}
hparams = {"EPOCHS": 1500,
           "BATCH_SIZE": 64,
           "EPS_DECAY": 4000,
           "layer_sizes": [64, 128, 256, 256]}
dqnAgent = DQNAgent_simple(env, hparams)
# dqnAgent.train()

if True:
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

# Show game
a = False
if a:
    # training_path = dqnAgent.training_path
    training_path = "runs/Comp/simple_1602-125027_retrain_testing"
    dqnAgent.set_nets(training_path)
    dqnAgent.show_game(agent_vision=True, fps=60)

    d = {"PLAYER_FLAP_ACC":  -9 ,
         "PLAYER_ACC_Y": 1,
         "pipes_are_random": True}

    dqnAgent.update_env(d)

    dqnAgent.show_game(agent_vision=True, fps=60)
