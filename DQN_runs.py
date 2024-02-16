

import time
import numpy as np
from DQN_agent_simple import DQNAgent
from flappy_bird_gym.envs import CustomEnvSimple as FlappyBirdEnv


t=time.perf_counter()
env = FlappyBirdEnv()
env.obs_var = ['player_x', 'player_y', 'pipe_center_x', 'pipe_center_y', 'v_dist', 'h_dist', 'player_vel_y']
env.rewards = {"alive": 0.1, "pass_pipe": 1, "dead": -1, 'score': 0}
hparams = {"EPOCHS": 1500, "BATCH_SIZE": 64, "layer_sizes": [64, 128, 256, 256]}
qdnAgent = DQNAgent(env, hparams)

#Show game
# path = "/Users/corentinvrmln/Desktop/memoire/flappybird/repo/DQN/runs/Comp/0902-104614_B64"
# qdnAgent.load_policy_net(path + "/policy_net.pt")
#
# dic = {'PLAYER_FLAP_ACC': -6, 'PIPE_VEL_X': -4, 'PLAYER_ACC_Y': 1, "random_pipes":False}
# qdnAgent.update_env(dic)
# qdnAgent.show_game(agent_vision=True, fps=60)



# # Training
# print("\n\tTRAINING ...")
# scores, durations, end_dic, path = qdnAgent.train()
# print(f"Mean training duration: {np.mean(durations)}")
#
# # Testing (only possible if score is > 2)
# if max(scores) > 2:
#     print("\n\tTESTING ...")
#     qdnAgent.load_policy_net(path + "/policy_net.pt")
#
#     print("Testing agent after training" + str(qdnAgent.test()))
#
#     print("\n\tRE-TRAINING ...")
#     # Switch game parameters
#     qdnAgent.env._game.switch_defined_pipes()
#     print("Testing agent after changes and before retraining" + str(qdnAgent.test()))
#
#     qdnAgent.retrain(path=path,
#                      memory_load=True,
#                      eps_start=0.01,
#                      epochs=1500)
#
#     print("Testing agent after changes and after retraining" + str(qdnAgent.test()))

# Show game
# qdnAgent.load_policy_net(path + "/policy_net.pt")
# qdnAgent.show_game(agent_vision=True, fps=60)



## LEARNING WITH STOP AT 50 SCORE REACHED

# Learn model until reaching score = 30
scores, durations, end_dic, path = qdnAgent.train()
print(
    f"Mean training duration: {np.mean(durations):.2f} - Mean score: {np.mean(scores):.2f} - Best score: {max(scores)} - n_episodes: {end_dic['n_episodes']}")

# Save model, memory and iterations to reach s=30
qdnAgent.load_policy_net(path + "/policy_net.pt")
print("Testing agent after training before change params" + str(qdnAgent.test()))
dic = {'PLAYER_FLAP_ACC': -6, 'PIPE_VEL_X': -4, 'PLAYER_ACC_Y': 1, "random_pipes": False}
# Change param
params_order = [{'PLAYER_FLAP_ACC':-6}, {'PLAYER_ACC_Y':0.5}, {'PLAYER_FLAP_ACC':-13}, {'PLAYER_ACC_Y':1.5}]





for i, params in enumerate(params_order):
    print("\n Changing params to "+ str(params))
    n_params = dic.copy()
    n_params.update(params)
    qdnAgent.update_env(n_params)

    # Check new score
    qdnAgent.load_policy_net(path + "/policy_net.pt")
    print("\tTesting agent after training after change params" + str(qdnAgent.test()))

    # Retrain with no memory load and no model load
    print("\n\tno_memory_no_model:")
    scores, durations, end_dic, _ = qdnAgent.retrain(name="no_memory_no_model_"+str(i)+f"_{str(params)}",
                                                     path=path,
                                                     model_load=False,
                                                     memory_load=False,
                                                     eps_start=0.01,
                                                     epochs=1500)
    print("\r")
    print(
        f"\tMean training duration: {np.mean(durations):.2f} - Mean score: {np.mean(scores):.2f} - Best score: {max(scores)} - n_episodes: {end_dic['n_episodes']}")
    print("\n")

    # Retrain with no memory load and model load
    print("\tno_memory_with_model:")
    scores, durations, end_dic, _ = qdnAgent.retrain(name="no_memory_with_model_"+str(i)+f"_{str(params)}",
                                                     path=path,
                                                     model_load=True,
                                                     memory_load=False,
                                                     eps_start=0.01,
                                                     epochs=1500)
    print("\r")
    print(
        f"\tMean training duration: {np.mean(durations):.2f} - Mean score: {np.mean(scores):.2f} - Best score: {max(scores)} - n_episodes: {end_dic['n_episodes']}")
    print("\n")

    # Retrain with memory load and no model load
    print("\n\twith_memory_no_model:")
    scores, durations, end_dic, _ = qdnAgent.retrain(name="with_memory_no_model_"+str(i)+f"_{str(params)}",
                                                     path=path,
                                                     model_load=False,
                                                     memory_load=True,
                                                     eps_start=0.01,
                                                     epochs=1500)
    print("\r")
    print(
        f"\tMean training duration: {np.mean(durations):.2f} - Mean score: {np.mean(scores):.2f} - Best score: {max(scores)} - n_episodes: {end_dic['n_episodes']}")
    print("\n")

    # Retrain with memory load and model load
    print("\n\twith_memory_with_model:")
    scores, durations, end_dic, _ = qdnAgent.retrain(name="with_memory_with_model_"+str(i)+f"_{str(params)}",
                                                     path=path,
                                                     model_load=True,
                                                     memory_load=True,
                                                     eps_start=0.01,
                                                     epochs=1500)
    print("\r")
    print(
        f"\tMean training duration: {np.mean(durations):.2f} - Mean score: {np.mean(scores):.2f} - Best score: {max(scores)} - n_episodes: {end_dic['n_episodes']}")

print("\nTime: ", time.perf_counter()-t)