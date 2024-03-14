from repo.agents.DQN_agent_simple import DQNAgent_simple
from flappy_bird_gym.envs import CustomEnvSimple as FlappyBirdEnv

erv = FlappyBirdEnv()
env.obs_var = ['player_x', 'player_y', 'pipe_center_x', 'pipe_center_y', 'v_dist', 'h_dist', 'player_vel_y']
hparams = {"layer_sizes": [256, 256, 256, 256]}
root = "runs/show_games/"
training_path = "runs/rand_ingame/jump_simple_0503-170844"

agent_vision = False

dqnAgent = DQNAgent_simple(env, hparams, root_path=root)

dqnAgent.set_nets(training_path)

# # Normal game
# dqnAgent.show_game(agent_vision=agent_vision, fps=60, stop_at=20)
#
# # Random pipes
# d = {"pipes_are_random": True}
# dqnAgent.update_env(d)
# dqnAgent.show_game(agent_vision=agent_vision, fps=60, stop_at=20)


# Random pipes and stronger gravity
dqnAgent.update_env({"pipes_are_random": True,
                  "PLAYER_FLAP_ACC": -6,
                  "PLAYER_FLAP_ACC_VARIANCE": 2
                  })
dqnAgent.show_game(agent_vision=agent_vision, fps=60, stop_at=20)

# # Random pipes and stronger gravity
# d = {"PLAYER_FLAP_ACC": -11,
#      "PLAYER_ACC_Y": 1,
#      "pipes_are_random": True}
#
# dqnAgent.update_env(d)
# dqnAgent.show_game(agent_vision=agent_vision, fps=60, stop_at=20)
