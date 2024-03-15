from repo.agents.DQN_agent_simple import DQNAgent_simple
from flappy_bird_gym.envs import CustomEnvSimple as FlappyBirdEnv

env = FlappyBirdEnv()
env.obs_var = ['player_x', 'player_y', 'pipe_center_x', 'pipe_center_y', 'v_dist', 'h_dist', 'player_vel_y']
hparams = {"layer_sizes": [256, 256, 256, 256]}
root = "runs/Rand_ingame/jump_force/second_test/"
training_path = "runs/Rand_ingame/jump_force/second_test/R2_var0.5_simple_0603-181745"

agent_vision = True

dqnAgent = DQNAgent_simple(env, hparams, root_path=root)

dqnAgent.set_nets(training_path)

# Random pipes and stronger gravity
dqnAgent.update_env({"pipes_are_random": True,
                  "PLAYER_FLAP_ACC": -6,
                  "PLAYER_FLAP_ACC_VARIANCE": 0.5
                  })
dqnAgent.show_game(agent_vision=agent_vision, fps=60, stop_at=20)

