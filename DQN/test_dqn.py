from flappy_bird_gym.envs.flappy_bird_env_simple import FlappyBirdEnvSimple
from DQN.DQN_Training import DQN
from itertools import count
import torch

env = FlappyBirdEnvSimple()
# Get number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state observations
state = env.reset()
n_observations = len(state)


for i in range(1,10):
    file_dqn = "fb_dqn_at_%s000.pt"%i
    # New network
    policy_net = DQN(n_observations, n_actions)
    policy_net.load_state_dict(torch.load(file_dqn))
    policy_net.eval()
    score_log = []
    # Play the game n times
    for _ in range(10):
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        for t in count():
            action = policy_net(state).max(1)[1].view(1, 1)
            observation, reward, terminated, info = env.step(action.item())
            done = terminated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)

            # Move to the next state
            state = next_state

            if done:
                score_log.append(info['score'])
                break
    print(f"Best score at {i}000: {score_log}")
    del policy_net


