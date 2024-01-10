from itertools import count
from flappy_bird_gym.envs.custom_env_simple import CustomEnvSimple as FlappyBirdEnv
from pytorch_DQN import DQN
import time


import torch

def show_model_game(env, policy_net, num_episodes=5, fps=60, agent_vision=False, wait=True):

    if agent_vision:
        env.set_custom_render()

    sleep_time = 1 / fps
    if wait:
        res = input("Do you want to see the model play? (y/n) ")
    else:
        res = "y"
    if res == "y":
        for i_episode in range(num_episodes):

            total_reward = torch.tensor([0])
            # Initialize the environment and get it's state
            state = env.reset()
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            for t in count():
                action = policy_net(state).max(1)[1].view(1, 1)
                observation, reward, terminated, info = env.step(action.item())

                q_0 = policy_net(state)[0][0].item()
                q_sum = q_0 + policy_net(state)[0][1].item()
                stats = q_0/q_sum # dont need q_1/q_sum because it's 1 - q_0/q_sum

                env.render(stats=stats)
                time.sleep(sleep_time)

                reward = torch.tensor([reward])
                total_reward = total_reward + reward
                done = terminated

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)

                # Move to the next state
                state = next_state

                if done:
                    print(f"score: {env._game.score}, total reward: {total_reward.item()}.\nInfo= {info}")
                    break
            time.sleep(1)
        else:
            print("---- END SHOWING RESULTS ----\n ")

env = FlappyBirdEnv()

n_actions = env.action_space.n
state = env.reset()
n_observations = len(state)


file_dqn = "/Users/corentinvrmln/Desktop/memoire/flappybird/repo/DQN/dqn_log/01_07_12_20/E880_S50.pt"

with torch.no_grad():
    policy_net = DQN(n_observations, n_actions)
    policy_net.load_state_dict(torch.load(file_dqn))
    policy_net.eval()
    show_model_game(env, policy_net, num_episodes=1,fps=40, wait=False, agent_vision=True)