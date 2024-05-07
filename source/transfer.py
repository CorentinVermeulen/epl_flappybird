import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from agent_simple import AgentSimple
from flappy_bird_gym.envs import FlappyBirdEnvSimpleFast as FlappyBirdEnv
from utils import HParams
from tqdm import tqdm
import seaborn as sns

baseline_HP = {"EPOCHS": 1000,
               "MEMORY_SIZE": 100000,
               "EPS_START": 0.9,
               "EPS_END": 0.001,
               "EPS_DECAY": 2000,
               "TAU": 0.01,
               "LAYER_SIZES": [256, 256, 256, 256],
               "GAMMA": 0.99,
               "UPDATE_TARGETNET_RATE": 1,
               "BATCH_SIZE": 256,
               "LR": 1e-4,
               }

# I want to evaluate the performance of an agent trained in random environments on fixed environments.


# for all id import the trained agent
# for all id import the variance (jumpforce or gravity) of the environment
# Create a new env with the same variance and test
# Create a new env with (jumpforce or gravity) += variance  and test



# out_df:
# id, train_score, train_std, var, var_type, test_score_rd, test_std_rd, test_score_fixed, test_std_fixed

def make_out_root(root):
    if root[-1] != '/':
        root += '/'
    out_root = f"{root}all_transfers"
    if not os.path.exists(out_root):
        os.mkdir(out_root)
    return out_root

def init_df(root, var_type):
    df = pd.DataFrame(columns=['id', 'agent_path', 'var', 'var_type', 'duration', 'duration_type' ])
    for dir_name in os.listdir(root):
        dir_path = os.path.join(root, dir_name)
        if os.path.isdir(dir_path):
            param_file = None
            res_file = None
            agent_path = None
            train_dur = None
            train_std = None
            for file in os.listdir(dir_path):
                if file.startswith('param_log'):
                    param_file = os.path.join(dir_path, file)
                elif file.startswith('results'):
                    res_file = os.path.join(dir_path, file)
                elif file.startswith('policy'):
                    agent_path = os.path.join(dir_path, file)

            if param_file and res_file:
                with open(param_file, 'r') as file:
                    content = file.read()
                    id = re.search(fr'id: (.*)', content).group(1)
                    var = re.search(fr'{var_type}: (.*)', content) .group(1)

                data = pd.read_csv(res_file)
                if len(data) < 1000:
                    print(f"File {res_file} in {dir_name} has less than 1000 rows. ({len(data)})")
                else:
                    train_dur = data['durations'].mean()
                if id and var and train_dur and agent_path:
                    row = {'id': id, 'agent_path': agent_path, 'var': var, 'var_type': var_type, 'duration': train_dur, 'duration_type': 'train'}
                    df.loc[len(df)] = row
    df = df.sort_values(by='id')
    df.reset_index(drop=True, inplace=True)
    return df

def test_df(df, n=50):
    dfi = df.copy()
    ## First test: random environment
    for i, row in tqdm(df.iterrows()):
        game_context = {'PLAYER_FLAP_ACC': -5,
                        'PLAYER_ACC_Y': 1,
                        row['var_type']: eval(row['var']),
                        }
        game_context['pipes_are_random'] = True

        env = FlappyBirdEnv()
        env.obs_jumpforce = False
        env.obs_gravity = False

        agent = AgentSimple(env, HParams(baseline_HP))
        agent.update_env(game_context)

        durations = agent.test(row['agent_path'], n)
        for dur in durations:
            row = {'id': row['id'],
                   'agent_path': row['agent_path'],
                   'var': row['var'],
                   'var_type': row['var_type'],
                   'duration': dur,
                   'duration_type': 'test_origin'}
            dfi.loc[len(dfi)] = row


    for i, row in tqdm(df.iterrows()):
        if row['var_type'] == 'PLAYER_FLAP_ACC_VARIANCE':
            game_context = {'PLAYER_FLAP_ACC': -5 * (1 + eval(row['var'])),
                            'PLAYER_ACC_Y': 1,
                            'pipes_are_random': True,
                            }

        elif row['var_type'] == 'GRAVITY_VARIANCE':
            game_context = {'PLAYER_FLAP_ACC': -5,
                            'PLAYER_ACC_Y': 1 * (1 + eval(row['var'])),
                            'pipes_are_random': True,
                            }
        elif row['var_type'] == 'pipes_are_random':
            game_context = {'PLAYER_FLAP_ACC': -5,
                            'PLAYER_ACC_Y': 1 ,
                            'pipes_are_random': False,
                            }
        else:
            raise ValueError(f"Unknown var_type: {row['var_type']}")

        env = FlappyBirdEnv()
        env.obs_jumpforce = False
        env.obs_gravity = False

        agent = AgentSimple(env, HParams(baseline_HP))
        agent.update_env(game_context)

        durations = agent.test(row['agent_path'], n)

        for dur in durations:
            row = {'id': row['id'],
                   'agent_path': row['agent_path'],
                   'var': row['var'],
                   'var_type': row['var_type'],
                   'duration': dur,
                   'duration_type': 'test_fixed'}
            dfi.loc[len(dfi)] = row

    dfi.to_csv(f"{out_root}/out.csv", index=False)
    return dfi

def make_plots(dfi, xlabel="Jump Force Variance"):
    df = dfi.reset_index()
    for duration_type in df['duration_type'].unique():

        plt.figure(figsize=(10, 5))
        order = np.sort(df['var'].unique())
        sns.boxplot(
            data=df.query('duration_type == @duration_type'),
            x='var',
            y='duration',
            order = order,
            hue='duration_type',
            fill=False,
            flierprops={"marker": "x"},
            medianprops={"linewidth": 2},
            showcaps=False,
            showmeans=True,
        )

        plt.title('Test duration in random environments')
        plt.xlabel(xlabel)
        plt.ylabel('Duration')
        plt.savefig(f"{out_root}/test_dur_{duration_type}.png")

# for all id import the trained agent

vars = [
    ('baseline', 'Baseline'),
    ('pipes_are_random', 'Random pipes'),
    ('PLAYER_FLAP_ACC_VARIANCE', 'Jump Force Variance'),
    ('GRAVITY_VARIANCE', 'Gravity Variance'),
]
for i in range(1,4):
    root = f'../../exps/exp_{i}_f/'
    out_root = make_out_root(root)

    focus = vars[i]

    # df = init_df(root, focus[0])
    # df = test_df(df, 50)

    df = pd.read_csv(f"{out_root}/out.csv")
    df = df[['id', 'duration_type', 'var', 'duration']].groupby(['id', 'duration_type']).mean()

    make_plots(df, xlabel=focus[1])




