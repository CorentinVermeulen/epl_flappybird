import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from agent_multi import AgentSimple
from flappy_bird_gym.envs import FlappyBirdEnvSimpleMulti as FlappyBirdEnv
from utils import HParams
from tqdm import tqdm
import seaborn as sns
tex_fonts = {
    # Use LaTeX to write all text
    "text.usetex": True,
    "font.family": "serif",
    # Use 10pt font in plots, to match 10pt font in document
    "axes.labelsize": 12,
    "font.size": 12,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10
}
plt.rcParams.update(tex_fonts)
sns.set_context("paper")
sns.set_style("whitegrid")
large_size = (11, 5.5)
small_size = (7, 3.5)
square_size = (5.5, 5.5)


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

"""
I want to test performances of both agent in randomcondition => just testing

For every saved agent, 
    - get the random pipes value
    - load agent
    - run the agent on 100 games and save all durations
    - save the results in a df
    - plot the results
"""
def make_out_root(root):
    if root[-1] != '/':
        root += '/'
    out_root = f"{root}all_testing"
    if not os.path.exists(out_root):
        os.mkdir(out_root)
    return out_root

def init_df(root, var_type='pipes_are_random'):
    df = pd.DataFrame(columns=['id', 'agent_path', 'var', 'duration', 'n_max', 'duration_type' ])
    for dir_name in os.listdir(root):
        dir_path = os.path.join(root, dir_name)
        if os.path.isdir(dir_path):
            param_file = None
            res_file = None
            agent_path = None
            train_dur = None
            n_max = None
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
                    n_max = np.sum(data['durations'] > 1516)
                if id and var and train_dur and agent_path:
                    row = {'id': id,
                           'agent_path': agent_path,
                           'var': var,
                           'duration': train_dur,
                           'n_max': n_max,
                           'duration_type': 'Train'}
                    df.loc[len(df)] = row
    df = df.sort_values(by='id')
    df.reset_index(drop=True, inplace=True)
    return df

def select_best_id(df, var, n=5):
    il = len(df)
    # For every var, I will only keep the best n ids
    best_ids = df[['id', var, 'duration']].groupby(['id', var]).mean().reset_index()
    best_ids = best_ids.sort_values(by='duration', ascending=False).groupby(var).head(n)
    best_df = df[df['id'].isin(best_ids['id'])]
    ol = len(best_df)
    print(f"Reduced from {il} to {ol} rows.")
    return best_df

def test_df(df, n=5):
    dfi = df.copy()
    ## First test: fixed environment
    for i, row in tqdm(df.iterrows()):
        game_context = {'PLAYER_FLAP_ACC': -5,
                        'PLAYER_ACC_Y': 1,
                        'pipes_are_random':True,
                        }

        env = FlappyBirdEnv(n_actions=eval(row['var']))
        env.obs_jumpforce = False
        env.obs_gravity = False

        agent = AgentSimple(env, HParams(baseline_HP))
        agent.update_env(game_context)

        durations = agent.test(row['agent_path'], n)
        for dur in durations:
            row = {'id': row['id'],
                   'agent_path': row['agent_path'],
                   'var': row['var'],
                   'duration': dur,
                   'duration_type': 'test',
                   }
            dfi = pd.concat([dfi, pd.DataFrame(row, index=[0])])



    dfi.to_csv(f"{out_root}/out.csv", index=False)
    return dfi

def make_plots(dfi, xlabel="Number of possible actions"):
    df = dfi.reset_index()
    durations_types = df['duration_type'].unique()
    for dur in durations_types:

        plt.figure(figsize=square_size)
        uniq = df['var'].unique()

        order = np.sort(uniq)
        sns.boxplot(
            data=df.query('duration_type == @dur'),
            x='var',
            y='duration',
            order = order,
            hue='duration_type',
            fill=False,
            flierprops={"marker": "x"},
            medianprops={"color": "r", "linewidth": 2},
            showcaps=False,
            showmeans=True,
        )
        sns.stripplot(data=df.query('duration_type == @dur'),
            x='var',
            y='duration',
            order = order,
            color='black',
            dodge=True,
            alpha=0.5,
            size=2)
        #plt.title(f'Test duration in {dur} condition')
        plt.xlabel(xlabel)
        plt.ylabel('Duration')
        plt.savefig(f"{out_root}/EXP4TEST_boxplot_{dur}.pdf", format = 'pdf', dpi=300, bbox_inches='tight')


    # plt.figure(figsize=(10, 5))
    # sns.displot(
    #     data=df,
    #     x='duration',
    #     hue='var',
    #     kind='kde',
    #     col="duration_type"
    # )
    # plt.savefig(f"{out_root}/EXP1TEST_displot_dur.pdf", format = 'pdf', dpi=300, bbox_inches='tight')


root = f'../../exps/exp_4_test/'
out_root = make_out_root(root)

var_value = 'n_actions'

df = init_df(root, var_value)
df = select_best_id(df, 'var', n=10)

dft = test_df(df, 100)
dft = pd.read_csv(f"{out_root}/out.csv")
dft_mean = dft[['id', 'duration_type', 'var', 'duration']].groupby(['id', 'duration_type', 'var']).mean()
make_plots(dft_mean, xlabel=var_value)

""" 
I want to Select best agent from both configuration
best = Highest number of mean configuration
best2 = Highest mean duration
"""

# print(df[['id', 'duration', 'n_max', 'var']].sort_values(by='duration', ascending=False).groupby('var').head(3))
# print('\n---\n')
# print(df[['id', 'duration', 'n_max', 'var']].sort_values(by='n_max', ascending=False).groupby('var').head(3))

# best_random = 'PARTrue_LR1e-05_R1_0305_123309'
# best_fixed = 'Baseline_LR1e-05_R2_3004_224251'
#
# # Plot both agent
#
# def load_id(root:str, id:str):
#     dir = os.path.join(root, id)
#     for file in os.listdir(dir):
#         if file.startswith('results'):
#             data = pd.read_csv(os.path.join(dir, file))
#             data.columns = ['t', 'durations', 'loss']
#             data["cumsum"] = np.cumsum(data['durations']) / np.arange(1, len(data['durations']) + 1)
#             return data
#
#     return None
#
# agent_rd = load_id(root, best_random)
# agent_fx = load_id(root, best_fixed)
#
# agent_rd_test = dft.query(f"id == '{best_random}'")
# agent_fx_test = dft.query(f"id == '{best_fixed}'")
# agents_test = pd.concat([agent_rd_test, agent_fx_test])
#
# plt.figure(figsize=small_size)
# sns.lineplot(x=agent_fx['t'], y=agent_fx['cumsum'], label='Fixed pipes')
# sns.lineplot(x=agent_rd['t'],y=agent_rd['cumsum'], label='Random pipes')
# plt.ylabel("Duration")
# plt.xlabel("Games played")
# plt.savefig(f"{out_root}/EXP1TEST_avg_best.pdf", format = 'pdf', dpi=300, bbox_inches='tight')
#
# plt.figure(figsize=square_size)
# g=sns.boxplot(
#         data=agents_test.query('duration_type == "test"'),
#         x='var',
#         y='duration',
#         order = ['False', 'True'],
#         fill=False,
#         flierprops={"marker": "x"},
#         medianprops={"color": "r", "linewidth": 2},
#         showcaps=False,
#         showmeans=True,
#     )
# plt.xlabel=('Configuration')
# g.set_xticklabels([f'Fixed pipes ({agents_test.query("var == False")["duration"].mean():.2f})',
#                      f'Random pipes ({agents_test.query("var == True")["duration"].mean():.2f})'])
# plt.savefig(f"{out_root}/EXP1TEST_boxplot_best.pdf", format = 'pdf', dpi=300, bbox_inches='tight')
