import time
from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import scipy.stats as stats
import plotly.express as px
from IPython.display import Image
import seaborn as sns

from pathlib import Path
import pandas as pd
import re
import glob
import itertools


def make_out_root(root):
    out_root = f"{root}/results"
    if not os.path.exists(out_root):
        os.mkdir(out_root)
    return out_root


def get_stacked_df(root, params_under_study):
    def extract_params(root, params_list):
        params = {}
        for dir_name in os.listdir(root):
            dir_path = os.path.join(root, dir_name)
            if os.path.isdir(dir_path):
                file_path = os.path.join(dir_path, 'param_log.txt')
                if os.path.isfile(file_path):
                    with open(file_path, 'r') as file:
                        content = file.read()
                        d = {}
                        for param in params_list:
                            search = re.search(fr'{param}: (.*)', content)
                            if search:
                                d[param] = search.group(1)
                        params[dir_name] = d
        return params

    params = extract_params(root, params_under_study)

    out_df = pd.DataFrame(columns=["id", "t"] + params_under_study + ["loss", "duration"])
    for dir_name, param in params.items():
        dir_path = os.path.join(root, dir_name)
        for file in os.listdir(dir_path):
            if file.endswith('.csv'):
                df = pd.DataFrame(columns=["id", "t"] + params_under_study + ["loss", "duration"])
                data = pd.read_csv(os.path.join(root, dir_name, file))
                if 'durations' in data.columns:
                    df['duration'] = data['durations']
                    df['cumsum'] = np.cumsum(data['durations']) / np.arange(1, len(data['durations']) + 1)
                    df['running_mean'] = data['durations'].rolling(window=100).mean()
                    df['loss'] = data['loss']
                    df['id'] = dir_name
                    df['t'] = data.index
                    for p in params_under_study:
                        df[p] = param[p]
                    out_df = pd.concat([out_df, df])
    return out_df


def normalize_stacked_df(df, bys):
    ndf = df.copy()
    max_duration = ndf.groupby(bys)['duration'].transform('max')
    max_cumsum = ndf.groupby(bys)['cumsum'].transform('max')
    max_running_mean = ndf.groupby(bys)['running_mean'].transform('max')

    ndf['duration'] /= max_duration
    ndf['cumsum'] /= max_cumsum
    ndf['running_mean'] /= max_running_mean

    return ndf


# 1 AVG PLOT and RM PLOT
def avg_plot(df, out_root, bys, plot_args={}, name=None):
    order = np.sort(df[bys[0]].unique())
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df,
                 x='t',
                 y='cumsum',
                 hue=bys[0],
                 hue_order=order,
                 **plot_args)
    plt.xlabel('Games played')
    plt.ylabel('Duration')
    plt.suptitle("")
    plt.title('')
    out_name = 'avg_plot'
    if name:
        out_name = f'{out_name}_{name}'
    plt.savefig(f'{out_root}/{out_name}.png')
    #plt.show()


def running_mean_plot(idf, bys, out_root, plot_args={}, name=None):
    df = idf.copy()
    df = df.dropna()
    plt.figure(figsize=(10, 6))
    order = np.sort(df[bys[0]].unique())

    sns.lineplot(data=df,
                 x='t',
                 y='running_mean',
                 hue=bys[0],
                 hue_order=order,
                 **plot_args)
    plt.xlabel('Games played')
    plt.ylabel('Duration')
    plt.suptitle("")
    plt.title('')
    out_name = 'running_mean_plot'
    if name:
        out_name = f'{out_name}_{name}'
    plt.savefig(f'{out_root}/{out_name}.png')
    #plt.show()


# 2 MEAN DURATION BOXPLOT
def mean_dur_boxplot(idf, bys, out_root, plot_args={}, name=None):
    df = idf.copy()
    df = df[['id'] + bys + ['duration']].groupby(['id'] + bys).mean()
    order = np.sort(df.index.get_level_values(bys[0]).unique())

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df,
                x=bys[0],
                order=order,
                y='duration',
                fill=False,
                flierprops={"marker": "x"},
                medianprops={"color": "r", "linewidth": 2},
                showcaps=False,
                **plot_args
                )
    plt.xlabel(bys[0])
    plt.ylabel('Duration')
    plt.suptitle("")
    plt.title('')
    out_name = 'mean_dur_boxplot'
    if name:
        out_name = f'{out_name}_{name}'
    plt.savefig(f'{out_root}/{out_name}.png')
    #plt.show()


# 3 N GAMES MAX DURATION BOXPLOT
def n_max_boxplot(idf, bys, out_root, plot_args={}, name=None):
    df = idf.copy()
    df['maxs'] = df.groupby(bys)['duration'].transform('max')
    df['greater'] = df['duration'] >= df['maxs'] * 0.999
    n_maxs = df.groupby(['id'] + bys)['greater'].sum()
    n_maxs = n_maxs.reset_index()
    order = np.sort(n_maxs[bys[0]].unique())
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=n_maxs,
                x=bys[0],
                order=order,
                y='greater',
                fill=False,
                flierprops={"marker": "x"},
                medianprops={"color": "r", "linewidth": 2},
                showcaps=False,
                **plot_args
                )
    plt.xlabel(bys[0])
    plt.ylabel('Number of games with max duration')
    plt.suptitle("")
    plt.title('')
    out_name = 'n_max_boxplot'
    if name:
        out_name = f'{out_name}_{name}'
    plt.savefig(f'{out_root}/{out_name}.png')
    #plt.show()


# ============================================================================= #

root = '../../exps/exp_0/'
out_root = make_out_root(root)

params_under_study = ['LR',]

bys = list(params_under_study[0])

t = time.perf_counter()
df = get_stacked_df(root, params_under_study)
df = df.query(" LR == '1e-05' ")
ndf = normalize_stacked_df(df, params_under_study)
print(f"Data loaded in {time.perf_counter() - t:.2f} seconds.")



t = time.perf_counter()
avg_plot(df, bys=bys, out_root=out_root, plot_args={}, name=None )
print(f"Plot 1 done in {time.perf_counter() - t:.2f} seconds.")

t = time.perf_counter()
avg_plot(ndf,bys=bys,out_root=out_root,plot_args={},name='normalized')
print(f"Plot 1b done in {time.perf_counter() - t:.2f} seconds.")



t = time.perf_counter()
running_mean_plot(df,bys=bys,out_root=out_root,plot_args={},name=None)
print(f"Plot 2 done in {time.perf_counter() - t:.2f} seconds.")

t = time.perf_counter()
running_mean_plot(ndf,bys=bys,out_root=out_root,plot_args={},name='normalized')
print(f"Plot 2b done in {time.perf_counter() - t:.2f} seconds.")



t = time.perf_counter()
mean_dur_boxplot(df,bys=bys,out_root=out_root,plot_args={}, name=None)
print(f"Plot 3 done in {time.perf_counter() - t:.2f} seconds.")

t = time.perf_counter()
mean_dur_boxplot(ndf,bys=bys,out_root=out_root,plot_args={}, name='normalized')
print(f"Plot 3b done in {time.perf_counter() - t:.2f} seconds.")



t = time.perf_counter()
n_max_boxplot(df,bys=bys,out_root=out_root,plot_args={}, name=None)
print(f"Plot 4 done in {time.perf_counter() - t:.2f} seconds.")

