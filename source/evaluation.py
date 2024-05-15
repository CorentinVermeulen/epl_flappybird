import time
from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt
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
import matplotlib.colors as mcolors
import scipy.stats as stats
import plotly.express as px
from IPython.display import Image
import seaborn as sns

import pandas as pd
import re
import itertools
sns.set_context("paper")
sns.set_style("whitegrid")
large_size = (11, 5.5)
small_size = (7, 3.5)
square_size = (5.5, 5.5)

def make_out_root(root):
    if root[-1] != '/':
        root += '/'
    out_root = f"{root}all_results"
    if not os.path.exists(out_root):
        os.mkdir(out_root)
    return out_root

def rename_results(out_root, prefix):
    for file in os.listdir(out_root):
        if file.endswith('.pdf'):
            if prefix not in file:
                new_name = f'{prefix}_{file}'
                os.rename(f'{out_root}/{file}', f'{out_root}/{new_name}')

def get_stacked_df(root, params_under_study, out_root):
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
                try:
                    data = pd.read_csv(os.path.join(root, dir_name, file))
                    if len(data) < 1000:
                        print(f"File {file} in {dir_name} has less than 1000 rows. ({len(data)})")

                except:
                    print(f"Error reading file {file} in {dir_name}.")
                    continue
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
    # Save to csv
    out_df.to_csv(f'{out_root}/stacked_df.csv', index=False)
    return out_df

def select_best_id(df, var, n=5):
    il = len(df)
    # For every var, I will only keep the best n ids
    best_ids = df[['id', var, 'duration']].groupby(['id', var]).mean().reset_index()
    best_ids = best_ids.sort_values(by='duration', ascending=False).groupby(var).head(n)
    best_df = df[df['id'].isin(best_ids['id'])]
    ol = len(best_df)
    print(f"Reduced from {il} to {ol} rows.")
    return best_df

def normalize_stacked_df(df, bys):
    ndf = df.copy()
    max_duration = ndf.groupby(bys)['duration'].transform('max')
    max_cumsum = ndf.groupby(bys)['cumsum'].transform('max')
    max_running_mean = ndf.groupby(bys)['running_mean'].transform('max')

    ndf['duration'] /= max_duration
    ndf['cumsum'] /= max_cumsum
    ndf['running_mean'] /= max_running_mean

    return ndf

def remove_bad_id(idf):
    df = idf.copy()
    df = df.groupby('id').filter(lambda x: x['duration'].mean() > 150)
    return df

## DESCRIPTIVE PLOTS ##
# 1 AVG PLOT and RM PLOT
def avg_plot(df, out_root, bys, plot_args={}, name=None):
    order = np.sort(df[bys[0]].unique())
    plt.figure(figsize=large_size)
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
    plt.savefig(f'{out_root}/{out_name}.pdf', format='pdf', bbox_inches='tight')
    ##plt.show()
    plt.figure(figsize=small_size)
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
    plt.savefig(f'{out_root}/{out_name}_small.pdf', format='pdf', bbox_inches='tight')


def running_mean_plot(idf, bys, out_root, plot_args={}, name=None):
    df = idf.copy()
    df = df.dropna()
    order = np.sort(df[bys[0]].unique())

    plt.figure(figsize=large_size)
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
    plt.savefig(f'{out_root}/{out_name}.pdf', format='pdf', bbox_inches='tight')
    ##plt.show()
    plt.figure(figsize=small_size)
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
    plt.savefig(f'{out_root}/{out_name}_small.pdf', format='pdf', bbox_inches='tight')


# 2 MEAN DURATION BOXPLOT
def mean_dur_boxplot(idf, bys, out_root, plot_args={}, name=None):
    df = idf.copy()
    df = df[['id'] + bys + ['duration']].groupby(['id'] + bys).mean()
    order = np.sort(df.index.get_level_values(bys[0]).unique())

    plt.figure(figsize=square_size)
    sns.boxplot(data=df,
                x=bys[0],
                order=order,
                y='duration',
                fill=False,
                flierprops={"marker": "x"},
                medianprops={"color": "r", "linewidth": 2},
                showcaps=False,
                showmeans=True,
                **plot_args
                )
    plt.xlabel(bys[0])
    plt.ylabel('Duration')
    plt.ylim(0, df['duration'].max() * 1.1)
    plt.suptitle("")
    plt.title('')
    out_name = 'mean_dur_boxplot'
    if name:
        out_name = f'{out_name}_{name}'
    plt.savefig(f'{out_root}/{out_name}.pdf', format='pdf', bbox_inches='tight')
    ##plt.show()


# 3 N GAMES MAX DURATION BOXPLOT
def n_max_boxplot(idf, bys, out_root, plot_args={}, name=None, normalized_max=False):
    df = idf.copy()
    if normalized_max:
        df['maxs'] = df.groupby(bys)['duration'].transform('max')
    else:
        df['maxs'] = df['duration'].max()
    df['greater'] = df['duration'] >= df['maxs'] * 0.999
    n_maxs = df.groupby(['id'] + bys)['greater'].sum()
    n_maxs = n_maxs.reset_index()
    order = np.sort(n_maxs[bys[0]].unique())
    plt.figure(figsize=square_size)
    sns.boxplot(data=n_maxs,
                x=bys[0],
                order=order,
                y='greater',
                fill=False,
                flierprops={"marker": "x"},
                medianprops={"color": "r", "linewidth": 2},
                showcaps=False,
                showmeans = True,
                **plot_args
                )
    plt.xlabel(bys[0])
    ylabel = "Number of games with maximum duration"
    if normalized_max:
        ylabel = "Number of games with normalized maximum duration"
    plt.ylabel(ylabel)
    plt.ylim(0, n_maxs['greater'].max() * 1.1 )
    plt.suptitle("")
    plt.title('')
    out_name = 'n_max_boxplot'
    if name:
        out_name = f'{out_name}_{name}'
    plt.savefig(f'{out_root}/{out_name}.pdf', format='pdf', bbox_inches='tight')
    ##plt.show()


## DSP PLOTS ##

def dsp_plot(idf, bys, out_root, plot_args={}, name=None, normalized_max=False):
    df = idf.copy()
    # DSP_X = Number of games reaching X% of the max duration
    if normalized_max:
        df['maxs'] = df.groupby(bys)['duration'].transform('max')
    else:
        df['maxs'] = df['duration'].max()

    plot_df = pd.DataFrame(columns=['id', 'X', 'DSP'] + bys)

    for X in np.arange(0.1, 1.1, 0.1):
        df['DSP'] = df['duration'] >= df['maxs'] * X
        n_over = df.groupby(['id'] + bys)['DSP'].sum() / 1000
        n_over = n_over.reset_index()
        n_over['X'] = X
        plot_df = pd.concat([plot_df, n_over])
    order = np.sort(plot_df[bys[0]].unique())

    plt.figure(figsize=large_size)
    sns.lineplot(data=plot_df,
                 x='X',
                 y='DSP',
                 hue=bys[0],
                hue_order=order,
                 **plot_args)
    xlabel = "Proportion of maximum duration"
    if normalized_max:
        xlabel = "Proportion of normalized maximum duration"
    plt.xlabel(xlabel)
    plt.ylabel('DSP')
    plt.ylim(0, 1)
    plt.suptitle("")
    plt.title('')
    out_name = 'dsp_plot'
    if name:
        out_name = f'{out_name}_{name}'
    plt.savefig(f'{out_root}/{out_name}.pdf', format='pdf', bbox_inches='tight')
    #plt.show()

    plt.figure(figsize=small_size)
    sns.lineplot(data=plot_df,
                 x='X',
                 y='DSP',
                 hue=bys[0],
                 hue_order=order,
                 **plot_args)
    xlabel = "Proportion of maximum duration"
    if normalized_max:
        xlabel = "Proportion of normalized maximum duration"
    plt.xlabel(xlabel)
    plt.ylabel('DSP')
    plt.ylim(0, 1)
    plt.suptitle("")
    plt.title('')
    out_name = 'dsp_plot'
    if name:
        out_name = f'{out_name}_{name}'
    plt.savefig(f'{out_root}/{out_name}_small.pdf', format='pdf', bbox_inches='tight')


def lvi_prepross(idf, bys, out_root, plot_args={}, name=None, normalized_max=False, var='cumsum'):
    df = idf.copy()
    if var=='running_mean':
        df = df.dropna()

    if normalized_max:
        df['maxs'] = df.groupby(bys)[var].transform('max')
    else:
        df['maxs'] = df[var].max()

    #plot_df = pd.DataFrame(columns=['id', 'X', 'nLVI'] + bys)
    for X in np.arange(0.1, 1.1, 0.1):
        df['nLVI'] = df[var] >= df['maxs'] * X
        n_over = df.groupby(['id'] + bys)['nLVI'].sum() / 1000
        n_over = n_over.reset_index()
        n_over['X'] = X
        plot_df = pd.concat([plot_df, n_over])
    plot_df = plot_df[plot_df["nLVI"] != 0]
    order = np.sort(plot_df[bys[0]].unique())
    plt.figure(figsize=large_size)
    sns.lineplot(data=plot_df,
                 x='X',
                 y='nLVI',
                 hue=bys[0],
                 hue_order=order,
                 **plot_args)
    xlabel = "Proportion of average curve above X thresold of maximum duration"
    if normalized_max:
        xlabel = "Proportion of average curve above X thresold of normalized maximum durationn"
    plt.xlabel(xlabel)
    plt.ylabel('Proportion of games above threshold')
    plt.ylim(0, 1)
    plt.suptitle("")
    plt.title('')
    out_name = 'lvi_pre_plot_' + var
    if name:
        out_name = f'{out_name}_{name}'
    plt.savefig(f'{out_root}/{out_name}.pdf', format='pdf', bbox_inches='tight')
    #plt.show()

def lvi(idf, bys, out_root, plot_args={}, name=None, normalized_max=False, var='cumsum'):
    # LVI_X = NUMBER OF GAMES NEEDED TO REACH X% OF THE MAX DURATION
    df = idf.copy()
    if var=='running_mean':
        df = df.dropna()

    if normalized_max:
        df['maxs'] = df.groupby(bys)[var].transform('max')
    else:
        df['maxs'] = df[var].max()

    plot_df = pd.DataFrame(columns=['id', 'X', 'LVI'] + bys)
    for X in np.arange(0.1, 1.1, 0.1):
        df['LVI'] = df[var] >= df['maxs'] * X
        over = df.groupby(['id'] + bys)['LVI'].idxmax()
        over = over.reset_index()
        over['X'] = X
        plot_df = pd.concat([plot_df, over])

    plot_df = plot_df[plot_df["LVI"] != 1000]
    plot_df.replace(0, 1000, inplace=True)

    order = np.sort(plot_df[bys[0]].unique())
    plt.figure(figsize=large_size)
    sns.lineplot(data=plot_df,
                 x='X',
                 y='LVI',
                 hue=bys[0],
                 hue_order=order,
                 **plot_args)
    xlabel = "Proportion of maximum duration"
    if normalized_max:
        xlabel = "Proportion of normalized maximum duration"
    plt.xlabel(xlabel)
    plt.ylabel("Number of games")
    plt.ylim(0, 1000)
    plt.suptitle("")
    plt.title('')
    out_name = 'lvi_plot_' + var
    if name:
        out_name = f'{out_name}_{name}'
    plt.savefig(f'{out_root}/{out_name}.pdf', format='pdf', bbox_inches='tight')
    #plt.show()

    plt.figure(figsize=small_size)
    sns.lineplot(data=plot_df,
                 x='X',
                 y='LVI',
                 hue=bys[0],
                 hue_order=order,
                 **plot_args)
    xlabel = "Proportion of maximum duration"
    if normalized_max:
        xlabel = "Proportion of normalized maximum duration"
    plt.xlabel(xlabel)
    plt.ylabel("Number of games")
    plt.ylim(0, 1000)
    plt.suptitle("")
    plt.title('')
    out_name = 'lvi_plot_' + var
    if name:
        out_name = f'{out_name}_{name}'
    plt.savefig(f'{out_root}/{out_name}_small.pdf', format='pdf', bbox_inches='tight')


# ============================================================================= #
if __name__ == '__main__':
    iexps = [3]
    for iexp  in iexps:
        root = f'../../exps/exp_{iexp}_f/'
        prefix = f'EXP{iexp}'
        out_root = make_out_root(root)

        params_list = [
            ["Random_pipes"],
            ["Random_pipes"],
            ["Jump_Force_k"],
            ["Gravity_k"]
                ]

        params_under_study = params_list[iexp]

        bys = params_under_study

        t = time.perf_counter()
        df = get_stacked_df(root, params_under_study, out_root)
        df = select_best_id(df, bys[0], n=5)
        #df = df.query(" LR == '1e-05' and `Obs gravity`== 'False' and `Obs jumpforce`== 'False' ")

        #ndf = normalize_stacked_df(df, bys=[bys[0]])
        #dfc = remove_bad_id(df)
        dfc = df.copy()
        print(f"Data loaded in {time.perf_counter() - t:.2f} seconds.")


        t = time.perf_counter()
        avg_plot(dfc, bys=bys, out_root=out_root, plot_args={}, name=None )
        print(f"Plot 1 done in {time.perf_counter() - t:.2f} seconds.")

        t = time.perf_counter()
        running_mean_plot(dfc,bys=bys,out_root=out_root,plot_args={},name=None)
        print(f"Plot 2 done in {time.perf_counter() - t:.2f} seconds.")

        t = time.perf_counter()
        mean_dur_boxplot(dfc,bys=bys,out_root=out_root,plot_args={}, name=None)
        print(f"Plot 3 done in {time.perf_counter() - t:.2f} seconds.")

        t = time.perf_counter()
        n_max_boxplot(dfc,bys=bys,out_root=out_root,plot_args={}, name=None)
        print(f"Plot 4 done in {time.perf_counter() - t:.2f} seconds.")

        t = time.perf_counter()
        dsp_plot(dfc,bys=bys,out_root=out_root,plot_args={}, name=None)
        print(f"Plot 5 done in {time.perf_counter() - t:.2f} seconds.")

        t = time.perf_counter()
        lvi(dfc,bys=bys,out_root=out_root,plot_args={}, name=None, var='cumsum')
        #lvi(dfc,bys=bys,out_root=out_root,plot_args={}, name=None, var='running_mean')
        print(f"Plot 7 done in {time.perf_counter() - t:.2f} seconds.")

        # # t = time.perf_counter()
        # # lvi_prepross(dfc,bys=bys,out_root=out_root,plot_args={}, name=None, var='cumsum')
        # # lvi_prepross(dfc,bys=bys,out_root=out_root,plot_args={}, name=None, var='running_mean')
        # #print(f"Plot 6 done in {time.perf_counter() - t:.2f} seconds.")

        rename_results(out_root, prefix)