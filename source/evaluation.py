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

def get_stacked_df(root, params_under_study, out_root, fill_to=None):
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

    out_df = pd.DataFrame(columns=["id", "t"] + params_under_study + ["loss", "duration", "linetype"])
    for dir_name, param in params.items():
        dir_path = os.path.join(root, dir_name)
        for file in os.listdir(dir_path):
            if file.endswith('.csv'):
                df = pd.DataFrame(columns=["id", "t"] + params_under_study + ["loss", "duration", "linetype"])
                try:
                    data = pd.read_csv(os.path.join(root, dir_name, file))
                    if len(data) < fill_to:
                        print(f"File {file} in {dir_name} has less than {fill_to} rows. ({len(data)})")

                except:
                    print(f"Error reading file {file} in {dir_name}.")
                    continue
                if 'durations' in data.columns:
                    if fill_to and len(data) < fill_to:
                        avg = data.loc[500:, 'durations'].mean()
                        full_dur = pd.concat([data['durations'], pd.Series([avg] * (fill_to - len(data)))]).reset_index(drop=True)
                        df['duration'] = full_dur
                        ts = np.arange(0, len(full_dur))
                        df['t']= ts
                        lty = ['data'] * len(data) + ['extrapolation'] * (fill_to - len(data))
                        df['linetype'] = lty

                    else:
                        df['duration'] = data['durations']
                        df['t'] = data.index
                        df['linetype'] = 'data'
                    df['cumsum'] = np.cumsum(df['duration']) / np.arange(1, len(df['duration']) + 1)
                    df['running_mean'] = df['duration'].rolling(window=50).mean()
                    #df['loss'] = data['loss']
                    df['id'] = dir_name

                    for p in params_under_study:
                        df[p] = eval(param[p])
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
    # removes lines where running_mean is NaN
    df = df.dropna(subset=['running_mean'])
    order = np.sort(df[bys[0]].unique())
    #df['n_actions'] = df['n_actions'].astype("category")
    pal = sns.color_palette("hls", len(order))
    plt.figure(figsize=large_size)
    sns.lineplot(data=df.query('linetype!="data" and n_actions == 2 '),
                 x='t',
                 y='running_mean',
                 linestyle='--',
                 color=pal[0],
                 **plot_args)
    sns.lineplot(data=df.query('linetype=="data"'),
                 x='t',
                 y='running_mean',
                 hue=bys[0],
                 hue_order=order,
                 palette=pal,
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
    sns.lineplot(data=df.query('linetype!="data" and n_actions == 2 '),
                 x='t',
                 y='running_mean',
                 linestyle='--',
                 color=pal[0],
                 **plot_args)
    sns.lineplot(data=df.query('linetype=="data"'),
                 x='t',
                 y='running_mean',
                 hue=bys[0],
                 hue_order=order,
                 palette=pal,
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
    """
    Compute proportion of games reaching X% of the max duration.
    The proportion is only computed on games after the first time the threshold is reached.
    """

    df = idf.copy()
    # DSP_X = Number of games reaching X% of the max duration
    if normalized_max:
        df['maxs'] = df.groupby(bys)['duration'].transform('max')
    else:
        df['maxs'] = df['duration'].max()

    plot_df = pd.DataFrame(columns=['id', 'X', 'DSP'] + bys)

    for X in np.arange(0.1, 1.05, 0.05):
        df['DSP'] = df['duration'] >= df['maxs'] * X
        # get index of first time the threshold is reached
        over = df.groupby(['id'] + bys)['DSP'].idxmax()
        over = over.reset_index()
        # remove from df lines where t is < than the first time the threshold is reached
        for i,row in over.iterrows():
            df = df[~((df['id'] == row['id']) & (df['t'] < row['DSP']))]

        n_over = df.groupby(['id'] + bys)['DSP'].sum() / df.groupby(['id'] + bys)['DSP'].count()
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

def lvi(idf, bys, out_root, plot_args={}, name=None, var='cumsum', normalized_max=False):
    # LVI_X = NUMBER OF GAMES NEEDED TO REACH X% OF THE MAX DURATION
    df = idf.copy()
    if var=='running_mean':
        df = df.dropna()

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

def lvi2(idf, bys, out_root, plot_args={}, var='running_mean', name=None):
    df = idf.copy()

    # Normalize cumsum on avg value for group_by
    mean_by = df.groupby(bys)[var].mean()
    df['mean_by_inf'] = df[bys + [var]].apply(lambda x: mean_by[x[0]], axis=1)
    df['cumsum_normalized_by_mean'] = df[var] / df['mean_by_inf']

    max_by = df.groupby(bys)[var].max()
    df['max_by_inf'] = df[bys + [var]].apply(lambda x: max_by[x[0]], axis=1)
    df['cumsum_normalized_by_max'] = df[var] / df['max_by_inf']

    # New df
    plot_df = pd.DataFrame(columns=['id', 'X', 'LVI'] + bys)
    for X in np.arange(0.1, 0.95, 0.05):
        df['LVI_mean'] = df['cumsum_normalized_by_mean'] >= X
        df['LVI_max'] = df['cumsum_normalized_by_max'] >= X

        over = df.groupby(['id'] + bys)[['LVI_mean', 'LVI_max']].idxmax()
        over = over.reset_index()
        over['X'] = X
        if X>0.2:
            over = over[over['LVI_mean'] != 0]
            over = over[over['LVI_max'] != 0]

        plot_df = pd.concat([plot_df, over])

    xlabel_mean = "Proportion of normalized maximum duration"
    xlabel_max = "Proportion of maximum duration"

    order = np.sort(plot_df[bys[0]].unique())
    # Normalized by mean
    plt.figure(figsize=large_size)
    sns.lineplot(data=plot_df,
                 x='X',
                 y='LVI_mean',
                 hue=bys[0],
                 hue_order=order,
                 **plot_args)


    plt.xlabel(xlabel_mean)
    plt.ylabel("Number of games")
    plt.ylim(0, 1000)
    plt.suptitle("")
    plt.title('')
    out_name = 'lvi_plot_' + var
    if name:
        out_name = f'{out_name}_{name}'
    out_name = f'{out_name}_normalized'
    plt.savefig(f'{out_root}/{out_name}.pdf', format='pdf', bbox_inches='tight')
    # plt.show()

    # Normalized by mean
    plt.figure(figsize=large_size)
    sns.lineplot(data=plot_df,
                 x='X',
                 y='LVI_max',
                 hue=bys[0],
                 hue_order=order,
                 **plot_args)

    plt.xlabel(xlabel_max)
    plt.ylabel("Number of games")
    plt.ylim(0, 1000)
    plt.suptitle("")
    plt.title('')
    out_name = 'lvi_plot_' + var
    if name:
        out_name = f'{out_name}_{name}'
    plt.savefig(f'{out_root}/{out_name}.pdf', format='pdf', bbox_inches='tight')


def _make_dsp_lvi_df(idf, bys):
    df = idf.copy()
    var = 'running_mean'
    # LVI: Normalize cumsum on avg value for group_by
    mean_by = df.groupby(bys)[var].mean()
    df['mean_by_inf'] = df[bys + [var]].apply(lambda x: mean_by[x[0]], axis=1)
    df['cumsum_normalized_by_mean'] = df[var] / df['mean_by_inf']

    # LVI: Normalize cumsum on max value for group_by
    max_by = df.groupby(bys)[var].max()
    df['max_by_inf'] = df[bys + [var]].apply(lambda x: max_by[x[0]], axis=1)
    df['cumsum_normalized_by_max'] = df[var] / df['max_by_inf']

    # DSP:
    df['maxs'] = df['duration'].max()

    # df with all res for plotting
    plot_df = pd.DataFrame(columns=['id', 'X', 'LVI_mean', 'LVI_max', 'DSP'] + bys)
    for X in np.arange(0.1, 0.95, 0.05):
        df['LVI_mean'] = df['cumsum_normalized_by_mean'] >= X
        df['LVI_max'] = df['cumsum_normalized_by_max'] >= X

        over_lvi = df.groupby(['id'] + bys)[['LVI_mean', 'LVI_max']].idxmax()
        # if X > 0.2:
        #     over_lvi = over_lvi[over_lvi['LVI_mean'] != 0]
        #     over_lvi = over_lvi[over_lvi['LVI_max'] != 0]

        # get val where running mean surpass X threshold
        df_copy = df.copy()
        df_copy['count_for_dsp'] = df_copy['running_mean'] > X * df_copy['maxs']
        df_copy['DSP'] = df['duration'] >= df['maxs'] * X

        over_dsp = df_copy.groupby(['id'] + bys)['count_for_dsp'].idxmax().reset_index()
        for i, row in over_dsp.iterrows():
            df_copy = df_copy[~((df_copy['id'] == row['id']) & (df_copy['t'] < row['count_for_dsp']))]

        n_over = df_copy.groupby(['id'] + bys)['DSP'].sum() / df_copy.groupby(['id'] + bys)['DSP'].count()

        over_merged = over_lvi.merge(n_over, on=['id']+ bys, how='left')
        over_merged['X'] = X
        over_merged = over_merged.reset_index()
        plot_df = pd.concat([plot_df, over_merged])
    return plot_df

def lvi22(plot_df,out_root,bys, plot_args={}, name=None):
    xlabel_mean = "Proportion of average duration"
    xlabel_max = "Proportion of maximum duration"
    var = ""
    order = np.sort(plot_df[bys[0]].unique())
    # Normalized by mean
    plt.figure(figsize=small_size)
    sns.lineplot(data=plot_df,
                 x='X',
                 y='LVI_mean',
                 hue=bys[0],
                 hue_order=order,
                 **plot_args)

    plt.xlabel(xlabel_mean)
    plt.ylabel("Number of games")
    plt.ylim(0)
    plt.suptitle("")
    plt.title('')
    out_name = 'lvi_plot'
    if name:
        out_name = f'{out_name}_{name}'
    out_name = f'{out_name}_normalized'
    plt.savefig(f'{out_root}/{out_name}.pdf', format='pdf', bbox_inches='tight')
    # plt.show()

    # Normalized by mean
    plt.figure(figsize=small_size)
    sns.lineplot(data=plot_df,
                 x='X',
                 y='LVI_max',
                 hue=bys[0],
                 hue_order=order,
                 **plot_args)

    plt.xlabel(xlabel_max)
    plt.ylabel("Number of games")
    plt.ylim(0)
    plt.suptitle("")
    plt.title('')
    out_name = 'lvi_plot_' + var
    if name:
        out_name = f'{out_name}_{name}'
    plt.savefig(f'{out_root}/{out_name}.pdf', format='pdf', bbox_inches='tight')

def dsp22(plot_df,out_root,bys, plot_args={}, name=None):
    order = np.sort(plot_df[bys[0]].unique())
    plt.figure(figsize=large_size)
    sns.lineplot(data=plot_df,
                 x='X',
                 y='DSP',
                 hue=bys[0],
                 hue_order=order,
                 **plot_args)
    xlabel = "Proportion of maximum duration"

    plt.xlabel(xlabel)
    plt.ylabel('DSP')
    plt.ylim(0, 1)
    plt.suptitle("")
    plt.title('')
    out_name = 'dsp_plot'
    if name:
        out_name = f'{out_name}_{name}'
    plt.savefig(f'{out_root}/{out_name}.pdf', format='pdf', bbox_inches='tight')
    # plt.show()

    plt.figure(figsize=small_size)
    sns.lineplot(data=plot_df,
                 x='X',
                 y='DSP',
                 hue=bys[0],
                 hue_order=order,
                 **plot_args)
    xlabel = "Proportion of maximum duration"

    plt.xlabel(xlabel)
    plt.ylabel('DSP')
    plt.ylim(0, 1)
    plt.suptitle("")
    plt.title('')
    out_name = 'dsp_plot'
    if name:
        out_name = f'{out_name}_{name}'
    plt.savefig(f'{out_root}/{out_name}_small.pdf', format='pdf', bbox_inches='tight')



# ============================================================================= #
if __name__ == '__main__':
    iexps = [4]
    for iexp  in iexps:
        root = f'../../exps/exp_{iexp}_test/'
        prefix = f'EXP{iexp}'
        out_root = make_out_root(root)

        params_list = [
            ["Random_pipes"],
            ["Random_pipes"],
            ["Jump_Force_k"],
            ["Gravity_k"],
            ["n_actions"],
                ]

        params_under_study = params_list[iexp]
        #params_under_study = ['n_actions']
        bys = params_under_study

        t = time.perf_counter()
        df = get_stacked_df(root, params_under_study, out_root, fill_to=2500)
        df = select_best_id(df, bys[0], n=10)
        name=""

        #df = df.query(" LR == '1e-05' and `Obs gravity`== 'False' and `Obs jumpforce`== 'False' ")

        #ndf = normalize_stacked_df(df, bys=[bys[0]])
        #dfc = remove_bad_id(df)
        dfc = df.copy()
        print(f"Data loaded in {time.perf_counter() - t:.2f} seconds.")


        # t = time.perf_counter()
        # avg_plot(dfc, bys=bys, out_root=out_root, plot_args={}, name=name )
        # print(f"Plot avg done in {time.perf_counter() - t:.2f} seconds.")

        t = time.perf_counter()
        running_mean_plot(dfc,bys=bys,out_root=out_root,plot_args={},name=name)
        print(f"Plot runnning_mean done in {time.perf_counter() - t:.2f} seconds.")

        # t = time.perf_counter()
        # mean_dur_boxplot(dfc,bys=bys,out_root=out_root,plot_args={}, name=name)
        # print(f"Plot mean_dur done in {time.perf_counter() - t:.2f} seconds.")
        #
        # t = time.perf_counter()
        # n_max_boxplot(dfc,bys=bys,out_root=out_root,plot_args={}, name=name)
        # print(f"Plot n_max done in {time.perf_counter() - t:.2f} seconds.")
        #
        # t = time.perf_counter()
        # plot_df = _make_dsp_lvi_df(dfc, bys)
        # print(f"plot_df done in {time.perf_counter() - t:.2f} seconds.")
        #
        # t = time.perf_counter()
        # lvi22(plot_df, out_root=out_root, bys=bys, plot_args={}, name=name)
        # print(f"Plot lvidone in {time.perf_counter() - t:.2f} seconds.")
        #
        # t = time.perf_counter()
        # dsp22(plot_df, out_root=out_root, bys=bys, plot_args={}, name=name)
        # print(f"Plot dsp done in {time.perf_counter() - t:.2f} seconds.")

        rename_results(out_root, prefix)