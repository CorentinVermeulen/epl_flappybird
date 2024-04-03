import time
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import scipy.stats as stats

from pathlib import Path
import pandas as pd
import re
import glob
import itertools


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

def create_duration_dataset(root, params):
    df = pd.DataFrame()
    for dir_name, param in params.items():
        dir_path = os.path.join(root, dir_name)
        for file in os.listdir(dir_path):
            if file.endswith('.csv'):
                    data = pd.read_csv(os.path.join(root, dir_name, file))
                    if 'durations' in data.columns:
                        df[dir_name] = data['durations']


    names = next(iter(params.values())).keys()
    values = []
    for col in df.columns:
        values.append([params[col][p] for p in names])

    df.columns = pd.MultiIndex.from_tuples(values, names=names)
    return df

def get_mean_by(df, bys, top_k=10):
    out = '\n--------------------------------\n'
    if len(bys) == 0:
        out += f"Top {top_k} mean durations:\n"
        mean = df.mean()
        sorted_mean = mean.sort_values(ascending=False)
        out += sorted_mean[:top_k].to_string()
    else:
        out += "Mean duration by " + ',  '.join(bys) + ":\n"
        grouped = df.groupby(bys, axis=1)
        mean_by = grouped.mean().mean()
        sorted_mean_by = mean_by.sort_values(ascending=False)
        out += sorted_mean_by.to_string()
    return out

def get_all_combinations(params):
    all_combinations = []
    for r in range(1, len(params) + 1):
        combinations_object = itertools.combinations(params, r)
        combinations = [list(combo) for combo in combinations_object]
        all_combinations.extend(combinations)

    return all_combinations

def convert_to_cumsum_df(df):
    cs = np.cumsum(df, axis=0)
    cs = cs.apply(lambda x: x / (x.index + 1))
    return cs

def plot_cumsum_by(df, bys, title, top_k = 10):
    colors = mcolors.TABLEAU_COLORS
    df = convert_to_cumsum_df(df)
    grouped = df.groupby(bys, axis=1)
    mean = grouped.mean()
    std = grouped.std()

    sorted_mean = mean.iloc[-1,:].sort_values(ascending=False)
    top_params = sorted_mean.head(min(len(sorted_mean), top_k)).index
    plt.figure(figsize=(20, 10))
    for i,param in enumerate(top_params):
        if len(bys)>1:
            label = ' - '.join([f"{p}:{b}" for p, b in zip(bys,param)])
        else:
            label = f"{bys[0]} - {param}"
        # avg = np.cumsum(mean[param]) / np.arange(1, len(mean[param]) + 1)
        avg = mean[param]
        ci = 1.96 * std[param] / np.sqrt(len(std))
        plt.plot(avg, label=label, color=list(colors.values())[i])
        plt.fill_between(range(len(avg)), avg + ci, avg - ci, alpha=0.4, color=list(colors.values())[i])

    plt.legend()
    plt.title(title)
    plt.xlabel('Games played')
    plt.ylabel('Duration')
    file_name = "plot_" + '_'.join(bys) + '.png'
    plt.savefig(f"{root}/{file_name}")
    print(f"Results saved in {root}/{file_name}")
    plt.show()

def main(root, params_under_study, res):

    params = extract_params(root, params_under_study)
    df = create_duration_dataset(root, params)
    combs = [[]] + get_all_combinations(params_under_study)

    for comb in combs:
        res += get_mean_by(df, comb, top_k=10)
        if len(comb) > 0:
            plot_cumsum_by(df, comb, 'Average duration by ' + ', '.join(comb))
        with(open(f'{root}/results.txt', 'w')) as file:
            file.write(res)
        print(f"Results saved in {root}/results.txt")

if __name__ == '__main__':
    root = '../experiments/hp_TAU'
    params_under_study = ['TAU',]
    res = "Gridsearch\n"
    res += "Experiment results about " + ', '.join(params_under_study).lower() + ":\n"

    main(root, params_under_study, res)


