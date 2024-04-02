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


root = '.'

def extract_params(root):
    params = {}
    for dir_name in os.listdir(root):
        dir_path = os.path.join(root, dir_name)
        if os.path.isdir(dir_path):
            file_path = os.path.join(dir_path, 'param_log.txt')
            if os.path.isfile(file_path):
                with open(file_path, 'r') as file:
                    content = file.read()
                    gamma = re.search(r'GAMMA: (\d+\.\d+)', content)
                    layer_sizes = re.search(r'LAYER_SIZES: (\[.*?\])', content)
                    if gamma and layer_sizes:
                        params[dir_name] = {
                            'GAMMA': float(gamma.group(1)),
                            'LAYER_SIZES': eval(layer_sizes.group(1))
                        }
    return params

def create_duration_dataset(root, params):
    df = pd.DataFrame()
    for dir_name, param in params.items():
        dir_path = os.path.join(root, dir_name)
        for file in os.listdir(dir_path):
            if file.endswith('.csv'):
                    data = pd.read_csv(os.path.join(root, dir_name, file))
                    if 'durations' in data.columns:
                        column_name = f"G:{param['GAMMA']}-L{param['LAYER_SIZES']}-{dir_name}"
                        df[column_name] = data['durations']
    return df


# Mean
out = "RESULTS FOR LAYER SIZE AND GAMMA COMBINAISON \n------------------------\n"
params = extract_params(root)
df = create_duration_dataset(root, params)
averages = df.mean()
sorted_averages = averages.sort_values(ascending=False)
out += "\nTop 10 sorted averages:"
for i, (key, value) in enumerate(sorted_averages.items()):
    if i == 10:
        break
    out += f"\n\t {i}- {value:<7.2f}  {key}"


# Mean by gamma
out += "\n------------------------\n"
gamma_values = df.columns.str.split('-').str[0].str.split(':').str[1]
grouped = df.groupby(gamma_values, axis=1)
mean_by_gamma = grouped.mean().mean()
sorted_mean_by_gamma = mean_by_gamma.sort_values(ascending=False)
out += "\nSorted mean by gamma:"
for i, (key, value) in enumerate(sorted_mean_by_gamma.items()):
    out += f"\n\t {i+1}- {value:<7.2f}  {key}"


# Mean by layer sizes
out += "\n------------------------\n"
layer_size_values = df.columns.str.split('-').str[1].str[1:]
grouped = df.groupby(layer_size_values, axis=1)
mean_by_layer_size = grouped.mean().mean()
sorted_mean_by_layer_size = mean_by_layer_size.sort_values(ascending=False)
out += "\nSorted mean by layer sizes:"
for i, (key, value) in enumerate(sorted_mean_by_layer_size.items()):
    out += f"\n\t {i+1}- {value:<7.2f}  {key}"

# Mean by gamma and layer sizes
out += "\n------------------------\n"
df.columns = pd.MultiIndex.from_arrays([gamma_values, layer_size_values], names=['gamma', 'layer_size'])
grouped = df.groupby(level=['gamma', 'layer_size'], axis=1)
mean_by_gamma_layer_size = grouped.mean().mean()
sorted_mean_by_gamma_layer_size = mean_by_gamma_layer_size.sort_values(ascending=False)
out += "\nSorted mean by gamma and layer sizes:"
for i, (key, value) in enumerate(sorted_mean_by_gamma_layer_size.items()):
    if i == 10:
        break
    out += f"\n\t {i+1}- {value:<7.2f}  {key}"

with open(f'{root}/results_mean.txt', 'w') as file:
    file.write(out)

print(f"Results saved in {root}/results_mean.txt")
## PLOTTING ##

# Top Gamma
df = create_duration_dataset(root, params)
column_means = df.mean()

top_columns = column_means.nlargest(10).index
unique_gammas = top_columns.str.split('-').str[0].unique()
colors = mcolors.TABLEAU_COLORS
color_dict = {gamma: color for gamma, color in zip(unique_gammas, colors)}

plt.figure(figsize=(20, 10))
used_labels = set()
for col in top_columns:
    gamma, layer_size, _ , id = col.split('-')
    id = id.split('_')[1]

    avg = np.cumsum(df[col]) / np.arange(1, len(df[col]) + 1)

    if gamma not in used_labels:
        plt.plot(avg, label=gamma, color=color_dict[gamma])
        used_labels.add(gamma)
    else:
        plt.plot(avg, color=color_dict[gamma])

plt.title("Top 10 sorted averages bu gamma values")
plt.legend()
#plt.show()

# Top Layersize
df = create_duration_dataset(root, params)
column_means = df.mean()

top_columns = column_means.nlargest(10).index
unique_layer_sizes = top_columns.str.split('-').str[1].unique()
colors = mcolors.TABLEAU_COLORS
color_dict = {layer_size: color for layer_size, color in zip(unique_layer_sizes, colors)}

plt.figure(figsize=(20, 10))
used_labels = set()
for col in top_columns:
    gamma, layer_size, _ , id = col.split('-')
    id = id.split('_')[1]

    avg = np.cumsum(df[col]) / np.arange(1, len(df[col]) + 1)

    if layer_size not in used_labels:
        plt.plot(avg, label=layer_size, color=color_dict[layer_size])
        used_labels.add(layer_size)
    else:
        plt.plot(avg, color=color_dict[layer_size])

plt.title("Top 10 sorted averages bu layer sizes")
plt.legend()
#plt.show()


## CUMSUM AVG
# Regrouper les données par gamma
df = create_duration_dataset(root, params)
for col in df.columns:
    df[col] = np.cumsum(df[col]) / np.arange(1, len(df[col]) + 1)
grouped = df.groupby(df.columns.str.split('-').str[0], axis=1)
# Calculer la moyenne et l'intervalle de confiance pour chaque groupe
mean = grouped.mean()
std = grouped.std()
ci = grouped.apply(lambda group: stats.t.interval(0.95, len(group)-1, loc=group.mean(), scale=stats.sem(group)))

sorted_mean = mean.mean(axis=0).sort_values(ascending=False)
top_10_params = sorted_mean.head(len(sorted_mean)).index
plt.figure(figsize=(20, 10))
for i,param in enumerate(top_10_params):
    avg = np.cumsum(mean[param]) / np.arange(1, len(mean[param]) + 1)
    ci = 1.96 * std[param] / np.sqrt(len(std))
    plt.plot(avg, label=param, color=list(colors.values())[i])
    plt.fill_between(range(len(avg)), avg + ci, avg - ci, alpha=0.4, color=list(colors.values())[i])

plt.title("Average duration by gamma values")
plt.xlabel('Games played')
plt.ylabel('Duration')
plt.legend()
plt.savefig(f'{root}/avg_duration_by_gamma.jpg')
print(f"Results saved in {root}/avg_duration_by_gamma.jpg")
#plt.show()

## Avg layer size
df = create_duration_dataset(root, params)
for col in df.columns:
    df[col] = np.cumsum(df[col]) / np.arange(1, len(df[col]) + 1)
grouped = df.groupby(df.columns.str.split('-').str[1], axis=1)
# Calculer la moyenne et l'intervalle de confiance pour chaque groupe
mean = grouped.mean()
std = grouped.std()
ci = grouped.apply(lambda group: stats.t.interval(0.95, len(group)-1, loc=group.mean(), scale=stats.sem(group)))

sorted_mean = mean.mean(axis=0).sort_values(ascending=False)
top_10_params = sorted_mean.head(len(sorted_mean)).index
plt.figure(figsize=(20, 10))
for i,param in enumerate(top_10_params):
    avg = np.cumsum(mean[param]) / np.arange(1, len(mean[param]) + 1)
    ci = 1.96 * std[param] / np.sqrt(len(std))
    plt.plot(avg, label=param, color=list(colors.values())[i])
    plt.fill_between(range(len(avg)), avg + ci, avg - ci, alpha=0.4, color=list(colors.values())[i])

plt.title("Average duration by layer sizes")
plt.xlabel('Games played')
plt.ylabel('Duration')
plt.legend()
plt.savefig(f'{root}/avg_duration_by_layer_size.jpg')
print(f"Results saved in {root}/avg_duration_by_layer_size.jpg")
#plt.show()


## Avg layer size and gamma
df = create_duration_dataset(root, params)
for col in df.columns:
    df[col] = np.cumsum(df[col]) / np.arange(1, len(df[col]) + 1)

gamma_values = df.columns.str.split('-').str[0]
layer_size_values = df.columns.str.split('-').str[1]
df.columns = pd.MultiIndex.from_arrays([gamma_values, layer_size_values], names=['gamma', 'layer_size'])
grouped = df.groupby(level=['gamma', 'layer_size'], axis=1)
mean = grouped.mean()
std = grouped.std()
ci = grouped.apply(lambda group: stats.t.interval(0.95, len(group)-1, loc=group.mean(), scale=stats.sem(group)))
sorted_mean = mean.mean(axis=0).sort_values(ascending=False)
# Sélectionner les 10 premières lignes
top_10_params = sorted_mean.head(10).index
# Tracer la moyenne avec l'intervalle de confiance pour chaque gamma
plt.figure(figsize=(20, 10))
for i,param in enumerate(top_10_params):
    avg = np.cumsum(mean[param]) / np.arange(1, len(mean[param]) + 1)
    ci = 1.96 * std[param] / np.sqrt(len(std))
    plt.plot(avg, label=param)
    plt.fill_between(range(len(avg)), avg + ci, avg - ci, alpha=0.4)

plt.title("Average duration by gamma and layer sizes")
plt.xlabel('Games played')
plt.ylabel('Duration')
plt.legend()
plt.savefig(f'{root}/avg_duration_by_gamma_layer_size.jpg')
print(f"Results saved in {root}/avg_duration_by_gamma_layer_size.jpg")
#plt.show()
