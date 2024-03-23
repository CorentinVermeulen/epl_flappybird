import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def make_dfs(root_dirs):
    scores = pd.DataFrame()
    durations = pd.DataFrame()
    losses = pd.DataFrame()
    i = 0
    for root_dir in root_dirs:
        for n, fichier in enumerate(os.listdir(root_dir)):
            if fichier.endswith(".csv"):
                path = os.path.join(root_dir, fichier)
                df = pd.read_csv(path)
                df = df.rename(columns={"scores": f"score_{i}", "durations": f"duration_{i}", "loss": f"loss_{i}"})
                scores = pd.concat([scores, df[f'score_{i}']], axis=1)
                durations = pd.concat([durations, df[f'duration_{i}']], axis=1)
                losses = pd.concat([losses, df[f'loss_{i}']], axis=1)
                i += 1

    print(len(scores.columns))

    return scores, durations, losses

def get_avg_losses(losses, durations):
    avg_losses = losses.copy()
    for i, col in enumerate(losses.columns):
        avg_losses[col] = avg_losses[col] / durations.iloc[:,i]
    return avg_losses

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    out = (cumsum[N:] - cumsum[:-N]) / N
    prefix = np.repeat(np.nan, N - 1)
    return np.concatenate((prefix, out))

def interval_plot(df):
    mean = df.mean(axis=1)
    std = df.std(axis=1)
    ic = 1.96 * std / np.sqrt(len(df.columns))
    lower_ic = (mean - ic).clip(lower=df.min().min() - 0.1)
    upper_ic = (mean + ic).clip(upper=df.max().max() + 0.1)

    plt.plot(mean.index, running_mean(mean, 50), color='blue')
    plt.fill_between(mean.index, lower_ic, upper_ic, color='blue', alpha=0.5)
    plt.show()

def all_interval_plot(dic, title=None, ylabel= None):
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    if len(dic) > len(colors):
        raise ValueError("Too many dataframes to plot")
    for i, (label, df) in enumerate(dic.items()):
        mean = df.mean(axis=1)
        plt.plot(mean.index, running_mean(mean, 50), color=colors[i], label=label)
        for col in df.columns:
            plt.plot(range(len(df[col])), running_mean(df[col], 50), color=colors[i], alpha=0.1)

    if title:
        plt.title(title)
    if ylabel:
        plt.ylabel(ylabel)
    plt.legend()
    plt.xlabel("Game Played")
    plt.show()

def all_interval_plot_runnning_mean(dic, title=None, ylabel= None):
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    if len(dic) > len(colors):
        raise ValueError("Too many dataframes to plot")
    for i, (label, df) in enumerate(dic.items()):
        max = df.max().max()
        min = df.min().min()
        mean = running_mean(df.mean(axis=1), 50)
        std = running_mean(df.std(axis=1), 50)
        ic = 1.96 * std / np.sqrt(len(df.columns))
        lower_ic1 = (mean - ic).clip(min - 0.1, max+0.1)
        upper_ic1 = (mean + ic).clip(min - 0.1, max+0.1)
        plt.plot(range(len(mean)), mean, color=colors[i], label=label)
        plt.fill_between(range(len(mean)), lower_ic1, upper_ic1, color=colors[i], alpha=0.3)

    if title:
        plt.title(title + ' (95% IC)')
    if ylabel:
        plt.ylabel(ylabel)
    plt.xlabel("Game Played")
    plt.legend()
    #plt.tight_layout()
    plt.show()

def avg_rewards(df):
    plt.figure(figsize=(10, 5))
    for col in df.columns:
        avg = np.cumsum(df[col]) / np.arange(1, len(df[col]) + 1)
        plt.plot(avg)
    plt.show()



if __name__ == '__main__':
    scores_f, durations_f, losses_f = make_dfs([ './results_False2/'])
    avg_losses_f = get_avg_losses(losses_f, durations_f)

    scores_t, durations_t, losses_t = make_dfs([ './results_True2'])
    avg_losses_t = get_avg_losses(losses_t, durations_t)

    davg_losses = {"Random Pipes = false": avg_losses_f,
                    "Random Pipes = true": avg_losses_t}

    dscore = {"Random Pipes = false": scores_f,
              "Random Pipes = true": scores_t}

    dduration = {"Random Pipes = false": durations_f,
                 "Random Pipes = true": durations_t}

    dloss = {"Random Pipes = false": losses_f,
             "Random Pipes = true": losses_t}

    # all_interval_plot_runnning_mean(dduration, title="Duration", ylabel="Duration")
    # all_interval_plot(dduration, title="Duration", ylabel="Duration")
    #
    # all_interval_plot_runnning_mean(dscore, title="Score", ylabel="Score")
    # all_interval_plot(dscore, title="Score", ylabel="Score")
    #
    # all_interval_plot_runnning_mean(dloss, title="Loss", ylabel="Loss")
    # all_interval_plot(dloss, title="Loss", ylabel="Loss")
    #
    # all_interval_plot_runnning_mean(davg_losses, title="Avg loss per game", ylabel="Loss")
    # all_interval_plot(davg_losses, title="Avg loss per game", ylabel="Loss")

    avg_rewards(durations_f)
    avg_rewards(durations_t)
