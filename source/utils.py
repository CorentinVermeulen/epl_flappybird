import time
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import re


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    out = (cumsum[N:] - cumsum[:-N]) / N
    prefix = np.repeat(np.nan, N - 1)
    return np.concatenate((prefix, out))


def log_df(df, name, scores, durations, end_dic, test_dic, t):
    df.loc[len(df)] = {'Name': name,
                       'n_to_30': end_dic['n_to_30'],
                       'mean_duration': np.mean(durations),
                       'max_score': max(scores),
                       'test_score': test_dic['score'],
                       'test_duration': test_dic['duration'],
                       'total_time': time.perf_counter() - t
                       }


def get_kpi(scores, durations):
    def get_nsm(scores, last_tier=False):
        if last_tier:
            length = len(scores)
            tier = length // 3
            scores = scores[length - tier:]
        n = np.sum(np.array(scores) == 20)
        return n

    def get_dm(durations, last_tier=False):
        if last_tier:
            length = len(durations)
            tier = length // 3
            scores = durations[length - tier:]
        dm = np.mean(durations)
        return dm

    def get_n_to10sm(scores):
        cumsum = np.cumsum(np.array(scores) == 20)
        index = list(cumsum).index(10) if 10 in list(cumsum) else None
        return index

    return {"nsm": get_nsm(scores),
            "nsm_last": get_nsm(scores, last_tier=True),
            "dm": get_dm(durations),
            "dm_last": get_dm(durations, last_tier=True),
            "n_to_10sm": get_n_to10sm(scores),
            }


class HParams():
    def __init__(self, dict):
        for key, value in dict.items():
            setattr(self, key, value)

    def __str__(self):
        s = "Hyperparameters Config:\n"
        s += "\n".join([f"   {k}: {v}" for k, v in self.__dict__.items()])
        return s

    def update(self, dict):
        for key, value in dict.items():
            setattr(self, key, value)


class MetricLogger:
    def __init__(self, save_dir):
        # self.save_log = save_dir / "log"
        # with open(self.save_log, "w") as f:
        #     f.write(
        #         f"{'Episode':>8}"
        #         f"{'Step':>8}"
        #         f"{'Epsilon':>10}"
        #         f"{'Score':>5}"
        #         f"{'Duration':>5}"
        #         f"{'Loss':>15}"
        #
        #     )
        if type(save_dir) == str:
            save_dir = Path(save_dir)
        self.ep_scores_plot = save_dir / "score_plot.jpg"
        self.ep_durations_plot = save_dir / "duration_plot.jpg"
        self.ep_avg_losses_plot = save_dir / "loss.jpg"

        # History metrics
        self.ep_scores = []
        self.ep_durations = []
        self.ep_avg_losses = []
        self.ep_steps = []
        self.ep_epsilons = []

        # Timing
        self.record_time = time.time()

        self.n_to_full_memory = None
        self.n_exploring = None

        plt.figure(figsize=(10, 5))

    def log_episode(self, score, duration, loss, step, epsilon):
        "Mark end of episode"
        self.ep_scores.append(score)
        self.ep_durations.append(duration)
        self.ep_avg_losses.append(loss)
        self.ep_steps.append(step)
        self.ep_epsilons.append(epsilon)

    def record(self):
        # with open(self.save_log, "a") as f:
        #     f.write(
        #         f"{episode:8d}"
        #         f"{step:8d}"
        #         f"{epsilon:10.3f}"
        #         f"{mean_ep_reward:15.3f}"
        #         f"{mean_ep_length:15.3f}"
        #         f"{mean_ep_loss:15.3f}"
        #         f"{mean_ep_q:15.3f}"
        #         f"{time_since_last_record:15.3f}\n"
        #     )

        for metric in ["ep_scores", "ep_durations", "ep_avg_losses"]:
            self._make_plot(getattr(self, metric), metric, 50, getattr(self, f"{metric}_plot"))
            # self._make_plot2(getattr(self, metric), metric, 50, getattr(self, f"{metric}_plot"), self.ep_epsilons, "Epsilon")
            plt.savefig(getattr(self, f"{metric}_plot"))

    def _make_plot(self, values, name, N, savepath):
        plt.clf()

        plt.plot(values, alpha=0.5, label="data")
        if len(values) > 100:
            plt.plot(running_mean(values, N), 'g', label="Moving Average")

        if self.n_exploring:
            plt.vlines(self.n_exploring, 0, max(values), colors='b', linestyles='dashed', label='End exploration')
        if self.n_to_full_memory:
            plt.vlines(self.n_to_full_memory, 0, max(values), colors='r', linestyles='dashed', label='Memory full')

        plt.title(f"max: {np.max(values)} - mean: {np.mean(values):.2f}")
        plt.xlabel('games_played')
        plt.ylabel(name)
        plt.legend()
        plt.tight_layout()
        plt.savefig(savepath)

    def _make_plot2(self, values, name, N, savepath, values2=None, name2=None):
        plt.clf()
        fig, ax1 = plt.subplots()
        x = np.arange(len(values))
        ax1.set_xlabel('episodes')

        ax1.plot(x, values, 'g-', alpha=0.5)
        ax1.plot(x, running_mean(values, N), 'g', label="Moving Average")
        ax1.set_ylabel(name, color='g')

        if values2:
            ax2 = ax1.twinx()
            ax2.plot(x, values2, 'r-', alpha=0.7)
            ax2.set_ylabel(name2, color='r')

        plt.title(f"{name} (max: {np.max(values)} - mean: {np.mean(values):.2f})")
        # plt.legend()
        plt.tight_layout()
        plt.savefig(savepath)


def avg_duration(df, title, path):
    plt.figure(figsize=(20, 10))
    for col in df.columns:
        avg = np.cumsum(df[col]) / np.arange(1, len(df[col]) + 1)
        plt.plot(avg, label=col)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path + f"/{title}.jpg")
    plt.show()

def plot_losses(df, title, path):
    plt.figure(figsize=(20, 10))
    for col in df.columns:
        rm_l = running_mean(df[col], 50)
        plt.plot(range(len(rm_l)), rm_l, label=col)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path + f"/{title}.jpg")
    plt.show()

def make_experiment_plot(path):
    durations = pd.DataFrame()
    losses = pd.DataFrame()

    for dir in os.listdir(path):
        id = re.findall(r'\((.*?)\)', dir)[0].strip("'")
        if os.path.isdir(os.path.join(path, dir)):
            for file in os.listdir(os.path.join(path, dir)):
                if file.endswith(".csv"):
                    df = pd.read_csv(os.path.join(path, dir, file))
                    df = df.rename(columns={"durations": f"d_{id}", "loss": f"l_{id}"})
                    durations = pd.concat([durations, df[f'd_{id}']], axis=1)
                    losses = pd.concat([losses, df[f'l_{id}']], axis=1)

    avg_duration(durations, "Average Durations", path)
    plot_losses(losses, "Losses", path)

if __name__ == "__main__":
    make_experiment_plot("../../experiments/layer_size/")