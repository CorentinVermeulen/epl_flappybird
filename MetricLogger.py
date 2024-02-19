import time
import numpy as np
import matplotlib.pyplot as plt


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    out = (cumsum[N:] - cumsum[:-N]) / N
    prefix = np.repeat(np.nan, N - 1)
    return np.concatenate((prefix, out))


class MetricLogger:
    def __init__(self, save_dir):
        self.save_log = save_dir / "log"
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
        self.ep_scores_plot = save_dir / "score_plot.jpg"
        self.ep_durations_plot = save_dir / "duration_plot.jpg"
        self.ep_losses_plot = save_dir / "loss.jpg"

        # History metrics
        self.ep_scores = []
        self.ep_durations = []
        self.ep_losses = []
        self.ep_steps = []
        self.ep_epsilons = []

        # Timing
        self.record_time = time.time()

        self.n_to_full_memory = None
        self.n_exploring = None
    def log_episode(self, score, duration, loss, step, epsilon):
        "Mark end of episode"
        self.ep_scores.append(score)
        self.ep_durations.append(duration)
        self.ep_losses.append(loss)
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

        for metric in ["ep_scores", "ep_durations", "ep_losses"]:
            self._make_plot(getattr(self, metric), metric, 50, getattr(self, f"{metric}_plot"))
            plt.savefig(getattr(self, f"{metric}_plot"))

    def _make_plot(self, values, name, N, savepath):
        plt.clf()

        plt.plot(values, alpha=0.5)
        plt.plot(running_mean(values, N), 'g', label="Moving Average")

        if self.n_exploring:
            plt.vlines(self.n_exploring, 0, max(values), colors='b', linestyles='dashed', label='End exploration')
        if self.n_to_full_memory:
            plt.vlines(self.n_to_full_memory, 0, max(values), colors='r', linestyles='dashed', label='Memory full')

        plt.title(f"{name} (max: {np.max(values)} - mean: {np.mean(values):.2f})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(savepath)