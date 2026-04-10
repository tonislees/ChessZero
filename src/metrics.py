import datetime
import json
import shutil
from pathlib import Path

import numpy as np
from flax import nnx
from matplotlib import pyplot as plt
from omegaconf import DictConfig

from .utils import run_bayeselo


"""
Training metrics tracking and visualization.

MetricsTracker accumulates per-step loss metrics via flax.nnx.MultiMetric,
records per-iteration self-play statistics (win rates, entropy, game length),
and generates standalone PDF plots for different metrics. Metrics
history is persisted as timestamped JSON files alongside PGN backups.
"""

class MetricsTracker:
    def __init__(self, cfg: DictConfig, dirs: dict[str, Path]):
        self.dirs = dirs
        self.cfg = cfg
        self.metrics_history = self._load_metrics(cfg.train.load_checkpoint)
        self.metrics: nnx.MultiMetric = nnx.MultiMetric(
            total_loss=nnx.metrics.Average(argname='total_loss'),
            policy_loss=nnx.metrics.Average(argname='policy_loss'),
            value_loss=nnx.metrics.Average(argname='value_loss'),
            value_acc=nnx.metrics.Average(argname='value_acc')
        )
        self.timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    def update_frames(self, frame_count: int) -> None:
        """
        Calculates the cumulative frame count and appends it to the history.
        """
        if not self.metrics_history['frames']:
            current_total = 0
        else:
            current_total = self.metrics_history['frames'][-1]

        self.metrics_history['frames'].append(current_total + frame_count)

    def update_step(self, total_loss: float, policy_loss: float, value_loss: float, value_acc: float) -> None:
        """
        Feeds the current batch's outputs into the MultiMetric.
        """
        self.metrics.update(
            total_loss=total_loss,
            policy_loss=policy_loss,
            value_loss=value_loss,
            value_acc=value_acc
        )

    def compute_and_record(self) -> None:
        """
        Computes the accumulated metrics, appends them to the history,
        and resets the internal state for the next interval.
        """
        current_metrics = self.metrics.compute()
        self.metrics_history['total_loss'].append(float(current_metrics['total_loss']))
        self.metrics_history['policy_loss'].append(float(current_metrics['policy_loss']))
        self.metrics_history['value_loss'].append(float(current_metrics['value_loss']))
        self.metrics_history['value_acc'].append(float(current_metrics['value_acc']))

        self.metrics.reset()

    def load_latest_metrics(self) -> tuple[dict, list[float]]:
        metrics_dir = self.dirs['training'] / 'metrics'
        metrics_files = sorted(metrics_dir.glob("metrics_*.json"))
        if not metrics_files:
            raise Exception('No metrics file found')
        with open(metrics_files[-1]) as f:
            history = json.load(f)
        frames_per_iter = history['frames'][0]
        n_iters = len(history['frames'])
        frames = [(i + 1) * frames_per_iter / 1e6 for i in range(n_iters)]
        return history, frames

    def plot_elo(self) -> None:
        metrics_dir = self.dirs['training'] / 'metrics'
        ratings = run_bayeselo(list(metrics_dir.glob("game_results_*.pgn"))[0], self.dirs['bayeselo'])

        sorted_items = sorted(ratings.items(), key=lambda x: int(x[0].split('_')[1]))
        frames_per_iter = self.load_latest_metrics()[0]['frames'][0]
        iters = [int(name.split('_')[1]) for name, _ in sorted_items]
        elos = [e - min(ratings.values()) for _, e in sorted_items]
        elo_frames = [i * frames_per_iter / 1e6 for i in iters]

        plt.figure(figsize=(6, 5))
        plt.plot(elo_frames, elos, marker='o', linewidth=2, color='royalblue')
        plt.ylabel("Elo")
        plt.xlabel("Frames (millions)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        elo_path = self.dirs['plots'] / f"elo_plot_{self.timestamp}.pdf"
        plt.savefig(elo_path, dpi=150)
        plt.close()

    def plot_elo_comparison(self) -> None:
        elos_dir = self.dirs['training'] / 'elos'
        metrics_dir = self.dirs['training'] / 'metric'

        plt.figure(figsize=(6, 5))

        pgn_files = sorted(elos_dir.glob("*.pgn"), key=lambda x: int(x.stem))

        colors = ['royalblue', 'crimson', 'forestgreen', 'darkorange', 'purple']
        descriptions = ['Baseline', '+ Augmentation & Buffer', '+ Past-Iteration Self-Play']

        for i, pgn_path in enumerate(pgn_files):
            version_id = int(pgn_path.stem)
            json_path = metrics_dir / f"{version_id}.json"

            if not json_path.exists():
                print(f"Warning: No matching metric json for {pgn_path.name}")
                continue

            ratings = run_bayeselo(pgn_path, self.dirs['bayeselo'])
            sorted_items = sorted(ratings.items(), key=lambda x: int(x[0].split('_')[1]))

            with open(json_path) as f:
                history = json.load(f)

            frames_per_iter = history['frames'][0]
            iters = [int(name.split('_')[1]) for name, _ in sorted_items]
            elo_frames = [it * frames_per_iter / 1e6 for it in iters]

            base_elo = min(ratings.values())
            elos = [e - base_elo for _, e in sorted_items]

            color = colors[i % len(colors)]
            plt.plot(elo_frames, elos, marker='o', markersize=4, linewidth=2,
                     color=color, label=descriptions[version_id - 1])

        plt.ylabel("Elo Rating ")
        plt.xlabel("Frames (millions)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        elo_path = self.dirs['plots'] / f"elo_comparison_{self.timestamp}.pdf"
        plt.savefig(elo_path, dpi=150)
        plt.close()

    def plot_results(self) -> None:
        history, frames = self.load_latest_metrics()

        window = 10
        attacker_win_rates = np.array(history['attacker_win_rate'])
        rolling_avg = np.convolve(attacker_win_rates, np.ones(window) / window, mode='valid')  # type: ignore[arg-type]
        rolling_frames = frames[window - 1:]

        plt.figure(figsize=(6, 5))

        plt.stackplot(
            frames,
            history['attacker_win_rate'],
            history['draw_rate'],
            history['defender_win_rate'],
            labels=['Attacker Win', 'Draw', 'Defender Win'],
            colors=['#4CAF50', '#9E9E9E', '#F44336'],
            alpha=0.85
        )

        plt.plot(
            rolling_frames,
            rolling_avg,
            color='white',
            linewidth=2,
            label=f'{window}-Iter Attacker Avg'
        )

        plt.ylabel("Rate")
        plt.xlabel("Frames (millions)")
        plt.ylim(0, 1)
        plt.legend(loc='upper left', fontsize='small', framealpha=0.5)
        plt.grid(True, alpha=0.2)
        plt.tight_layout()

        outcomes_path = self.dirs['plots'] / f"outcomes_plot_{self.timestamp}.pdf"
        plt.savefig(outcomes_path, dpi=150)
        plt.close()

    def plot_loss(self) -> None:
        history, frames = self.load_latest_metrics()

        plt.figure(figsize=(6, 5))
        plt.plot(frames, history['total_loss'], label='Total Loss', color='black', alpha=0.3, linestyle='--')
        plt.plot(frames, history['policy_loss'], label='Policy (CE)', color='#1f77b4')
        plt.plot(frames, history['value_loss'], label='Value (MSE)', color='#ff7f0e')
        plt.yscale('log')
        plt.xlabel("Frames (millions)")
        plt.ylabel("Loss (Log Scale)")
        plt.legend()
        plt.grid(True, which="both", alpha=0.2)
        plt.tight_layout()
        plt.savefig(self.dirs['plots'] / f"loss_plot_{self.timestamp}.pdf")
        plt.close()

    def plot_entropy(self) -> None:
        history, frames = self.load_latest_metrics()

        plt.figure(figsize=(6, 5))
        plt.plot(frames, history['entropy'], color='#9467bd', linewidth=2)
        plt.xlabel("Frames (millions)")
        plt.ylabel("Entropy (nats)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.dirs['plots'] / f"entropy_plot_{self.timestamp}.pdf")
        plt.close()

    def plot_avg_pieces(self) -> None:
        history, frames = self.load_latest_metrics()

        plt.figure(figsize=(6, 5))
        plt.plot(frames, history['pieces_left'], color='#8c564b', linewidth=2)
        plt.xlabel("Frames (millions)")
        plt.ylabel("Piece Count")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.dirs['plots'] / f"pieces_plot_{self.timestamp}.pdf")
        plt.close()

    def save_metrics(self) -> None:
        """
        Writes current metrics history into a JSON file and backups the PGN.
        """
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        filename = f"metrics_{timestamp}.json"
        file_dir = self.dirs['metrics'] / filename
        with open(file_dir, 'w') as f:
            json.dump(self.metrics_history, f, indent=4)

        if self.dirs['pgn'].exists():
            pgn_backup = self.dirs['metrics'] / f"game_results_{timestamp}.pgn"
            shutil.copy(self.dirs['pgn'], pgn_backup)

    def _load_metrics(self, load_checkpoint: bool) -> dict[str, list[float]]:
        """
        Loads the most recent metrics history from the disk if load_checkpoint is true.
        Returns the previous metrics or a new dict.
        """
        default_metrics = {
            'total_loss': [],
            'policy_loss': [],
            'value_loss': [],
            'value_acc': [],
            'attacker_win_rate': [],
            'defender_win_rate': [],
            'draw_rate': [],
            'frames': [],
            'game_lengths': [],
            'pieces_left': [],
            'entropy': [],
            'attacker_ev': [],
            'attacker_score': []
        }

        if not load_checkpoint or not self.dirs['metrics'].exists():
            return default_metrics

        metric_files = list(self.dirs['metrics'].glob("metrics_*.json"))

        if not metric_files:
            return default_metrics

        latest_file = sorted(metric_files)[-1]

        print(f"Loading metrics from {latest_file.name}")
        with open(latest_file, 'r') as f:
            loaded_metrics = json.load(f)

        return loaded_metrics