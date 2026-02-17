import datetime
from functools import partial
from pathlib import Path

import flashbax as fbx
import hydra
import jax
import jax.numpy as jnp
import optax
from flax import nnx
from matplotlib import pyplot as plt
from omegaconf import DictConfig
import orbax.checkpoint as ocp
from tqdm import tqdm

from src.hnefatafl.hnefatafl import Hnefatafl
from src.mcts import run_mcts
from src.model import HnefataflZeroNet


def rl_loss_fn(model, batch, train=True):
    logits, value = model(batch['observation'], train=train)
    policy_loss = optax.softmax_cross_entropy(
        logits=logits, labels=batch['policy_target']
    ).mean()
    value_loss = optax.l2_loss(
        predictions=value.squeeze(), targets=batch['value_target']
    ).mean()
    total_loss = policy_loss + value_loss
    return total_loss, (policy_loss, value_loss)


@nnx.jit
def train_step_rl(model, optimizer, batch):
    grad_fn = nnx.value_and_grad(rl_loss_fn, has_aux=True)
    (loss, (p_loss, v_loss)), grads = grad_fn(model, batch)
    optimizer.update(model, grads)
    return loss, p_loss, v_loss


@partial(nnx.jit, static_argnames=('num_simulations', 'env', 'buffer'))
def self_play_step(model, env_state, buffer_state, rng_key, num_simulations, env, buffer):
    key_search, key_act = jax.random.split(rng_key)

    mcts_output = run_mcts(model, env_state, key_search, num_simulations, env)
    actions = mcts_output.action
    mcts_values = mcts_output.search_tree.node_values[:, 0]
    next_env_state = jax.vmap(env.step)(env_state, actions)
    batch_indices = jnp.arange(env_state.current_player.shape[0])
    current_player_rewards = next_env_state.rewards[batch_indices, env_state.current_player]
    final_value_target = jnp.where(
        next_env_state.terminated,
        current_player_rewards,
        mcts_values
    )

    transition = {
        "observation": env_state.observation,
        "policy_target": mcts_output.action_weights,
        "value_target": final_value_target
    }
    new_buffer_state = buffer.add(buffer_state, transition)

    return next_env_state, new_buffer_state


class Coach:
    def __init__(self, cfg: DictConfig):
        # Directories
        root_dir = self.root = Path(__file__).resolve().parents[1]
        self.model_dir = root_dir / 'models'
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir = self.model_dir / 'checkpoints'
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.plot_dir = root_dir / 'plots'
        self.plot_dir.mkdir(parents=True, exist_ok=True)

        # Setup
        self.cfg = cfg
        self.seed = cfg.train.seed
        self.rngs: nnx.Rngs = nnx.Rngs(self.seed)
        self.checkpointer = ocp.StandardCheckpointer()
        self.num_epochs = cfg.train.num_epochs
        self.num_simulations = cfg.mcts.simulations
        self.metrics_history = {
            'train_loss': [],
            'train_policy_loss': [],
            'train_value_loss': [],
        }
        self.metrics: nnx.MultiMetric = nnx.MultiMetric(
            accuracy=nnx.metrics.Accuracy(),
            loss=nnx.metrics.Average(argname='loss'),
        )

        # Buffer
        self.buffer = fbx.make_flat_buffer(
            max_length=cfg.buffer.size,
            min_length=cfg.buffer.min_size,
            sample_batch_size=cfg.train.batch_size,
            add_batch_size=cfg.train.batch_size
        )
        example_transition = {
            "observation": jnp.zeros((11, 11, 43), dtype=jnp.float32),
            "policy_target": jnp.zeros((121 * 40,), dtype=jnp.float32),
            "value_target": jnp.zeros((), dtype=jnp.float32)
        }
        self.buffer_state = self.buffer.init(example_transition)

        # Environment
        self.train_env = Hnefatafl()
        key_env = jax.random.PRNGKey(self.seed + 1)
        self.env_state = jax.jit(jax.vmap(self.train_env.init))(
            jax.random.split(key_env, cfg.train.batch_size)
        )

        # Model & optimizer
        self.model: nnx.Module = HnefataflZeroNet(depth=cfg.model.depth, filter_count=cfg.model.filter_count,
                                              rngs=self.rngs)
        self.optimizer: nnx.Optimizer = nnx.Optimizer(
            self.model, optax.adamw(learning_rate=cfg.train.learning_rate), wrt=nnx.Param
        )
        if self.checkpoints_dir.exists() and any(self.checkpoints_dir.iterdir()):
            graphdef, abstract_state = nnx.split(self.model)
            restored_state = self.checkpointer.restore(self.checkpoints_dir, abstract_state)
            self.model = nnx.merge(graphdef, restored_state)
            print(f"Restored model from {self.checkpoints_dir}")

    def learn_rl(self):
        print("Starting RL Loop...")

        for iteration in range(self.cfg.train.iterations):
            print(f"--- Iteration {iteration} ---")
            self._run_self_play_loop()
            self._run_training_loop()
            self.save_model()
        self._plot_metrics_rl()

    def _run_self_play_loop(self):
        self.model.eval()

        steps = self.cfg.train.self_play_steps
        print(f"Generating data ({steps} steps x {self.cfg.train.batch_size} games)...")

        for _ in tqdm(range(steps)):
            rng_key = self.rngs.split()

            self.env_state, self.buffer_state = self_play_step(
                self.model,
                self.env_state,
                self.buffer_state,
                rng_key,
                self.cfg.mcts.simulations,
                self.train_env,
                self.buffer
            )

    def _run_training_loop(self):
        self.model.train()

        steps = self.cfg.train.num_epochs
        pbar = tqdm(range(steps), desc="Training")

        for _ in pbar:
            rng_key = self.rngs.split()
            batch = self.buffer.sample(self.buffer_state, rng_key)
            training_data = batch.experience.first

            loss, p_loss, v_loss = train_step_rl(
                self.model,
                self.optimizer,
                training_data
            )

            self.metrics_history['train_loss'].append(loss.item())
            self.metrics_history['train_policy_loss'].append(p_loss.item())
            self.metrics_history['train_value_loss'].append(v_loss.item())

            pbar.set_postfix({
                'L': f"{loss:.4f}",
                'P': f"{p_loss:.4f}",
                'V': f"{v_loss:.4f}"
            })

    def _plot_metrics_rl(self):
        def smooth(scalars, weight=0.9):
            if not scalars: return []
            last = scalars[0]
            smoothed = []
            for point in scalars:
                smoothed_val = last * weight + (1 - weight) * point
                smoothed.append(smoothed_val)
                last = smoothed_val
            return smoothed

        fig, axes = plt.subplots(1, 3, figsize=(20, 6))

        metrics = [
            ('Total Loss', self.metrics_history['train_loss']),
            ('Policy Loss (Cross Entropy)', self.metrics_history['train_policy_loss']),
            ('Value Loss (MSE)', self.metrics_history['train_value_loss'])
        ]

        for ax, (title, data) in zip(axes, metrics):
            if not data: continue

            ax.plot(data, alpha=0.25, color='gray', label='Raw')
            smoothed_data = smooth(data, weight=0.95)
            ax.plot(smoothed_data, alpha=1.0, linewidth=2, label='Smoothed')

            ax.set_title(title)
            ax.set_xlabel('Training Steps')
            ax.grid(True, alpha=0.3)
            ax.legend()

            iterations = len(data) // self.cfg.train.num_epochs
            for i in range(iterations):
                ax.axvline(x=i * self.cfg.train.num_epochs, color='red', alpha=0.1)

        plt.tight_layout()

        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        plot_path = self.plot_dir / f"RL_Training_Metrics_{timestamp}.png"
        plt.savefig(plot_path)
        plt.close(fig)
        print(f"Saved training plot to {plot_path}")


    def save_model(self):
        _, state = nnx.split(self.model)
        self.checkpointer.save(self.checkpoints_dir, state, force=True)
        print(f"Checkpoint saved to {self.checkpoints_dir}")

@hydra.main(version_base=None, config_path='..', config_name='config')
def main(cfg: DictConfig):
    coach = Coach(cfg)
    coach.learn_rl()


if __name__ == '__main__':
    main()
