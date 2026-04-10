import random
import time
from pathlib import Path

import flashbax as fbx
import hydra
import jax
from jax import Array
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from jax.experimental import mesh_utils
import jax.numpy as jnp
import optax
from flax import nnx
from omegaconf import DictConfig
import orbax.checkpoint as ocp
from tqdm import tqdm

from .evaluation import Evaluator
from .tablut.tablut import Tablut, State
from .metrics import MetricsTracker
from .model import TablutZeroNet
from .self_play import self_play, self_play_vs_opponent, set_pbar
from .utils import add_to_buffer_cpu, train_step, create_path_dict, GameStats, compute_game_stats, _format_stats_line

"""
AlphaZero training loop orchestrator for TablutZero.

Each iteration consists of:
    1. Self-play data generation (split between self-play and opponent-play batches)
    2. Training on uniformly sampled replay buffer data
    3. Periodic evaluation against a pool of past checkpoints (BayesElo rating)
    4. Periodic checkpointing of model, optimizer, and buffer state

The Coach class manages the full pipeline: environment setup, multi-device
sharding, replay buffer, and checkpoint save/restore.
"""


class Coach:
    def __init__(self, cfg: DictConfig):
        print('Initializing coach...')
        self.cfg = cfg
        self.rngs: nnx.Rngs = nnx.Rngs(cfg.train.seed)
        self.checkpointer = ocp.StandardCheckpointer()

        # Multiple devices
        self.devices = mesh_utils.create_device_mesh((len(jax.devices()),))
        self.mesh = Mesh(self.devices, axis_names=('batch',))
        self.data_sharding = NamedSharding(self.mesh, PartitionSpec('batch'))
        self.replicated_sharding = NamedSharding(self.mesh, PartitionSpec())

        # Directories
        root_dir = self.root = Path(__file__).resolve().parents[1]
        self.dirs = create_path_dict(root_dir)
        if not cfg.train.load_checkpoint:
            self.dirs['pgn'].unlink(missing_ok=True)

        # Buffer
        cpu_device = jax.devices('cpu')[0]
        min_buffer_size = cfg.train.batch_size * cfg.train.self_play_steps
        self.buffer = fbx.make_flat_buffer(
            max_length=min_buffer_size * cfg.train.buffer_multiplier,
            min_length=min_buffer_size,
            sample_batch_size=cfg.train.batch_size,
            add_batch_size=cfg.train.batch_size
        )

        with jax.default_device(cpu_device):
            example_transition = {
                "observation": jnp.zeros((9, 9, 43), dtype=jnp.float32),
                "policy_target": jnp.zeros((81 * 32,), dtype=jnp.float32),
                "value_target": jnp.zeros((), dtype=jnp.float32),
                "legal_action_mask": jnp.zeros((81 * 32,), dtype=jnp.bool_),
                "player": jnp.zeros((), dtype=jnp.int32)
            }
        self.sample_fn = jax.jit(self.buffer.sample, backend='cpu')

        # Model & optimizer
        self._init_or_restore_checkpoint(example_transition, cpu_device)
        graph_def, state = nnx.split((self.model, self.optimizer))
        state = jax.tree_util.tree_map(lambda x: jax.device_put(x, self.replicated_sharding), state)
        self.model, self.optimizer = nnx.merge(graph_def, state)

        # Environment
        self.env = Tablut()

        self.opp_ratio = cfg.train.get('opponent_ratio', 0.25)
        self.opp_batch = int(cfg.train.batch_size * self.opp_ratio)
        self.self_batch = cfg.train.batch_size - self.opp_batch

        key_env = jax.random.PRNGKey(cfg.train.seed + 1)
        key_self, key_opp = jax.random.split(key_env)

        # Persistent env state for self-play portion
        self.env_state_self = jax.jit(jax.vmap(self.env.init))(
            jax.random.split(key_self, self.self_batch)
        )
        self.env_state_self = jax.tree_util.tree_map(
            lambda x: jax.device_put(x, self.data_sharding), self.env_state_self
        )

        # Setup
        self.train_step_fn = nnx.jit(train_step)
        self.last_iteration = self._get_last_iteration() if cfg.train.load_checkpoint else 0
        self.evaluator = Evaluator(cfg, self.dirs, self.rngs, self.model, self.checkpointer, self.env)
        self.metrics_tracker = MetricsTracker(cfg, self.dirs)
        self.reward_consts = [1, -1, 1, -1, 0.0, 0.0]
        # [attacker_win_r, attacker_loss_r, defender_win_r, defender_loss_r, attacker_draw_r, defender_draw_r]

    def _create_optimizer(self) -> nnx.Optimizer:
        total_training_steps = 400 * self.cfg.train.num_epochs * self.cfg.train.self_play_steps

        schedule = optax.warmup_cosine_decay_schedule(
            init_value=1e-4,
            peak_value=2e-3,
            warmup_steps=500,
            decay_steps=total_training_steps,
            end_value=1e-5
        )

        return nnx.Optimizer(
            self.model,
            optax.chain(
                optax.clip_by_global_norm(1.0),
                optax.adamw(learning_rate=schedule, weight_decay=1e-4)
            ),
            wrt=nnx.Param
        )

    def _get_last_iteration(self) -> int:
        dirs = [d.name for d in self.dirs['eval_pool'].iterdir() if d.is_dir()]
        dirs = list(map(lambda x: int(x.split('_')[1]), dirs))
        dirs.sort()
        print(dirs)
        return dirs[-1]

    def _init_or_restore_checkpoint(self, example_transition: dict[str, Array], cpu_device: jax.Device) -> None:
        load = self.cfg.train.load_checkpoint
        ckpt_dir = self.dirs['checkpoints']
        has_checkpoint = load and ckpt_dir.exists() and any(ckpt_dir.iterdir())

        self.model = TablutZeroNet(
            depth=self.cfg.model.depth,
            filter_count=self.cfg.model.filter_count,
            rngs=self.rngs
        )

        if has_checkpoint:
            print("Restoring from checkpoint...")
            graph_def, abstract_model = nnx.split(self.model)

            temp_opt = self._create_optimizer()
            _, abstract_opt = nnx.split(temp_opt)
            del temp_opt

            with jax.default_device(cpu_device):
                full_buffer = self.buffer.init(example_transition)

            abstract_checkpoint = {
                'model': abstract_model,
                'optimizer': abstract_opt,
                'buffer': full_buffer
            }

            restored = self.checkpointer.restore(
                ckpt_dir,
                target=abstract_checkpoint
            )
            del abstract_model, abstract_opt

            self.model = nnx.merge(graph_def, restored['model'])
            self.optimizer = self._create_optimizer()
            nnx.update(self.optimizer, restored['optimizer'])
            self.buffer_state = restored['buffer']

            del restored, full_buffer
        else:
            self.optimizer = self._create_optimizer()
            with jax.default_device(cpu_device):
                self.buffer_state = self.buffer.init(example_transition)

    def train(self) -> None:
        eval_interval = self.cfg.train.eval_interval
        eval_start = self.cfg.train.eval_start
        save_interval = self.cfg.train.save_interval

        for i in range(self.cfg.train.iterations):
            start_time = time.time()
            iteration = i + self.last_iteration + 1
            print(f"--- Iteration {iteration} ---")

            self._run_self_play_loop()
            self._run_training_loop()

            self.metrics_tracker.update_frames(self.cfg.train.self_play_steps * self.cfg.train.batch_size)

            t_loss = self.metrics_tracker.metrics_history['total_loss'][-1]
            p_loss = self.metrics_tracker.metrics_history['policy_loss'][-1]
            v_loss = self.metrics_tracker.metrics_history['value_loss'][-1]
            v_acc = self.metrics_tracker.metrics_history['value_acc'][-1]

            print(f"    Total Loss: {t_loss:.4f} (Policy: {p_loss:.4f}, Value: {v_loss:.4f}, Acc: {v_acc:.1%})",
                  flush=True)

            if iteration % eval_interval == 0 and iteration >= eval_start:
                self.evaluator.evaluate_model(iteration)

            if iteration % save_interval == 0:
                self._save_progress()

            elapsed = time.time() - start_time
            print(f"    Iteration {iteration} took {elapsed / 60:.2f} minutes.\n", flush=True)

        self._save_progress()

    def _load_random_sp_opponent(self) -> tuple[nnx.Module, str]:
        """Load a random opponent from the eval pool for mixed self-play."""
        pool = self.evaluator.eval_pool

        name = random.choice(list(pool.keys()))
        graph_def, _ = nnx.split(self.model)
        gpu_state = jax.device_put(pool[name])
        opponent = nnx.merge(graph_def, gpu_state)
        opponent.eval()
        return opponent, name

    def _init_opponent_env(self, player_order: Array) -> State:
        """Create a fresh opponent env state with forced player_order."""
        key = self.rngs.split()
        env_state_opp = jax.jit(jax.vmap(self.env.init))(
            jax.random.split(key, self.opp_batch)
        )
        forced_order = jnp.broadcast_to(player_order[None, :], (self.opp_batch, 2))
        env_state_opp = env_state_opp.replace(
            _player_order=forced_order,
            current_player=jnp.full(self.opp_batch, player_order[0], dtype=jnp.int32)
        )
        env_state_opp = jax.tree_util.tree_map(
            lambda x: jax.device_put(x, self.data_sharding), env_state_opp
        )
        return env_state_opp

    def _run_self_play_loop(self) -> None:
        self.model.eval()

        steps = self.cfg.train.self_play_steps
        opponent, opponent_name = self._load_random_sp_opponent()

        print(
            f"Generating data ({steps} steps x {self.self_batch}+{self.opp_batch} games, opponent: {opponent_name})...")

        player_order = jnp.array([1, 0])

        rng_key_self = self.rngs.split()
        rng_key_opp = self.rngs.split()

        pbar = tqdm(total=steps * 2, desc="Self-Play", mininterval=self.cfg.train.tqdm_interval,
                    ncols=100, unit='steps')
        set_pbar(pbar)

        reward_consts = jnp.array(self.reward_consts, dtype=jnp.float32)

        (self.env_state_self, self_transitions, self_terminals, self_rewards,
         self_step_counts, self_entropies, self_pieces_left, self_hm_draws) = self_play(
            model=self.model,
            env_state=self.env_state_self,
            rng_key=rng_key_self,
            num_steps=steps,
            num_simulations=self.cfg.mcts.simulations,
            env=self.env,
            batch_size=self.self_batch,
            reward_consts=reward_consts
        )

        env_state_opp = self._init_opponent_env(player_order)

        (_, opp_transitions, opp_terminals, opp_rewards,
         opp_step_counts, opp_entropies, opp_pieces_left, opp_hm_draws) = self_play_vs_opponent(
            model=self.model,
            opponent=opponent,
            env_state=env_state_opp,
            rng_key=rng_key_opp,
            num_steps=steps,
            num_simulations=self.cfg.mcts.simulations,
            env=self.env,
            batch_size=self.opp_batch,
            reward_consts=reward_consts,
            player_order=player_order
        )

        pbar.close()
        set_pbar(None)

        # Self-play stats (recorded to metrics)
        self_stats = compute_game_stats(
            self_terminals, self_rewards, self_step_counts,
            self_entropies, self_pieces_left, self_hm_draws
        )
        self._record_stats(self_stats)
        print(f"  [self-play]  {_format_stats_line(self_stats)}")

        # Opponent-play stats (logged only)
        opp_stats = compute_game_stats(
            opp_terminals, opp_rewards, opp_step_counts,
            opp_entropies, opp_pieces_left, opp_hm_draws
        )
        print(f"  [vs {opponent_name} as Defender]  {_format_stats_line(opp_stats)}")

        all_transitions = jax.tree_util.tree_map(
            lambda a, b: jnp.concatenate([a, b], axis=1),
            self_transitions, opp_transitions
        )
        all_transitions_cpu = jax.device_get(all_transitions)
        self.buffer_state = add_to_buffer_cpu(self.buffer_state, all_transitions_cpu, self.buffer)

    def _record_stats(self, s: GameStats) -> None:
        """Append self-play game stats to the metrics tracker."""
        for value, name in zip(
                (s.attacker_win_rate, s.defender_win_rate, s.draw_rate, s.avg_length, s.avg_pieces,
                 s.avg_entropy, s.attacker_ev, s.attacker_score),
                ('attacker_win_rate', 'defender_win_rate', 'draw_rate', 'game_lengths', 'pieces_left',
                 'entropy', 'attacker_ev', 'attacker_score')
        ):
            self.metrics_tracker.metrics_history[name].append(value)

    def _run_training_loop(self) -> None:
        self.model.train()

        total_steps = self.cfg.train.self_play_steps * self.cfg.train.num_epochs
        pbar = tqdm(range(total_steps), desc="Training", mininterval=self.cfg.train.tqdm_interval,
                    ncols=100, unit='steps')

        for _ in pbar:
            rng_key = self.rngs.split()
            batch = self.sample_fn(self.buffer_state, rng_key)
            training_data = batch.experience.first
            training_data = jax.tree_util.tree_map(
                lambda x: jax.device_put(x, self.data_sharding), training_data
            )

            aug_key = self.rngs.split()
            loss, p_loss, v_loss, v_acc = self.train_step_fn(
                self.model,
                self.optimizer,
                training_data,
                aug_key
            )

            self.metrics_tracker.update_step(total_loss=loss, policy_loss=p_loss, value_loss=v_loss, value_acc=v_acc)

            pbar.set_postfix({
                'L': f"{loss:.4f}",
                'P': f"{p_loss:.4f}",
                'V': f"{v_loss:.4f}",
                'Acc': f"{v_acc:.1%}"
            })

        self.metrics_tracker.compute_and_record()

    def _save_progress(self) -> None:
        """Saves the model parameters, loss data, and evaluation data."""
        _, model_state = nnx.split(self.model)
        _, opt_state = nnx.split(self.optimizer)
        checkpoint = {
            'model': model_state,
            'optimizer': opt_state,
            'buffer': self.buffer_state
        }
        self.checkpointer.save(self.dirs['checkpoints'], checkpoint, force=True)
        self.checkpointer.wait_until_finished()
        self.checkpointer.save(self.dirs['model'], {'model': model_state}, force=True)
        self.checkpointer.wait_until_finished()
        self.evaluator.save_eval_pool()
        self.metrics_tracker.save_metrics()


@hydra.main(version_base=None, config_path='..', config_name='config')
def main(cfg: DictConfig):
    coach = Coach(cfg)
    coach.train()


if __name__ == '__main__':
    main()