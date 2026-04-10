import subprocess
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import jax
import optax
from flashbax.buffers.trajectory_buffer import TrajectoryBuffer, BufferState
from flax import nnx
from jax import numpy as jnp, Array

from .tablut.tablut_jax import ROTATION_PERM


def augment_batch(batch: dict[str, Array], rng_key: Array) -> dict[str, Array]:
    """
    Apply a random D4 rotation (0°, 90°, 180°, 270°) to a training batch.

    A single random rotation k is sampled and applied uniformly to the entire
    batch. The observation is spatially rotated, and the policy
    target and legal action mask are permuted via the precomputed ROTATION_PERM table.
    """
    k = jax.random.randint(rng_key, (), 0, 4)
    obs = jax.lax.switch(k, [
        lambda x: x,
        lambda x: jnp.rot90(x, 1, axes=(1, 2)),
        lambda x: jnp.rot90(x, 2, axes=(1, 2)),
        lambda x: jnp.rot90(x, 3, axes=(1, 2)),
    ], batch['observation'])
    return {
        **batch,
        'observation': obs,
        'policy_target': batch['policy_target'][:, ROTATION_PERM[k]],
        'legal_action_mask': batch['legal_action_mask'][:, ROTATION_PERM[k]],
    }


def policy_value_by_player(model_outputs: tuple[jax.Array, ...], player: jax.Array) -> tuple[jax.Array, jax.Array]:
    """
    Select the appropriate policy logits and value from the dual-head output.

    Maps player index {0, 1} to the corresponding head pair:
        player=0 (attacker) → p0_logits, p0_value
        player=1 (defender) → p1_logits, p1_value

    Operates per-sample in a batch via broadcasting.
    """
    p0_logits, p0_value, p1_logits, p1_value = model_outputs
    logits = jnp.where(player[:, None] == 0, p0_logits, p1_logits)
    value = jnp.where(player == 0, p0_value, p1_value)
    return logits, value


def loss_fn(model: nnx.Module, batch: dict[str, Array], train: bool = True) -> tuple[Array, tuple[Array, Array, Array]]:
    logits, value = policy_value_by_player(model(batch['observation'], train=train), batch['player'])

    masked_logits = jnp.where(batch['legal_action_mask'], logits, -1e9)
    policy_loss = optax.softmax_cross_entropy(
        logits=masked_logits, labels=batch['policy_target']
    ).mean()
    value_sq = value.squeeze()
    value_loss = optax.l2_loss(
        predictions=value_sq, targets=batch['value_target']
    ).mean()
    total_loss = policy_loss + value_loss

    pred_sign = jnp.sign(jnp.round(value_sq, decimals=1))
    target_sign = jnp.sign(batch['value_target'])
    value_acc = (pred_sign == target_sign).mean()

    return total_loss, (policy_loss, value_loss, value_acc)


def dir_safe(dir_name: str, parent_dir: Path) -> Path:
    """
    Creates the specified directory if it doesn't already exist.
    Returns the directory path.
    """
    dir_ = parent_dir / dir_name
    dir_.mkdir(parents=True, exist_ok=True)
    return dir_


@partial(jax.jit, backend='cpu', static_argnames=('buffer',))
def add_to_buffer_cpu(buffer_state: BufferState, transitions: dict[str, Array], buffer: TrajectoryBuffer) -> BufferState:
    def add_step(buf_state: BufferState, transition_batch: dict[str, Array]) -> tuple[TrajectoryBuffer, None]:
        return buffer.add(buf_state, transition_batch), None
    new_buffer_state, _ = jax.lax.scan(add_step, buffer_state, transitions)
    return new_buffer_state


def train_step(model: nnx.Module, optimizer: nnx.Optimizer,
               batch: dict[str, Array], rng_key: Array) -> tuple[Array, Array, Array, Array]:
    batch = augment_batch(batch, rng_key)
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, (p_loss, v_loss, v_acc)), grads = grad_fn(model, batch)
    optimizer.update(model, grads)
    return loss, p_loss, v_loss, v_acc


def create_path_dict(root_dir: Path) -> dict[str, Path]:
    model_dir = dir_safe('models', root_dir)
    data_dir = dir_safe('data', root_dir)
    return {
        'checkpoints': dir_safe('checkpoints', model_dir),
        'eval_pool': dir_safe('eval_pool', model_dir),
        'model': dir_safe('model', model_dir),
        'plots': dir_safe('plots', data_dir),
        'metrics': dir_safe('metrics', data_dir),
        'bayeselo': root_dir / 'bayeselo',
        'pgn': root_dir / 'game_results.pgn',
        'training': dir_safe('training_data', root_dir)
    }


def run_bayeselo(pgn_path: Path, bayes_elo_path: Path) -> dict[str, int]:
    """Runs the BayesElo program by executing it as a subprocess.
    Returns a dict with a dict with Elo ratings for each iteration in the pgn file.
    """
    commands = f"""readpgn {pgn_path}
elo
mm
exactdist
ratings
x
x
"""
    process = subprocess.Popen(
        [bayes_elo_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    stdout, stderr = process.communicate(commands)

    ratings = {}
    parsing_table = False

    for line in stdout.splitlines():
        if "Rank Name" in line:
            parsing_table = True
            continue
        if parsing_table:
            parts = line.split()
            if not parts or not parts[0].isdigit():
                if "ResultSet" in line:
                    break
                continue
            ratings[parts[1]] = int(parts[2])

    return ratings


def scale_rewards(internal_rewards: Array, reward_consts: Array) -> Array:
    """Scale raw game rewards using configurable reward constants.

    reward_consts layout: [attacker_win, attacker_loss, defender_win,
                           defender_loss, attacker_draw, defender_draw]
    """
    att_raw = internal_rewards[:, 0]
    def_raw = internal_rewards[:, 1]

    scaled_att = jnp.where(att_raw > 0, reward_consts[0], jnp.where(att_raw < 0, reward_consts[1], reward_consts[4]))
    scaled_def = jnp.where(def_raw > 0, reward_consts[2], jnp.where(def_raw < 0, reward_consts[3], reward_consts[5]))
    return jnp.stack([scaled_att, scaled_def], axis=1)


@dataclass
class GameStats:
    attacker_win_rate: float
    defender_win_rate: float
    draw_rate: float
    avg_length: float
    avg_pieces: float
    avg_entropy: float
    attacker_ev: float
    attacker_score: float
    hm_draw_rate: float


def compute_game_stats(terminals, rewards, step_counts, entropies, pieces_left, half_move_draws) -> GameStats:
    """Compute summary statistics from self-play or opponent-play results."""
    terminals = jax.device_get(terminals)
    attacker_rewards = jax.device_get(rewards)
    step_counts = jax.device_get(step_counts)
    entropies = jax.device_get(entropies)
    pieces_left = jax.device_get(pieces_left)
    half_move_draws = jax.device_get(half_move_draws)

    completed_game_lengths = step_counts[terminals]
    avg_length = float(completed_game_lengths.mean()) if len(completed_game_lengths) > 0 else 0.0

    total_terminated = int(terminals.sum())
    if total_terminated > 0:
        attacker_wins = int((terminals & (attacker_rewards == 1)).sum())
        defender_wins = int((terminals & (attacker_rewards == -1)).sum())
        total_draws = int((terminals & (attacker_rewards == 0)).sum())
        return GameStats(
            attacker_win_rate=attacker_wins / total_terminated,
            defender_win_rate=defender_wins / total_terminated,
            draw_rate=total_draws / total_terminated,
            avg_length=avg_length,
            avg_pieces=float(pieces_left[terminals].mean()),
            avg_entropy=float(entropies.mean()),
            attacker_ev=(attacker_wins - defender_wins) / total_terminated,
            attacker_score=(attacker_wins + 0.5 * total_draws) / total_terminated,
            hm_draw_rate=float(half_move_draws.sum() / total_terminated),
        )

    return GameStats(
        attacker_win_rate=0.0, defender_win_rate=0.0, draw_rate=0.0,
        avg_length=0.0, avg_pieces=0.0, avg_entropy=0.0,
        attacker_ev=0.0, attacker_score=0.0, hm_draw_rate=0.0,
    )


def _format_stats_line(s: GameStats) -> str:
    return (f"Att: {s.attacker_win_rate:.1%}  Def: {s.defender_win_rate:.1%}  Draw: {s.draw_rate:.1%}"
            f"  EV: {s.attacker_ev:+.3f}  Entropy: {s.avg_entropy:.4f}"
            f"  Pieces: {s.avg_pieces:.1f}  HMDraw: {s.hm_draw_rate:.1%}"
            f"  AvgLen: {s.avg_length:.1f}")
