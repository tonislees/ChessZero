from functools import partial

import jax
from flax import nnx
from jax import numpy as jnp, lax, Array

from .tablut.tablut import State, Tablut
from .tablut.tablut_jax import MAX_HALF_MOVE_COUNT
from .mcts import run_mcts
from .utils import policy_value_by_player, scale_rewards
from pgx._src.types import PRNGKey


"""Self-play data generation for HnefataflZero training.

Two modes of play:
    self_play():             current model vs itself, random side assignment
    self_play_vs_opponent(): current model vs a past checkpoint, fixed side assignment

Both functions use jax.lax.scan over num_steps, with automatic environment reset
on terminal states. Training targets are computed via a backward scan that
propagates returns from terminal rewards, negating at each step to account for
the alternating-player zero-sum structure.
"""

_self_play_pbar = None

def set_pbar(pbar):
    global _self_play_pbar
    _self_play_pbar = pbar


def _build_transition(state: State, next_env_state: State, mcts_output,
                      reward_consts: Array, batch_size: int, env: Tablut) -> dict:
    """Build the per-step transition dict used by both self-play modes."""
    internal_rewards = jax.vmap(env.game.rewards)(next_env_state.game_state)
    att_raw = internal_rewards[:, 0]

    scaled_internal_rewards = scale_rewards(internal_rewards, reward_consts)

    scaled_player_rewards = jax.vmap(lambda r, order: r[order])(
        scaled_internal_rewards, next_env_state._player_order
    )

    batch_indices = jnp.arange(batch_size)
    current_player_rewards = scaled_player_rewards[batch_indices, state.current_player]

    probs = mcts_output.action_weights
    entropy = -jnp.sum(probs * jnp.log(probs + 1e-8), axis=-1)

    pieces_left = jnp.sum(next_env_state.game_state.board != 0, axis=-1)

    is_half_move_draw = next_env_state.terminated & (next_env_state.game_state.half_move_count >= MAX_HALF_MOVE_COUNT)

    return {
        "observation": state.observation,
        "policy_target": mcts_output.action_weights,
        "reward": current_player_rewards,
        "attacker_reward": att_raw,
        "terminated": next_env_state.terminated,
        "step_count": next_env_state._step_count,
        "legal_action_mask": state.legal_action_mask,
        "player": (state.game_state.color + 1) // 2,
        "entropy": entropy,
        "pieces_left": pieces_left,
        "is_half_move_draw": is_half_move_draw
    }


def _auto_reset(env: Tablut, next_env_state: State, key_reset: PRNGKey,
                batch_size: int, forced_player_order: Array | None = None) -> State:
    """Auto-reset terminated environments, optionally forcing a player_order."""
    reset_keys = jax.random.split(key_reset, batch_size)
    reset_states = jax.vmap(lambda k: env.init(k))(reset_keys)

    if forced_player_order is not None:
        forced_order = jnp.broadcast_to(forced_player_order[None, :], (batch_size, 2))
        reset_states = reset_states.replace( # type: ignore[attr-defined]
            _player_order=forced_order,
            current_player=jnp.full(batch_size, forced_player_order[0], dtype=jnp.int32)
        )

    def select_if_terminated(reset_val, next_val):
        shape = (batch_size,) + (1,) * (next_val.ndim - 1)
        mask = next_env_state.terminated.reshape(shape)
        return jnp.where(mask, reset_val, next_val)

    return jax.tree_util.tree_map(select_if_terminated, reset_states, next_env_state)


def _backward_scan(model: nnx.Module, final_env_state: State, history: dict):
    """Compute value targets via backward return propagation.

    The backward scan (step_back) computes value targets by propagating terminal
    rewards backward through the game trajectory. The negation (-next_return) at
    each step handles the zero-sum alternating-player structure: my opponent's
    gain is my loss.

    For non-terminal steps at the end of the scan window, the value is
    bootstrapped from the network's prediction on the final state.
    """
    # If the game doesn't end in a terminal state, bootstrap using the network's prediction
    _, next_value = policy_value_by_player(model(final_env_state.observation), final_env_state.current_player)

    def step_back(next_return, transition):
        return_ = jnp.where(transition['terminated'], transition['reward'], -next_return)
        out_transition = {
            'observation': transition['observation'],
            'policy_target': transition['policy_target'],
            'value_target': return_,
            'legal_action_mask': transition['legal_action_mask'],
            'player': transition['player']
        }
        return return_, out_transition

    _, final_transitions = jax.lax.scan(step_back, next_value, history, reverse=True)
    return final_transitions


def _collect_results(final_env_state, final_transitions, history):
    """Pack scan outputs into the standard return tuple."""
    return (
        final_env_state,
        final_transitions,
        history['terminated'],
        history['attacker_reward'],
        history['step_count'],
        history['entropy'],
        history['pieces_left'],
        history['is_half_move_draw']
    )


@partial(nnx.jit, static_argnames=('num_steps', 'batch_size', 'num_simulations', 'env'))
def self_play(model: nnx.Module, env_state: State, rng_key: PRNGKey, num_steps: int, num_simulations: int,
              env: Tablut, batch_size: int, reward_consts: Array) -> tuple[State, dict, Array, Array, Array, Array, Array, Array]:
    """
    Generate training data from self-play with the current model.

    Returns:
        (final_env_state, transitions, terminals, attacker_rewards,
         step_counts, entropies, pieces_left, half_move_draws)
        where transitions is a dict with keys: observation, policy_target,
        value_target, legal_action_mask, player.
    """
    graph_def, model_state = nnx.split(model)

    def step_fn(state: State, key: PRNGKey) -> tuple[State, dict]:
        key_reset, key_search = jax.random.split(key)

        mcts_output = run_mcts(graph_def, model_state, state, key_search,
                               num_simulations, env, reward_consts=reward_consts)
        actions = mcts_output.action
        next_env_state = jax.vmap(env.step)(state, actions)

        # Auto reset if some game is terminal
        auto_reset_state = _auto_reset(env, next_env_state, key_reset, batch_size)

        transition = _build_transition(state, next_env_state, mcts_output,
                                       reward_consts, batch_size, env)

        def update_pbar(_):
            global _self_play_pbar
            if '_self_play_pbar' in globals():
                _self_play_pbar.update(1)

        jax.debug.callback(update_pbar, key)

        return auto_reset_state, transition

    keys = jax.random.split(rng_key, num_steps)
    final_env_state, history = jax.lax.scan(step_fn, env_state, keys)

    final_transitions = _backward_scan(model, final_env_state, history)
    return _collect_results(final_env_state, final_transitions, history)


@partial(nnx.jit, static_argnames=('num_steps', 'batch_size', 'num_simulations', 'env'))
def self_play_vs_opponent(model: nnx.Module, opponent: nnx.Module, env_state: State, rng_key: PRNGKey,
                          num_steps: int, num_simulations: int, env: Tablut, batch_size: int,
                          reward_consts: Array, player_order: Array):
    """
    Generate training data from the current model vs. a fixed past opponent.

    player_order controls side assignment: [1, 0] means the current model
    plays as defender (player 1) and the opponent plays as attacker (player 0).
    All games in the batch use the same player_order to keep the scan uniform
    across the batch dimension.

    The opponent's MCTS output is used for action selection on its turns, but
    only the current model's observations are stored as training data.

    Returns: same structure as self_play().
    """
    graph_def, model_state = nnx.split(model)
    opp_graph_def, opp_model_state = nnx.split(opponent)

    def step_fn(state, key):
        key_reset, key_search = jax.random.split(key)

        is_model_turn = (state.current_player[0] == 0)

        mcts_output = lax.cond(
            is_model_turn,
            lambda: run_mcts(graph_def, model_state, state, key_search,
                             num_simulations, env, reward_consts=reward_consts),
            lambda: run_mcts(opp_graph_def, opp_model_state, state, key_search,
                             num_simulations, env, reward_consts=reward_consts),
        )
        actions = mcts_output.action
        next_env_state = jax.vmap(env.step)(state, actions)

        # Auto reset with forced player_order to keep games in sync
        auto_reset_state = _auto_reset(env, next_env_state, key_reset, batch_size,
                                       forced_player_order=player_order)

        transition = _build_transition(state, next_env_state, mcts_output,
                                       reward_consts, batch_size, env)

        def update_pbar(_):
            global _self_play_pbar
            if '_self_play_pbar' in globals():
                _self_play_pbar.update(1)

        jax.debug.callback(update_pbar, key)

        return auto_reset_state, transition

    keys = jax.random.split(rng_key, num_steps)
    final_env_state, history = jax.lax.scan(step_fn, env_state, keys)

    final_transitions = _backward_scan(model, final_env_state, history)
    return _collect_results(final_env_state, final_transitions, history)