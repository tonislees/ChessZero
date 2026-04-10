from functools import partial

import jax
import mctx
from flax import nnx
import jax.numpy as jnp
from jax import Array

from .tablut.tablut import State, Tablut
from .utils import policy_value_by_player, scale_rewards

"""
Gumbel MuZero MCTS via the mctx library.

Uses mctx.gumbel_muzero_policy with a recurrent function that steps the game
internally (no external simulator calls during search). Reward shaping is applied
inside the tree through configurable reward_consts.

The discount convention follows mctx's two-player formulation:
    discount = -1.0 for non-terminal states (negates value for opponent's turn)
    discount =  0.0 for terminal states (cuts off future value)
"""

def recurrent_fn(model_state: nnx.GraphState, rng_key: Array, action: Array, embedding, env: Tablut,
                 graph_def: nnx.GraphDef, reward_consts: Array) -> tuple[mctx.RecurrentFnOutput, State]:
    """
    MCTS expansion function: simulate one game step and return (output, next_state).

    Applies reward scaling via reward_consts before passing rewards into the tree.
    Reward_consts format: [a_win, a_loss, d_win, d_loss, a_draw, d_draw] where
    each element replaces the corresponding raw {-1, 0, 1} outcome.

    The returned reward is from the perspective of the player who just moved
    (embedding.current_player), and the discount is -1.0 (non-terminal) or
    0.0 (terminal) to handle the alternating-player value negation.
    """
    next_game_state = jax.vmap(env.game.step)(embedding.game_state, action)

    batch_idx = jnp.arange(action.shape[0])[:, None]
    color_idx = (next_game_state.color + 1) // 2

    is_term, raw_rewards = jax.vmap(env.game.mcts_status)(next_game_state)
    scaled_rewards = scale_rewards(raw_rewards, reward_consts)

    next_state = embedding.replace(
        game_state=next_game_state,
        terminated=is_term,
        rewards=scaled_rewards[batch_idx, embedding._player_order],
        current_player=embedding._player_order[jnp.arange(action.shape[0]), color_idx]
    )

    next_obs = jax.vmap(env.game.observe)(next_game_state)

    local_model = nnx.merge(graph_def, model_state)
    role = (next_game_state.color + 1) // 2
    logits, value = policy_value_by_player(local_model(next_obs), role)

    rewards = next_state.rewards[jnp.arange(next_state.rewards.shape[0]), embedding.current_player]
    discounts = jnp.where(jnp.asarray(next_state.terminated), 0.0, -1.0)

    output = mctx.RecurrentFnOutput(
        reward=rewards,
        discount=discounts,
        prior_logits=logits,
        value=value
    )

    return output, next_state


def run_mcts(graph_def: nnx.GraphDef, model_state: nnx.GraphState, env_state,
             rng_key: Array, num_simulations: int, env: Tablut,
             reward_consts: Array = jnp.array([1.0, -1.0, 1.0, -1.0, 0.0, 0.0])) -> mctx.PolicyOutput:
    """
    Run Gumbel MuZero MCTS from the given environment state.

    Handles both batched and unbatched input: if env_state.observation is 3D
    (single state), it is automatically expanded with a batch dimension of 1.

    Returns mctx.PolicyOutput with action, action_weights (improved policy),
    and search statistics.
    """
    if env_state.observation.ndim == 3:
        env_state = jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, axis=0), env_state)

    # Root model
    local_model = nnx.merge(graph_def, model_state)
    role = (env_state.game_state.color + 1) // 2
    root_logits, root_value = policy_value_by_player(local_model(env_state.observation, train=False), role)

    root = mctx.RootFnOutput(
        prior_logits=root_logits,
        value=root_value,
        embedding=env_state
    )

    rec_fn = partial(recurrent_fn, env=env, graph_def=graph_def, reward_consts=reward_consts)

    policy_output = mctx.gumbel_muzero_policy(
        params=model_state,
        rng_key=rng_key,
        root=root,
        recurrent_fn=rec_fn,
        num_simulations=num_simulations,
        invalid_actions=~env_state.legal_action_mask,
        qtransform=mctx.qtransform_completed_by_mix_value
    )

    return policy_output
