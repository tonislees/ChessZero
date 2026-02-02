from functools import partial

import chex
import jax
import mctx
import pgx
from flax import nnx
import jax.numpy as jnp


def recurrent_fn(params: chex.ArrayTree, rng_key: jax.Array, action: jax.Array,
                 embedding: pgx.State, env: pgx.Env, graphdef: nnx.GraphDef):
    next_state: pgx.State = jax.vmap(env.step)(embedding, action)

    model = nnx.merge(graphdef, params)
    logits, value = model(next_state.observation)

    rewards = next_state.rewards[jnp.arange(next_state.rewards.shape[0]), embedding.current_player]
    discounts = jnp.where(next_state.terminated, 0.0, 1.0)

    output = mctx.RecurrentFnOutput(
        reward=rewards,
        discount=discounts,
        prior_logits=logits,
        value=value
    )

    return output, next_state


def run_mcts(model: nnx.Module, env_state, rng_key: jax.Array, num_simulations: int, env: pgx.Env):
    graphdef, state = nnx.split(model)
    root_logits, root_value = model(env_state.observation, train=False)

    root = mctx.RootFnOutput(
        prior_logits=root_logits,
        value=root_value,
        embedding=env_state
    )

    rec_fn = partial(recurrent_fn, env=env, graphdef=graphdef)

    policy_output = mctx.gumbel_muzero_policy(
        params=state,
        rng_key=rng_key,
        root=root,
        recurrent_fn=rec_fn,
        num_simulations=num_simulations,
        invalid_actions=~env_state.legal_action_mask,
        qtransform=mctx.qtransform_completed_by_mix_value
    )

    return policy_output
