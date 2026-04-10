import jax
import jax.numpy as jnp

import pgx.core as core
from .tablut_jax import Game, GameState, INIT_LEGAL_ACTION_MASK, _flip
from pgx._src.struct import dataclass
from pgx._src.types import Array, PRNGKey

"""
Pgx-compatible environment wrapper for the Tablut JAX game engine.

Bridges the raw GameState from tablut_jax.py into pgx's State/Env interface,
handling player ordering, observation generation, and reward mapping. The
_player_order field maps internal roles (attacker=0, defender=1) to the two
external player slots, allowing randomized side assignment.
"""


@dataclass
class State(core.State):
    current_player: Array = jnp.int32(0)
    rewards: Array = jnp.float32([0.0, 0.0])
    terminated: Array = jnp.bool_(False)
    truncated: Array = jnp.bool_(False)
    legal_action_mask: Array = INIT_LEGAL_ACTION_MASK
    observation: Array = jnp.zeros((9, 9, 43), dtype=jnp.float32)
    _step_count: Array = jnp.int32(0)
    _player_order: Array = jnp.int32([0, 1])  # [0, 1] or [1, 0]
    game_state: GameState = GameState()

    @property
    def env_id(self) -> core.EnvId:
        return "tablut"


class Tablut(core.Env):
    def __init__(self):
        super().__init__()
        self.game = Game()

    def _init(self, key: PRNGKey) -> State:
        game_state = GameState()
        _player_order = jnp.array([[0, 1], [1, 0]])[jax.random.bernoulli(key).astype(jnp.int32)]
        state = State(  # type: ignore
            current_player=_player_order[(game_state.color + 1) // 2],
            _player_order=_player_order,
            game_state=game_state,
        )
        return state

    def _step(self, state: State, action: Array, key: PRNGKey) -> State:
        del key
        assert isinstance(state, State)
        game_state = self.game.step(state.game_state, action)
        state = state.replace(  # type: ignore
            game_state=game_state,
            legal_action_mask=game_state.legal_action_mask,
            terminated=self.game.is_terminal(game_state),
            rewards=self.game.rewards(game_state)[state._player_order],
            current_player=state._player_order[(game_state.color + 1) // 2],
        )
        return state  # type: ignore

    def step(self, state: State, action: Array, key: PRNGKey | None = None) -> State:
        return super().step(state, action, key)  # type: ignore[return-value]

    def _observe(self, state: State, player_id: Array) -> Array:
        assert isinstance(state, State)
        game_state = jax.lax.cond(state.current_player == player_id, lambda: state.game_state, lambda: _flip(state.game_state))
        return self.game.observe(game_state)

    @property
    def id(self):
        return "tablut"

    @property
    def version(self) -> str:
        return "v0"

    @property
    def num_players(self) -> int:
        return 2
