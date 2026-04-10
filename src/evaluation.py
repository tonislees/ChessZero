import random
import shutil
from functools import partial
from pathlib import Path

import jax
from flax import nnx
from jax import numpy as jnp, lax, Array
from omegaconf import DictConfig
import orbax.checkpoint as ocp
from tqdm import tqdm

from .tablut.tablut import Tablut, State
from .mcts import run_mcts
from .model import TablutZeroNet
from .utils import run_bayeselo

"""
Evaluation and BayesElo rating for TablutZero checkpoints.

The Evaluator maintains a pool of past model checkpoints (CPU-resident state dicts).
On each evaluation call, the current model plays against a subset of pool members
from both sides (attacker and defender). Results are appended to a PGN file and
processed by the BayesElo program to compute ratings.
"""

class Evaluator:
    def __init__(self, cfg: DictConfig, dirs: dict[str, Path], rngs: nnx.Rngs,
                 model: nnx.Module, checkpointer: ocp.StandardCheckpointer, env: Tablut):
        self.cfg = cfg
        self.dirs = dirs
        self.rngs = rngs
        self.model = model
        self.checkpointer = checkpointer
        self.env = env
        self.eval_pool = self._load_eval_pool(cfg.train.load_checkpoint)
        self._add_to_eval_pool(iteration=0)

    def _init_eval_state(self, is_starter: bool, batch_size: int) -> State:
        key_env = jax.random.split(self.rngs.split(), batch_size)
        state = jax.jit(jax.vmap(self.env.init))(key_env)

        if is_starter:
            player_order = jnp.tile(jnp.array([0, 1]), (batch_size, 1))
        else:
            player_order = jnp.tile(jnp.array([1, 0]), (batch_size, 1))

        return state.replace(
            _player_order=player_order,
            current_player=player_order[:, 0]
        )

    def evaluate_model(self, iteration: int) -> None:
        current_model = f"iter_{iteration}"
        num_opponents = min(4, len(self.eval_pool))
        opponents = self._load_random_opponents(num_opponents)
        eval_sims = self.cfg.mcts.simulations // 2

        games_per_opponent = self.cfg.train.eval_batch_size // num_opponents

        self.model.eval()

        global _eval_pbar
        _eval_pbar = tqdm(
            total=None,
            desc='Evaluation',
            mininterval=self.cfg.train.tqdm_interval,
            ncols=100,
            unit='steps'
        )

        all_match_data = []
        opponent_summaries = []

        for opponent_name, opponent_model in opponents.items():
            p0_state = self._init_eval_state(is_starter=True, batch_size=games_per_opponent)
            p1_state = self._init_eval_state(is_starter=False, batch_size=games_per_opponent)

            rewards_p0 = jax.device_get(evaluate(
                model_A=self.model,
                model_B=opponent_model,
                state=p0_state,
                rng_key=self.rngs.default(),
                num_simulations=eval_sims,
                env=self.env
            ))
            rewards_p1 = jax.device_get(evaluate(
                model_A=self.model,
                model_B=opponent_model,
                state=p1_state,
                rng_key=self.rngs.default(),
                num_simulations=eval_sims,
                env=self.env
            ))

            match_data, summary = self._get_eval_metrics(
                rewards_p0, rewards_p1, opponent_name, current_model
            )
            all_match_data.extend(match_data)
            opponent_summaries.append(summary)

        _eval_pbar.close()

        self._log_eval_results(current_model, opponent_summaries)

        self._generate_minimal_pgn(all_match_data)
        ratings = run_bayeselo(self.dirs['pgn'], self.dirs['bayeselo'])

        elo = ratings.get(current_model, 0)
        last_elo = ratings.get(f'iter_{iteration - 1}', 0)
        elo_delta = elo - last_elo

        print(f"  {'─' * 48}")
        print(f"  Elo: {elo - ratings.get('iter_0', 0):+d}  (Δ {elo_delta:+d})")
        print(f"  {'─' * 48}")

        self._add_to_eval_pool(iteration)

    @staticmethod
    def _get_eval_metrics(rewards_p0: jax.Array, rewards_p1: jax.Array,
                          opponent: str, current_model: str) -> tuple[list, dict]:
        match_data = []

        p0_stats = {"wins": 0, "losses": 0, "draws": 0}
        p1_stats = {"wins": 0, "losses": 0, "draws": 0}

        for i in range(len(rewards_p0)):
            rew = rewards_p0[i, 0]
            match_data.append((current_model, opponent, rew))
            if rew == 1:    p0_stats["wins"] += 1
            elif rew == -1: p0_stats["losses"] += 1
            else:           p0_stats["draws"] += 1

        for i in range(len(rewards_p1)):
            rew = rewards_p1[i, 0]
            match_data.append((opponent, current_model, -rew))
            if rew == 1:    p1_stats["wins"] += 1
            elif rew == -1: p1_stats["losses"] += 1
            else:           p1_stats["draws"] += 1

        total_games = len(rewards_p0) + len(rewards_p1)
        total_wins = p0_stats["wins"] + p1_stats["wins"]
        total_losses = p0_stats["losses"] + p1_stats["losses"]
        total_draws = p0_stats["draws"] + p1_stats["draws"]
        score = (total_wins + 0.5 * total_draws) / total_games if total_games > 0 else 0.0

        summary = {
            "opponent": opponent,
            "p0": p0_stats,   # current model as attacker
            "p1": p1_stats,   # current model as defender
            "total": {"wins": total_wins, "losses": total_losses,
                      "draws": total_draws, "games": total_games},
            "score": score,
        }

        return match_data, summary

    @staticmethod
    def _log_eval_results(current_model: str, summaries: list[dict]) -> None:
        total_wins = total_losses = total_draws = total_games = 0
        att_wins = att_losses = att_draws = 0
        def_wins = def_losses = def_draws = 0

        print(f"\n{'  ═' * 17}")
        print(f"  Evaluation: {current_model}")
        print(f"{'  ═' * 17}")
        print(f"  {'Opponent':<18} {'As Attacker':^22} {'As Defender':^22} {'Score':>6}")
        print(f"  {'─' * 72}")

        for s in summaries:
            p0, p1 = s['p0'], s['p1']
            opp_short = s['opponent'][-14:] if len(s['opponent']) > 14 else s['opponent']

            att_str = f"{p0['wins']}W {p0['losses']}L {p0['draws']}D"
            def_str = f"{p1['wins']}W {p1['losses']}L {p1['draws']}D"
            score_str = f"{s['score']:.1%}"

            print(f"  {opp_short:<18} {att_str:^22} {def_str:^22} {score_str:>6}")

            total_wins   += s['total']['wins']
            total_losses += s['total']['losses']
            total_draws  += s['total']['draws']
            total_games  += s['total']['games']

            att_wins   += p0['wins'];   att_losses += p0['losses'];   att_draws += p0['draws']
            def_wins   += p1['wins'];   def_losses += p1['losses'];   def_draws += p1['draws']

        overall_score = (total_wins + 0.5 * total_draws) / total_games if total_games > 0 else 0.0

        print(f"  {'─' * 72}")
        print(f"  {'TOTAL':<18} "
              f"{f'{att_wins}W {att_losses}L {att_draws}D':^22} "
              f"{f'{def_wins}W {def_losses}L {def_draws}D':^22} "
              f"{overall_score:>6.1%}")

        # Attacker/Defender balance insight
        att_total = att_wins + att_losses + att_draws
        def_total = def_wins + def_losses + def_draws
        att_winrate = att_wins / att_total if att_total > 0 else 0.0
        def_winrate = def_wins / def_total if def_total > 0 else 0.0

        print(f"\n  Attacker win rate: {att_winrate:.1%}   Defender win rate: {def_winrate:.1%}")
        print()

    def _generate_minimal_pgn(self, match_data) -> None:
        """
        Appends match results to the PGN file for BayesElo processing.

        Uses dummy moves (1. d4 d5) since BayesElo only needs player names and
        results, not actual move sequences. Appends rather than overwrites so
        results accumulate across iterations.
        """
        with open(self.dirs['pgn'], "a") as f:
            for attacker, defender, reward in match_data:
                if reward == 1:     result_str = "1-0"
                elif reward == -1:  result_str = "0-1"
                else:               result_str = "1/2-1/2"

                f.write(f'[White "{attacker}"]\n')
                f.write(f'[Black "{defender}"]\n')
                f.write(f'[Result "{result_str}"]\n')
                f.write("1. d4 d5\n\n")

    def _load_eval_pool(self, load_checkpoint: bool) -> dict[str, TablutZeroNet]:
        pool = {}
        dir_path = self.dirs['eval_pool']
        if not load_checkpoint:
            shutil.rmtree(dir_path, ignore_errors=True)
            dir_path.mkdir(parents=True, exist_ok=True)
            return pool

        _, abstract_state = nnx.split(
            TablutZeroNet(depth=self.cfg.model.depth,
                          filter_count=self.cfg.model.filter_count, rngs=self.rngs)
        )
        dirs = [d for d in dir_path.iterdir() if d.is_dir()]

        for ckpt_dir in dirs[-self.cfg.train.max_eval_pool:]:
            restored_state = self.checkpointer.restore(ckpt_dir.resolve())
            pool[ckpt_dir.name] = jax.device_get(restored_state)

        print(f"  Loaded {len(pool)} models into the evaluation pool.")
        return pool

    def _add_to_eval_pool(self, iteration: int) -> None:
        _, current_state = nnx.split(self.model)
        model_name = f"iter_{iteration}"

        if len(self.eval_pool) >= self.cfg.train.max_eval_pool:
            protected = {'iter_0'}
            sorted_names = sorted(
                self.eval_pool.keys(),
                key=lambda x: int(x.split('_')[1])
            )
            if sorted_names:
                protected.add(sorted_names[-1])

            candidates = [n for n in self.eval_pool.keys() if n not in protected]
            if candidates:
                victim_name = random.choice(candidates)
                del self.eval_pool[victim_name]
                victim_path = self.dirs['eval_pool'] / victim_name
                if victim_path.exists():
                    shutil.rmtree(victim_path)

        self.eval_pool[model_name] = jax.device_get(current_state)

    def save_eval_pool(self) -> None:
        if not hasattr(self, 'eval_pool') or not self.eval_pool:
            return

        for ckpt_dir in self.dirs['eval_pool'].iterdir():
            if ckpt_dir.is_dir() and ckpt_dir.name not in self.eval_pool:
                shutil.rmtree(ckpt_dir, ignore_errors=True)

        for model_name, cpu_state in self.eval_pool.items():
            save_path = self.dirs['eval_pool'] / model_name
            if not save_path.exists():
                self.checkpointer.save(save_path.resolve(), cpu_state)
        self.checkpointer.wait_until_finished()

    def _load_random_opponents(self, n: int) -> dict[str, nnx.Module]:
        """
        Selects n opponents from the eval pool with an anchor strategy.

        Always includes the two most recent checkpoints (to track incremental
        progress), then fills remaining slots randomly from the rest of the pool.
        Returns a dict mapping model names to GPU-resident nnx.Module instances.
        """
        pool_names = list(self.eval_pool.keys())
        if not pool_names:
            return {}

        sorted_names = sorted(pool_names, key=lambda x: int(x.split('_')[1]))

        anchors = sorted_names[-2:]

        remaining = [name for name in pool_names if name not in anchors]
        random_picks = random.sample(remaining, min(n - len(anchors), len(remaining)))

        selected = (anchors + random_picks)[:n]

        result = {}
        graph_def, _ = nnx.split(self.model)
        for name in selected:
            gpu_state = jax.device_put(self.eval_pool[name])
            opponent_model = nnx.merge(graph_def, gpu_state)
            opponent_model.eval()
            result[name] = opponent_model

        return result


@partial(nnx.jit, static_argnames=('num_simulations', 'env'))
def evaluate(model_A: nnx.Module, model_B: nnx.Module, state: State,
             rng_key: Array, num_simulations: int, env: Tablut) -> Array:
    """
    Play out batched games between model_A and model_B to completion.

    Uses a lax.while_loop that runs until all games in the batch have terminated.
    A termination mask tracks which games have finished; their rewards are frozen
    at the terminal value while remaining games continue.

    model_A plays when current_player == 0, model_B when current_player == 1.
    Returns final rewards with shape (batch, 2).
    """
    graph_def_A, model_A_state = nnx.split(model_A)
    graph_def_B, model_B_state = nnx.split(model_B)

    def step_fn(loop_vars):
        step_state, key, t_mask, rewards = loop_vars

        is_terminal = step_state.terminated
        should_update = is_terminal & ~t_mask

        next_rewards = jnp.where(should_update[:, None], step_state.rewards, rewards)
        next_mask = t_mask | is_terminal

        key_A, key_B, next_key = jax.random.split(key, 3)
        is_p0 = (step_state.current_player[0] == 0)

        action = lax.cond(
            is_p0,
            lambda: run_mcts(graph_def_A, model_A_state, step_state, key_A, num_simulations, env).action,
            lambda: run_mcts(graph_def_B, model_B_state, step_state, key_B, num_simulations, env).action
        )

        def update_pbar(_):
            global _eval_pbar
            if '_eval_pbar' in globals():
                _eval_pbar.update(1)

        jax.debug.callback(update_pbar, key)

        return jax.vmap(env.step)(step_state, action), next_key, next_mask, next_rewards

    def cond_fn(loop_vars):
        _, _, t_mask, _ = loop_vars
        return ~jnp.all(t_mask)

    termination_mask = jnp.zeros_like(state.terminated, dtype=jnp.bool_)
    init_rewards = jnp.zeros_like(state.rewards)
    _, _, _, final_rewards = lax.while_loop(
        cond_fn, step_fn, (state, rng_key, termination_mask, init_rewards)
    )

    return final_rewards