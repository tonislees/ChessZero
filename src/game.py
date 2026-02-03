import random
from pathlib import Path

import chess
import hydra
import jax
import pgx
from flax import nnx
from omegaconf import DictConfig
import orbax.checkpoint as ocp
from stockfish import Stockfish

from src.mcts import run_mcts
from src.model import ChessZeroNet
from src.utils import get_action_index, get_move_from_action


class Game:
    def __init__(self, cfg: DictConfig):
        root_dir = self.root = Path(__file__).resolve().parents[1]
        self.model_dir = root_dir / 'models'
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir = self.model_dir / 'checkpoints'
        self.checkpointer = ocp.StandardCheckpointer()
        self.plot_dir = root_dir / 'plots'
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        self.num_epochs = cfg.train.num_epochs
        self.num_simulations = cfg.mcts.simulations
        self.seed = cfg.train.seed
        self.rngs: nnx.Rngs = nnx.Rngs(self.seed)
        self.model: nnx.Module = ChessZeroNet(depth=cfg.model.depth, filter_count=cfg.model.filter_count,
                                              rngs=self.rngs)

        if self.checkpoints_dir.exists() and any(self.checkpoints_dir.iterdir()):
            graphdef, abstract_state = nnx.split(self.model)
            restored_state = self.checkpointer.restore(self.checkpoints_dir, abstract_state)
            self.model = nnx.merge(graphdef, restored_state)
            print(f"Restored model from {self.checkpoints_dir}")

        self.env = pgx.make('chess')
        key_env = jax.random.PRNGKey(self.seed + 1)
        self.env_state = jax.jit(jax.vmap(self.env.init))(
            jax.random.split(key_env, cfg.train.batch_size)
        )
        self.testing_env = pgx.make('chess')
        self.testing_env_state = jax.jit(self.testing_env.init)(key_env)
        self.board = chess.Board()
        self.start_player = cfg.interface.start_player
        self.step_fn = jax.jit(self.env.step)
        self.stockfish = Stockfish()


    def _make_move(self, move_uci: str):
        move = chess.Board.parse_uci(self.board, move_uci)

        action = get_action_index(self.board, move_uci)
        self.board.push(move)
        self.testing_env_state = self.step_fn(self.testing_env_state, action)

        self.stockfish.make_moves_from_current_position([move_uci])


    def _find_move_mcts(self):
        rng_key = self.rngs.split()
        mcts_output = run_mcts(self.model, self.testing_env_state, rng_key, self.num_simulations, self.testing_env)
        action = mcts_output.action[0].item()
        return action


    def play_single_game_with_stockfish(self, skill_level: int):
        self.board = chess.Board()
        key_env = jax.random.PRNGKey(self.seed + random.randint(0, 1000000))
        self.testing_env_state = jax.jit(self.testing_env.init)(key_env)

        beginner = random.choice([1, 0])
        self.stockfish.make_moves_from_start()
        self.stockfish.set_skill_level(skill_level)
        self.model.eval()

        if beginner == 1:
            best_move = self.stockfish.get_best_move_time(100)
            self._make_move(best_move)

        while not self.board.is_game_over() :
            action = self._find_move_mcts()
            uci_move = get_move_from_action(action, self.board)
            self._make_move(uci_move)

            if self.board.is_game_over():
                break

            best_move = self.stockfish.get_best_move_time(100)
            self._make_move(best_move)

        outcome = self.board.outcome().winner
        return 1 if outcome is None else 0 if outcome and beginner == 0 else 2


    def play_games_with_stockfish(self, num_games: int, skill_level: int):
        stats = [0, 0, 0] # wins, draws, losses
        for i in range(num_games):
            result = self.play_single_game_with_stockfish(skill_level)
            stats[result] += 1
        print(f'wins: {stats[0]}\ndraws: {stats[1]}\nlosses: {stats[2]}\n')


@hydra.main(version_base=None, config_path='..', config_name='config')
def main(cfg: DictConfig):
    game = Game(cfg)
    game.play_games_with_stockfish(10, 1)


if __name__ == '__main__':
    main()