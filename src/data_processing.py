from pathlib import Path

import chess.pgn
import hydra
import jax.random
import jax.numpy as jnp
import numpy as np
import pgx
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.utils import get_action_index, save_compressed_batch, count_games


class DataProcessor:
    def __init__(self, states_per_file, min_half_moves: int, min_elo: int, game_count: int):
        self.root = Path(__file__).resolve().parents[1]
        self.raw_dir = self.root / 'data/raw'
        (self.root / 'data/processed').mkdir(parents=True, exist_ok=True)
        self.processed_dir = self.root / 'data/processed'
        self.pgx_env = pgx.make("chess")
        self.step_fn = jax.jit(self.pgx_env.step)
        self.obs_buffer = []
        self.action_buffer = []
        self.state_idx = 0
        self.batch_counter = 1
        self.states_per_file = states_per_file
        self.min_half_moves = min_half_moves
        self.min_elo = min_elo
        self.game_count = game_count


    def _filter_game_quality(self, game):
        """Filter out all games where both players' ratings are above min_elo
        and the game lasted for at least 20 full moves"""

        headers = game.headers
        white_elo = headers.get('WhiteElo', '?')
        black_elo = headers.get("BlackElo", '?')

        if not (white_elo.isdigit() and black_elo.isdigit()):
            return False

        if int(white_elo) < self.min_elo or int(black_elo) < self.min_elo:
            return False

        ply_count = headers.get("PlyCount")
        if ply_count and ply_count.isdigit():
            return int(ply_count) > self.min_half_moves

        return game.end().ply() > self.min_half_moves


    def _save_batch(self):
        np_obs = np.stack(self.obs_buffer)
        np_action = np.array(self.action_buffer, dtype=np.int32)

        file_name = Path(f'../data/processed/batch_{self.batch_counter:05d}')
        save_compressed_batch(np_obs, np_action, file_name)

        self.obs_buffer = []
        self.action_buffer = []
        self.batch_counter += 1


    def _game_to_samples(self, game: chess.pgn.Game):
        """Process a single game to tensors"""

        board = game.board()
        key = jax.random.PRNGKey(0)
        state = self.pgx_env.init(key)
        for move in game.mainline_moves():
            observation = state.observation
            action_idx = get_action_index(board, move.uci())

            self.obs_buffer.append(observation)
            self.action_buffer.append(action_idx)
            self.state_idx += 1

            board.push(move)
            action_tensor = jnp.array(action_idx, dtype=jnp.int32)
            state = self.step_fn(state, action_tensor)

            if len(self.obs_buffer) >= self.states_per_file:
                self._save_batch()


    def process_data(self):
        """Process all PGN-files in raw data directory"""

        files = list(self.raw_dir.glob('*.pgn'))
        if len(files) == 0:
            print("Raw data directory empty")

        game_counter = 0

        for file in files:
            total_games = self.game_count if self.game_count >= 0 else count_games(file)
            with open(file, 'r', encoding='utf-8', errors='replace') as f:
                with tqdm(total=total_games, desc=f"Processing {file.name}", unit='games') as pbar:
                    while True:
                        try:
                            game = chess.pgn.read_game(f)
                        except ValueError:
                            continue
                        if game is None: break
    
                        if not self._filter_game_quality(game):
                            continue

                        self._game_to_samples(game)
                        game_counter += 1
                        pbar.update(1)
                        if game_counter >= self.game_count:
                            break
            if game_counter >= self.game_count:
                break
        if (len(self.obs_buffer)) > 0:
            self._save_batch()


@hydra.main(version_base=None, config_path='../configs', config_name='config')
def main(cfg: DictConfig):
    print(f'Processing PGN-s with {cfg.preprocessing.states_per_file} states per file, '
          f'{cfg.preprocessing.min_elo} as min elo and {cfg.preprocessing.min_half_moves / 2} as min full moves')
    processor = DataProcessor(
        states_per_file=cfg.preprocessing.states_per_file,
        min_elo=cfg.preprocessing.min_elo,
        min_half_moves=cfg.preprocessing.min_half_moves,
        game_count=cfg.preprocessing.game_count)
    processor.process_data()


if __name__ == "__main__":
    main()