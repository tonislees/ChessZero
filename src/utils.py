from pathlib import Path

import chess
import chess.pgn
import jax
import numpy as np
import pgx
import jax.numpy as jnp


def create_deltas():
    """Create move deltas for cleaner implementation"""
    zeros = [0] * 7
    sequence = list(range(1, 8))
    reverse_sequence = list(range(-7, 0))

    # Concatenate vectors to match the loop order 9..72
    # Order: down, up, left, right, down-left, down-right, up-right, up-left, knight, and knight
    dr = reverse_sequence + sequence + zeros            + zeros    + reverse_sequence + sequence + sequence[::-1]   + reverse_sequence[::-1]
    dc = zeros            + zeros    + reverse_sequence + sequence + reverse_sequence + sequence + reverse_sequence + sequence

    # Add Knight moves (Planes 65-72)
    dr += [-1, +1, -2, +2, -1, +1, -2, +2]
    dc += [-2, -2, -1, -1, +2, +2, +1, +1]
    return dr, dc


DR, DC = create_deltas()


def get_action_index(board: chess.Board, uci: str):
    """Convert a move from uci to AlphaZero action index"""
    move = board.parse_san(uci)
    from_sq = move.from_square
    to_sq = move.to_square

    # Pgx uses File-Major: a1=0, a2=1, ..., a8=7, b1=8, ...
    def to_pgx_idx(sq, flip=False):
        file = chess.square_file(sq)
        rank = chess.square_rank(sq)
        if flip:
            rank = 7 - rank
        return file * 8 + rank

    flip = (board.turn == chess.BLACK)
    pgx_from = to_pgx_idx(from_sq, flip)
    pgx_to = to_pgx_idx(to_sq, flip)

    # In Pgx File-Major, rank is row (0-7), file is col (0-7)
    # dx is file delta, dy is rank delta
    dx = (pgx_to // 8) - (pgx_from // 8)  # Delta File
    dy = (pgx_to % 8) - (pgx_from % 8)  # Delta Rank

    plane = -1

    if move.promotion and move.promotion != chess.QUEEN:
        # plane // 3 == 0: rook, 1: bishop, 2: knight
        # plane % 3 == 0: up (dy=1, dx=0), 1: right (dy=1, dx=1), 2: left (dy=1, dx=-1)
        direction = -1
        if dy == 1:
            if dx == 0: direction = 0
            elif dx == 1: direction = 1
            elif dx == -1: direction = 2

        piece_offset = -1
        if move.promotion == chess.ROOK: piece_offset = 0
        elif move.promotion == chess.BISHOP: piece_offset = 1
        elif move.promotion == chess.KNIGHT: piece_offset = 2

        if direction != -1 and piece_offset != -1:
            plane = (piece_offset * 3) + direction

    else:
        for i in range(64):
            if dy == DR[i] and dx == DC[i]:
                plane = 9 + i
                break

    if plane == -1:
        raise ValueError(f"Could not map move {uci} to a Pgx plane.")

    return pgx_from * 73 + plane


def run_test_scenario(name: str, moves_san: list):
    """Run a single test chess game scenario"""

    print(f"--- Testing Scenario: {name} ---")

    env = pgx.make("chess")
    state = env.init(jax.random.PRNGKey(0))
    step_fn = jax.jit(env.step)
    py_board = chess.Board()

    for move_san in moves_san:
        print(f"Playing: {move_san} ({'White' if py_board.turn else 'Black'})")

        try:
            action_idx = get_action_index(py_board, move_san)
        except ValueError as e:
            print(f"FAIL: Conversion error - {e}")
            return False

        is_legal = state.legal_action_mask[action_idx]
        if not is_legal:
            print(f"FAIL: Pgx says move {move_san} (Index {action_idx}) is ILLEGAL.")
            print(f"Legal indices: {jnp.flatnonzero(state.legal_action_mask)}")
            return False

        state = step_fn(state, action_idx)
        py_board.push_san(move_san)

    print("SUCCESS: All moves applied correctly.\n")
    return True


def test_scenarios():
    """Run 6 different scenarios testing different edge cases"""
    # Scenario 1: Basic Opening (White and Black)
    moves_1 = ["e4", "e5", "Nf3", "Nc6"]

    # Scenario 2: Castling
    moves_2 = ["e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5", "O-O"]

    # Scenario 3: En Passant
    moves_3 = ["e4", "h6", "e5", "d5", "exd6"]

    # Scenario 4: Knight Jumps
    moves_5 = ["Na3", "Nh6", "Nc4"]

    # Scenario 5: Promotion
    moves_6 = ["a4", "b5", "axb5", "c5", "bxc6", "a5", "c7", "a4", "cxb8=Q"]

    # Scenario 6: Underpromotion
    moves_7 = ["g4", "h5", "gxh5", "f5", "h6", "e5", "hxg7", "e4", "gxh8=N"]

    tests = [
        ("Basic Opening", moves_1),
        ("Castling", moves_2),
        ("En Passant", moves_3),
        ("Knight Hopping", moves_5),
        ("Promotion", moves_6),
        ("Underpromotion", moves_7)
    ]

    all_passed = True
    for name, moves in tests:
        if not run_test_scenario(name, moves):
            all_passed = False
            break

    if all_passed:
        print("\nAll Scenarios PASSED!")
    else:
        print("\nSome tests FAILED.")


def count_games(self, file_path):
    """Quickly count games by looking for the start of PGN headers"""

    count = 0
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            if line.startswith('[Event '):
                count += 1
    return count


def save_compressed_batch(obs_batch: np.ndarray, action_batch: np.ndarray, filename: Path):
    """Save a batch of processed data as compressed boolean and scalar arrays"""

    moves_scalar = obs_batch[:, 0, 0, 113]
    no_prog_scalar = obs_batch[:, 0, 0, 118]
    scalars = np.stack([moves_scalar, no_prog_scalar], axis=1).astype(np.float16)

    # We keep 0-113, skip 113, keep 114-118, skip 118
    binary_part = np.concatenate([
        obs_batch[..., :113],
        obs_batch[..., 114:118]
    ], axis=-1)

    binary_packed = binary_part.astype(np.bool_)

    np.savez_compressed(
        filename,
        obs_bin=binary_packed,
        obs_scalar=scalars,
        policy=action_batch.astype(np.int16),
    )


def reconstruct_observations(obs_bin: np.ndarray, obs_scalar: np.ndarray):
    """Reconstruct the AlphaZero 8x8x119 tensor from compressed boolean and scalar arrays"""
    batch_size = obs_bin.shape[0]

    move_count = obs_scalar[:, 0].reshape(batch_size, 1, 1, 1)
    rep_count = obs_scalar[:, 1].reshape(batch_size, 1, 1, 1)

    move_plane = np.tile(move_count, (1, 8, 8, 1))
    prog_plane = np.tile(rep_count, (1, 8, 8, 1))

    bin_part = obs_bin.astype(np.float32)

    part_A = bin_part[..., :113]
    part_B = bin_part[..., 113:]

    full_obs = np.concatenate([part_A, move_plane, part_B, prog_plane], axis=-1)

    return full_obs


if __name__ == "__main__":
    test_scenarios()