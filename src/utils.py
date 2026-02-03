from pathlib import Path

import chess
import chess.pgn
import jax
import numpy as np
import pgx
import jax.numpy as jnp


def create_lookup_tables():
    """
    Create O(1) lookup tables for Pgx action mapping.
    Pgx uses:
    - File-Major Board (a1=0, a2=1, ..., b1=8)
    - 73 Planes: [0-8]: Underpromotion, [9-72]: Queen/Knight
    """
    # Replicating pgx delta generation exactly
    zeros = [0] * 7
    seq = list(range(1, 8))
    rseq = list(range(-7, 0))

    # 0: Down, 1: Up, 2: Left, 3: Right
    # 4: DL, 5: DR, 6: UR, 7: UL
    # 8-15: Knight moves
    dr = rseq + seq + zeros + zeros + rseq + seq + seq[::-1] + rseq[::-1]
    dc = zeros + zeros + rseq + seq + rseq + seq + rseq + seq
    dr += [-1, +1, -2, +2, -1, +1, -2, +2]
    dc += [-2, -2, -1, -1, +2, +2, +1, +1]

    # Map (dy, dx) -> plane_index (relative to 9)
    delta_to_plane = {}
    plane_to_delta = {}
    for i in range(len(dr)):
        delta_to_plane[(dr[i], dc[i])] = 9 + i
        plane_to_delta[9 + i] = (dr[i], dc[i])

    return delta_to_plane, plane_to_delta


DELTA_TO_PLANE, PLANE_TO_DELTA = create_lookup_tables()


# --- 2. Helper: Coordinate Transforms ---
def file_major_to_rank_major(pgx_idx):
    """Convert Pgx (File-Major) index to python-chess (Rank-Major) index"""
    file = pgx_idx // 8
    rank = pgx_idx % 8
    return chess.square(file, rank)


def rank_major_to_file_major(sq):
    """Convert python-chess index to Pgx index"""
    file = chess.square_file(sq)
    rank = chess.square_rank(sq)
    return file * 8 + rank


def get_action_index(board: chess.Board, uci: str):
    """Convert a move from uci to AlphaZero action index"""
    move = board.parse_uci(uci)
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
        plane = DELTA_TO_PLANE.get((dy, dx), -1)

    if plane == -1:
        raise ValueError(f"Could not map move {uci} to a Pgx plane.")

    return pgx_from * 73 + plane


def get_move_from_action(action_idx: int, board: chess.Board):
    plane = action_idx % 73
    pgx_from = action_idx // 73

    promotion_piece = None
    dy, dx = 0, 0

    if plane < 9:
        piece_type = plane // 3
        piece_dir = plane % 3

        promotion_piece = [chess.ROOK, chess.BISHOP, chess.KNIGHT][piece_type]
        dx = [0, 1, -1][piece_dir]  # Up, Right, Left
        dy = 1
    else:
        dy, dx = PLANE_TO_DELTA[plane]

    from_file = pgx_from // 8
    from_rank = pgx_from % 8

    to_file = from_file + dx
    to_rank = from_rank + dy

    is_black = board.turn == chess.BLACK if board else False

    actual_from_rank = (7 - from_rank) if is_black else from_rank
    actual_to_rank = (7 - to_rank) if is_black else to_rank

    from_sq = chess.square(from_file, actual_from_rank)
    to_sq = chess.square(to_file, actual_to_rank)

    if promotion_piece is None and ((is_black and actual_to_rank == 0) or (not is_black and actual_to_rank == 7)) \
            and board and board.piece_type_at(from_sq) == chess.PAWN:
                promotion_piece = chess.QUEEN

    move = chess.Move(from_sq, to_sq, promotion_piece)
    return move.uci()


def run_test_scenario(name: str, moves_san: list):
    """Run a single test chess game scenario"""

    print(f"--- Testing Scenario: {name} ---")

    env = pgx.make("chess")
    state = env.init(jax.random.PRNGKey(0))
    step_fn = jax.jit(env.step)
    py_board = chess.Board()

    for move_san in moves_san:
        move_obj = py_board.parse_san(move_san)
        expected_uci = move_obj.uci()

        print(f"Playing: {move_san} -> {expected_uci} ({'White' if py_board.turn else 'Black'})")

        try:
            action_idx = get_action_index(py_board, move_san)
        except ValueError as e:
            print(f"FAIL: Conversion error - {e}")
            return False

        is_legal = state.legal_action_mask[action_idx]
        if not is_legal:
            print(f"FAIL: Pgx says move {move_san} (Index {action_idx}) is ILLEGAL.")
            return False

        decoded_uci = get_move_from_action(action_idx, py_board)

        if decoded_uci != expected_uci:
            print(f"FAIL: Round trip mismatch!")
            print(f"  Input SAN: {move_san}")
            print(f"  Expected UCI: {expected_uci}")
            print(f"  Decoded UCI:  {decoded_uci}")
            print(f"  Action Index: {action_idx}")
            return False

        state = step_fn(state, action_idx)
        py_board.push(move_obj)

    print("SUCCESS: All moves applied correctly.\n")
    return True


def test_scenarios():
    """Run 6 different scenarios testing different edge cases"""
    moves_1 = ["e4", "e5", "Nf3", "Nc6"]
    moves_2 = ["e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5", "O-O"]
    moves_3 = ["e4", "h6", "e5", "d5", "exd6"]
    moves_5 = ["Na3", "Nh6", "Nc4"]
    moves_6 = ["a4", "b5", "axb5", "c5", "bxc6", "a5", "c7", "a4", "cxb8=Q"]
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

@jax.jit
def reconstruct_observations(obs_bin: jnp.ndarray, obs_scalar: jnp.ndarray):
    """Reconstruct the AlphaZero 8x8x119 tensor from compressed boolean and scalar arrays"""
    batch_size = obs_bin.shape[0]

    move_count = obs_scalar[:, 0].reshape(batch_size, 1, 1, 1)
    rep_count = obs_scalar[:, 1].reshape(batch_size, 1, 1, 1)

    move_plane = jnp.tile(move_count, (1, 8, 8, 1))
    prog_plane = jnp.tile(rep_count, (1, 8, 8, 1))

    bin_part = obs_bin.astype(jnp.float32)

    part_A = bin_part[..., :113]
    part_B = bin_part[..., 113:]

    full_obs = jnp.concatenate([part_A, move_plane, part_B, prog_plane], axis=-1)

    return full_obs


if __name__ == "__main__":
    test_scenarios()