from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, lax

"""
Pure-JAX Tablut game engine on a 9×9 board.

Board representation:
    -1 = attacker (taflman), 1 = defender (taflman), 2 = king, 0 = empty.
    The board is stored from the current player's perspective: positive pieces are
    friendly, negative pieces are enemy. _flip() negates the board on turn change.

Action encoding (AlphaZero-style):
    Each action is a flat index into BOARD_SIZE × ACTION_PLANES (81 × 32 = 2592).
    ACTION_PLANES = 4 directions × (BOARD_EDGE - 1) max distance = 32.
    Plane layout: [right 1..8, up 1..8, down 1..8, left 1..8].

Coordinate system:
    Square index = row * BOARD_EDGE + col, with row 0 at the bottom (rank 1).

Precomputed arrays (FROM_PLANE, TO_PLANE, BETWEEN, ATTACK_PAIR, etc.) are computed
once at import time as NumPy arrays, then converted to JAX arrays for use in jitted
functions.
"""

BOARD_EDGE = 9
BOARD_SIZE = BOARD_EDGE * BOARD_EDGE # 81
THRONE = BOARD_SIZE // 2 # 40
MAX_SHIELD_WALL_PARTNERS = BOARD_EDGE - 4

ACTION_PLANES = 4 * (BOARD_EDGE - 1)

EMPTY, TAFLMAN, KING = tuple(range(3)) # 0, 1, 2
NUM_ATTACKERS = (BOARD_EDGE - 5) * 4
MAX_TERMINATION_STEPS = 512
MAX_HALF_MOVE_COUNT: float = 200.0

INIT_BOARD = jnp.int32([
    0,  0,  0, -1, -1, -1,  0,  0,  0,
    0,  0,  0,  0, -1,  0,  0,  0,  0,
    0,  0,  0,  0,  1,  0,  0,  0,  0,
   -1,  0,  0,  0,  1,  0,  0,  0, -1,
   -1, -1,  1,  1,  2,  1,  1, -1, -1,
   -1,  0,  0,  0,  1,  0,  0,  0, -1,
    0,  0,  0,  0,  1,  0,  0,  0,  0,
    0,  0,  0,  0, -1,  0,  0,  0,  0,
    0,  0,  0, -1, -1, -1,  0,  0,  0,
])

ZERO_INIT_ACTION_MASK = jnp.zeros(BOARD_SIZE * ACTION_PLANES, dtype=jnp.bool_)

#  9  72  73  74  75  76  77  78  79  80
#  8  63  64  65  66  67  68  69  70  71
#  7  54  55  56  57  58  59  60  61  62
#  6  45  46  47  48  49  50  51  52  53
#  5  36  37  38  39  40  41  42  43  44
#  4  27  28  29  30  31  32  33  34  35
#  3  18  19  20  21  22  23  24  25  26
#  2   9  10  11  12  13  14  15  16  17
#  1   0   1   2   3   4   5   6   7   8
#      a   b   c   d   e   f   g   h   i

# Action: AlphaZero style labels (BOARD_SIZE x ACTION_PLANES = 81 x 32 = 2592)
#                             15
#                             14
#                             13
#                             12
#                             11
#                             10
#                              9
#                              8
#  31 30 29 28 27 26 25 24  X  0  1  2  3  4  5  6  7
#                             16
#                             17
#                             18
#                             19
#                             20
#                             21
#                             22
#                             23


def calc_hostile_squares() -> tuple[np.ndarray, np.ndarray]:
    """Calculate the hostile squares: corners and throne."""
    bottom_left = 0
    bottom_right = BOARD_EDGE
    top_left = BOARD_SIZE - BOARD_EDGE
    top_right = BOARD_SIZE

    corners = [bottom_left, bottom_right - 1, top_left, top_right - 1]

    hostile_squares_mask = np.zeros(BOARD_SIZE, dtype=np.bool)
    hostile_squares_mask[corners] = True
    corners_mask = hostile_squares_mask.copy()
    hostile_squares_mask[THRONE] = True
    return hostile_squares_mask, corners_mask


HOSTILE_SQUARES_MASK, CORNERS_MASK = calc_hostile_squares()


def calc_rows_columns() -> tuple[np.ndarray, np.ndarray]:
    """Calculate matrices for row and column indices."""
    rows = np.zeros((BOARD_EDGE, BOARD_EDGE), dtype=np.int32)
    columns = np.zeros((BOARD_EDGE, BOARD_EDGE), dtype=np.int32)

    for i in range(BOARD_SIZE):
        row = i // BOARD_EDGE
        column = i % BOARD_EDGE
        row_idx = column
        column_idx = row

        rows[row, row_idx] = i
        columns[column, column_idx] = i
    return rows, columns


ROWS, COLUMNS = calc_rows_columns()


def calc_edges() -> tuple[np.ndarray, np.ndarray]:
    """Calculate arrays for edge indices, possible shield wall partners,
    and neighbor indices for each shield wall inner neighbor"""
    top = ROWS[BOARD_EDGE - 1]
    bottom = ROWS[0]
    left = COLUMNS[0]
    right = COLUMNS[BOARD_EDGE - 1]

    edges = np.zeros(BOARD_SIZE, dtype=np.bool)
    all_edge_indices = np.unique(np.concatenate([top, bottom, left, right]))
    edges[all_edge_indices] = True

    inner_neighbor = -np.ones(BOARD_SIZE, dtype=np.int32)
    inner_neighbor[top] = top - BOARD_EDGE
    inner_neighbor[bottom] = bottom + BOARD_EDGE
    inner_neighbor[left] = left + 1
    inner_neighbor[right] = right - 1

    return edges, inner_neighbor


EDGES, INNER_NEIGHBOR = calc_edges()


def calc_action_arrays() -> tuple[np.ndarray, np.ndarray]:
    """Calculate mappings for conversions between Action objects and action indices."""

    from_plane = -np.ones((BOARD_SIZE, ACTION_PLANES), dtype=np.int32)
    to_plane = -np.ones((BOARD_SIZE, BOARD_SIZE), dtype=np.int32)

    max_action_length = BOARD_EDGE - 1
    zeros = [0] * max_action_length

    delta_col = list(range(1, BOARD_EDGE)) + zeros + zeros + list(range(-1, -BOARD_EDGE, -1))
    delta_row = zeros + list(range(1, BOARD_EDGE)) + list(range(-1, -BOARD_EDGE, -1)) + zeros

    # Create mappings for square -> action
    # move with action plane from square from_
    for from_sq in range(BOARD_SIZE):
        from_row = from_sq // BOARD_EDGE
        from_col = from_sq % BOARD_EDGE
        for action in range(ACTION_PLANES):
            to_row = from_row + delta_row[action]
            to_col = from_col + delta_col[action]

            # Check if the action fits on the board
            if 0 <= to_row < BOARD_EDGE and 0 <= to_col < BOARD_EDGE:
                to = to_row * BOARD_EDGE + to_col
                from_plane[from_sq, action] = to
                to_plane[from_sq, to] = action
    return from_plane, to_plane


FROM_PLANE, TO_PLANE = calc_action_arrays()


def calc_capture_arrays() -> tuple[np.ndarray, np.ndarray]:
    """Calculate the arrays representing attack pairs and neighboring squares."""

    attack_pair = -np.ones((BOARD_SIZE, 4), dtype=np.int32)
    neighbors = -np.ones((BOARD_SIZE, 4), dtype=np.int32)
    for to_sq in range(BOARD_SIZE):
        row, col = to_sq // BOARD_EDGE, to_sq % BOARD_EDGE
        #UP
        if row - 1 >= 0:
            neighbors[to_sq, 0] = (row - 1) * BOARD_EDGE + col

        if row - 2 >= 0:
            attack_pair[to_sq, 0] = (row - 2) * BOARD_EDGE + col
        #RIGHT
        if col + 1 < BOARD_EDGE:
            neighbors[to_sq, 1] = row * BOARD_EDGE + (col + 1)

        if col + 2 < BOARD_EDGE:
            attack_pair[to_sq, 1] = row * BOARD_EDGE + (col + 2)
        #DOWN
        if row + 1 < BOARD_EDGE:
            neighbors[to_sq, 2] = (row + 1) * BOARD_EDGE + col

        if row + 2 < BOARD_EDGE:
            attack_pair[to_sq, 2] = (row + 2) * BOARD_EDGE + col
        #LEFT
        if col - 1 >= 0:
            neighbors[to_sq, 3] = row * BOARD_EDGE + (col - 1)

        if col - 2 >= 0:
            attack_pair[to_sq, 3] = row * BOARD_EDGE + (col - 2)

    return attack_pair, neighbors


ATTACK_PAIR, NEIGHBORS = calc_capture_arrays()


def calc_action_legality_arrays() -> np.ndarray:
    """Calculate a legal destinations array for each square."""

    legal_dest = -np.ones((3, BOARD_SIZE, BOARD_SIZE), dtype=np.int32)

    for from_sq in range(BOARD_SIZE):
        legal_dest_for_sq = {p: [] for p in range(1, 3)}
        for to_sq in range(BOARD_SIZE):
            if from_sq == to_sq: continue
            row_from, col_from, row_to, col_to = from_sq // BOARD_EDGE, from_sq % BOARD_EDGE, to_sq // BOARD_EDGE, to_sq % BOARD_EDGE

            if abs(row_to - row_from) == 0 or abs(col_to - col_from) == 0:
                if not HOSTILE_SQUARES_MASK[to_sq]:
                    legal_dest_for_sq[TAFLMAN].append(to_sq)
                legal_dest_for_sq[KING].append(to_sq)
        for piece in range(1, 3):
            legal_dest[piece, from_sq, : len(legal_dest_for_sq[piece])] = legal_dest_for_sq[piece]

    return legal_dest


LEGAL_DEST = calc_action_legality_arrays()


def calc_between_squares() -> np.ndarray:
    """Calculate all squares indices between two squares."""

    between = -np.ones((BOARD_SIZE, BOARD_SIZE, BOARD_EDGE - 2), dtype=np.int32)
    for from_sq in range(BOARD_SIZE):
        for to_sq in range(BOARD_SIZE):
            row_from = from_sq // BOARD_EDGE
            col_from = from_sq % BOARD_EDGE
            row_to = to_sq // BOARD_EDGE
            col_to = to_sq % BOARD_EDGE

            # Skip if from- and to-squares are the same
            if not (abs(row_to - row_from) == 0 or abs(col_to - col_from) == 0):
                continue

            row_sign = max(min(row_to - row_from, 1), -1)
            col_sign = max(min(col_to - col_from, 1), -1)
            for i in range(BOARD_EDGE - 2):
                row = row_from + row_sign * (i + 1)
                col = col_from + col_sign * (i + 1)
                if row == row_to and col == col_to:
                    break
                between[from_sq, to_sq, i] = row * BOARD_EDGE + col
    return between


BETWEEN = calc_between_squares()


DIR_MAP_90 = {0: 2, 1: 0, 2: 3, 3: 1}  # right→down, up→right, down→left, left→up

def _compute_rotation_perms() -> np.ndarray:
    """
    Precompute D4 rotation permutation tables for policy vector augmentation.

    Returns a (4, BOARD_SIZE * ACTION_PLANES) int32 array where perms[k] maps each
    action label in the rotated-by-k*90° frame back to the corresponding label in the
    original frame. Used in augment_batch() to rotate policy targets and legal action
    masks consistently with the board observation rotation.

    Rotation convention: k=1 is 90° counterclockwise (matching jnp.rot90 on the board).
    Direction remapping per 90° step: right→down, up→right, down→left, left→up.
    """
    dist = BOARD_EDGE - 1
    perms = np.zeros((4, BOARD_SIZE * ACTION_PLANES), dtype=np.int32)
    perms[0] = np.arange(BOARD_SIZE * ACTION_PLANES)
    for k in range(1, 4):
        for old_label in range(BOARD_SIZE * ACTION_PLANES):
            old_sq, old_plane = old_label // ACTION_PLANES, old_label % ACTION_PLANES
            r, c = divmod(old_sq, BOARD_EDGE)
            for _ in range(k):
                r, c = BOARD_EDGE - 1 - c, r
            new_sq = r * BOARD_EDGE + c
            d = old_plane // dist
            for _ in range(k):
                d = DIR_MAP_90[d]
            new_plane = d * dist + old_plane % dist
            perms[k][new_sq * ACTION_PLANES + new_plane] = old_label
    return perms

ROTATION_PERM = _compute_rotation_perms()

(FROM_PLANE, TO_PLANE, LEGAL_DEST,
 BETWEEN, EDGES, ATTACK_PAIR, NEIGHBORS, INNER_NEIGHBOR,
 HOSTILE_SQUARES_MASK, ROWS, COLUMNS, ROTATION_PERM) = (
    jnp.array(x) for x in
    (FROM_PLANE, TO_PLANE, LEGAL_DEST,
     BETWEEN, EDGES, ATTACK_PAIR, NEIGHBORS, INNER_NEIGHBOR,
     HOSTILE_SQUARES_MASK, ROWS, COLUMNS, ROTATION_PERM))

keys = jax.random.split(jax.random.PRNGKey(12345), 4)
ZOBRIST_BOARD = jax.random.randint(keys[0], shape=(BOARD_SIZE, 5, 2), minval=0, maxval=2 ** 31 - 1, dtype=jnp.uint32)
ZOBRIST_SIDE = jax.random.randint(keys[1], shape=(2,), minval=0, maxval=2 ** 31 - 1, dtype=jnp.uint32)


class GameState(NamedTuple):
    color: Array = jnp.int32(-1)  # attacker: -1, defender: 1
    board: Array = -INIT_BOARD
    board_history: Array = jnp.zeros((8, BOARD_SIZE), dtype=jnp.int32).at[0, :].set(-INIT_BOARD)
    hash_history: Array = jnp.zeros((MAX_TERMINATION_STEPS + 1, 2), dtype=jnp.uint32)
    legal_action_mask: Array = ZERO_INIT_ACTION_MASK
    step_count: Array = jnp.int32(0)
    half_move_count: Array = jnp.int32(0)


class Action(NamedTuple):
    from_sq: Array = jnp.int32(-1)
    to_sq: Array = jnp.int32(-1)

    @staticmethod
    def from_label(label: Array):
        from_sq, plane = label // ACTION_PLANES, label % ACTION_PLANES
        return Action(from_sq=from_sq, to_sq=FROM_PLANE[from_sq, plane])

    def to_label(self):
        return self.from_sq * ACTION_PLANES + TO_PLANE[self.from_sq, self.to_sq]


def legal_moves(state: GameState, from_sq: Array) -> Array:
    """Calculate all legal moves """

    piece = state.board[from_sq]

    def legal_label(to_sq: Array) -> Array:
        dest_valid = (to_sq >= 0) & (to_sq < BOARD_SIZE)

        between_idxs = BETWEEN[from_sq, to_sq]
        path_clear = jnp.all((between_idxs == -1) | (state.board[between_idxs] == EMPTY))
        target_empty = state.board[to_sq] == EMPTY

        ok = dest_valid & path_clear & (piece > 0) & target_empty

        return lax.select(ok, Action(from_sq=from_sq, to_sq=to_sq).to_label(), -1)

    return jax.vmap(legal_label)(LEGAL_DEST[piece, from_sq])


def _legal_action_mask(state: GameState) -> Array:
    """Calculate the legal moves mask of the current game state."""

    possible_piece_positions = jnp.nonzero(state.board > 0, size=NUM_ATTACKERS, fill_value=-1)[0]
    actions = jax.vmap(lambda p: legal_moves(state, p))(possible_piece_positions).flatten()

    mask = jnp.zeros(BOARD_SIZE * ACTION_PLANES, jnp.bool_)
    mask = mask.at[actions].set(actions >= 0)

    return mask


def initialize_legal_actions(state: GameState) -> Array:
    """
    Dynamically calculates the mask for the initial board state.
    """
    return _legal_action_mask(state)

_init_check_state = GameState(board=-INIT_BOARD, color=jnp.int32(-1))
INIT_LEGAL_ACTION_MASK = initialize_legal_actions(_init_check_state)


class Game:
    @staticmethod
    def init() -> GameState:
        dummy_state = GameState()
        initial_hash = _zobrist_hash(dummy_state)

        return dummy_state._replace(
            legal_action_mask=INIT_LEGAL_ACTION_MASK,
            hash_history=jnp.zeros((MAX_TERMINATION_STEPS + 1, 2), dtype=jnp.uint32).at[0].set(initial_hash)
        )

    @staticmethod
    def step(state: GameState, action: Array) -> GameState:
        state = _apply_move(state, Action.from_label(action))
        state = _flip(state)
        state = _update_history(state)
        state = state._replace(legal_action_mask=_legal_action_mask(state))
        state = state._replace(step_count=state.step_count + 1)
        return state

    @staticmethod
    def observe(state: GameState) -> Array:
        ones = jnp.ones((1, BOARD_EDGE, BOARD_EDGE), dtype=jnp.float32)
        color = (state.color + 1) // 2

        history_2d = state.board_history.reshape((8, BOARD_EDGE, BOARD_EDGE))
        board = jnp.rot90(history_2d, k=1, axes=(1, 2))

        friendly_pieces = (board == TAFLMAN).astype(jnp.float32)
        enemy_pieces = (board == -TAFLMAN).astype(jnp.float32)
        king = (jnp.abs(board) == KING).astype(jnp.float32)

        hash_ = state.hash_history[:8, :]
        rep = (state.hash_history == hash_[:, None, :]).all(axis=2).sum(axis=1) - 1
        rep = jnp.where((hash_ == 0).all(axis=1), 0, rep)

        rep0 = jnp.broadcast_to((rep >= 1)[:, None, None], (8, BOARD_EDGE, BOARD_EDGE)).astype(jnp.float32)
        rep1 = jnp.broadcast_to((rep >= 2)[:, None, None], (8, BOARD_EDGE, BOARD_EDGE)).astype(jnp.float32)

        history_features = jnp.stack([friendly_pieces, enemy_pieces, king, rep0, rep1], axis=1)
        history_features = history_features.reshape(40, BOARD_EDGE, BOARD_EDGE)

        return jnp.vstack(
            [
                history_features,
                color * ones,
                (state.step_count / MAX_TERMINATION_STEPS) * ones,
                (state.half_move_count.astype(jnp.float32) / MAX_HALF_MOVE_COUNT) * ones
            ]
        ).transpose((1, 2, 0))

    @staticmethod
    def legal_action_mask(state: GameState) -> Array:
        return state.legal_action_mask

    @staticmethod
    def is_terminal(state: GameState) -> Array:
        # Stalemate
        terminated = ~state.legal_action_mask.any()

        # King escaped
        king_pos_mask = jnp.abs(state.board) == 2
        terminated |= (king_pos_mask & CORNERS_MASK).any()

        # King captured or encircled
        terminated |= _check_king_captured(state)

        # Repetition
        repetition = (state.hash_history == _zobrist_hash(state)).all(axis=1).sum() - 1
        terminated |= repetition >= 2

        # Draw conditions
        terminated |= state.half_move_count >= MAX_HALF_MOVE_COUNT
        terminated |= MAX_TERMINATION_STEPS <= state.step_count

        return terminated

    @staticmethod
    def mcts_status(state: GameState) -> tuple[Array, Array]:
        """
        Terminal status check used inside MCTS tree expansion.

        Unlike rewards(), returns explicit per-side scores that can be scaled by
        reward_consts in the recurrent function. Separates attacker wins, defender
        wins, and draws into distinct cases.

        Returns:
            (terminated, scores) where scores are [attacker_score, defender_score],
            each in {-1, 0, 1}.
        """
        king_captured = _check_king_captured(state)
        attacker_won = king_captured

        king_pos_mask = jnp.abs(state.board) == 2
        king_on_corner = (king_pos_mask & CORNERS_MASK).any()
        defender_won = king_on_corner

        repetition = (state.hash_history == _zobrist_hash(state)).all(axis=1).sum() - 1
        rep_loss = repetition >= 2
        attacker_won |= rep_loss & (state.color == -1)
        defender_won |= rep_loss & (state.color == 1)

        no_moves = ~state.legal_action_mask.any()
        attacker_won |= no_moves & (state.color == 1)
        defender_won |= no_moves & (state.color == -1)

        draw = (state.half_move_count >= MAX_HALF_MOVE_COUNT) | (state.step_count >= MAX_TERMINATION_STEPS)
        draw = draw & (~attacker_won) & (~defender_won)

        terminated = attacker_won | defender_won | draw

        attacker_score = jnp.where(attacker_won, 1.0, jnp.where(defender_won, -1.0, 0.0))
        defender_score = jnp.where(defender_won, 1.0, jnp.where(attacker_won, -1.0, 0.0))

        return terminated, jnp.array([attacker_score, defender_score], dtype=jnp.float32)

    @staticmethod
    def rewards(state: GameState) -> Array:
        """
        Final game rewards for the pgx State wrapper, indexed by player_order.

        Returns [attacker_score, defender_score] in {-1, 0, 1}. Used for terminal
        reward assignment in self-play and evaluation. Not used inside MCTS — see
        mcts_status() for that.
        """
        # Attackers win
        king_captured = _check_king_captured(state)

        # Defenders win
        king_on_corner = ((jnp.abs(state.board) == KING) & CORNERS_MASK).any()

        # Loss
        repetition = (state.hash_history == _zobrist_hash(state)).all(axis=1).sum() - 1
        rep_loss = repetition >= 2
        no_moves = ~state.legal_action_mask.any()

        # Technical Draws
        draw = (state.half_move_count >= MAX_HALF_MOVE_COUNT) | (state.step_count >= MAX_TERMINATION_STEPS)

        attacker_score = jnp.float32(0.0)
        defender_score = jnp.float32(0.0)

        attacker_won = king_captured
        defender_won = king_on_corner

        attacker_won |= rep_loss & (state.color == -1)
        defender_won |= rep_loss & (state.color == 1)

        attacker_won |= no_moves & (state.color == 1)
        defender_won |= no_moves & (state.color == -1)

        attacker_score = lax.select(attacker_won, 1.0, attacker_score)
        defender_score = lax.select(attacker_won, -1.0, defender_score)

        defender_score = lax.select(defender_won, 1.0, defender_score)
        attacker_score = lax.select(defender_won, -1.0, attacker_score)

        is_draw = draw & (~attacker_won) & (~defender_won)
        attacker_score = lax.select(is_draw, 0.0, attacker_score)
        defender_score = lax.select(is_draw, 0.0, defender_score)

        return jnp.array([attacker_score, defender_score])


def _check_king_captured(state: GameState) -> Array:
    return ~jnp.any(jnp.abs(state.board) == KING)


def _flip(state: GameState) -> GameState:
    return state._replace(
        board=-state.board,
        color=-state.color,
        board_history=-state.board_history
    )


def _check_captures(state: GameState, to_sq: Array) -> GameState:
    attack_indices = ATTACK_PAIR[to_sq]
    victim_indices = NEIGHBORS[to_sq]

    # Piece types of attackers and victims
    attack_pieces = jnp.take(state.board, attack_indices, mode='fill', fill_value=0)
    victim_pieces = jnp.take(state.board, victim_indices, mode='fill', fill_value=0)

    is_victim_enemy = (victim_pieces < 0)  # Check if victims are enemies
    is_attacker_friendly = (attack_pieces > 0)  # Check if attackers are friendly

    is_attacker_square_hostile = jnp.take(HOSTILE_SQUARES_MASK, jnp.maximum(attack_indices, 0), mode='fill',
                                          fill_value=False)
    is_attacker_square_hostile &= (
            attack_pieces != -KING)  # If Defenders king is on the throne, it isn't hostile for defenders

    capture_mask = (attack_indices != -1) & is_victim_enemy & (is_attacker_friendly | is_attacker_square_hostile)

    safe_victim_indices = jnp.maximum(victim_indices, 0)
    current_values = state.board[safe_victim_indices]
    new_values = jnp.where(capture_mask, EMPTY, current_values)

    return state._replace(board=state.board.at[safe_victim_indices].set(new_values))


def _apply_move(state: GameState, action: Action) -> GameState:
    piece = state.board[action.from_sq]

    # Move the piece
    state = state._replace(board=state.board.at[action.from_sq].set(EMPTY).at[action.to_sq].set(piece))

    # Check for captures
    pieces_before = jnp.count_nonzero(state.board)
    state = _check_captures(state, action.to_sq)

    pieces_after = jnp.count_nonzero(state.board)
    is_capture = pieces_after < pieces_before
    half_move_count = lax.select(is_capture, 0, state.half_move_count + 1)

    return state._replace(half_move_count=half_move_count)


def _update_history(state: GameState) -> GameState:
    board_history = jnp.roll(state.board_history, 1, axis=0)
    board_history = board_history.at[0, :].set(state.board)
    hash_history = jnp.roll(state.hash_history, 1, axis=0)
    hash_history = hash_history.at[0].set(_zobrist_hash(state))
    return state._replace(board_history=board_history, hash_history=hash_history)


def _zobrist_hash(state: GameState) -> Array:
    """
    Compute a 2-element uint32 Zobrist hash of the current board and side to move.

    The hash is XOR-reduced over per-square piece keys and a side-to-move key.
    Two uint32 elements are used instead of one to reduce collision probability
    for repetition detection across the full game history.
    """
    hash_ = lax.select(state.color == -1, ZOBRIST_SIDE, jnp.zeros_like(ZOBRIST_SIDE))
    to_reduce = ZOBRIST_BOARD[jnp.arange(BOARD_SIZE), state.board + 2]
    hash_ ^= lax.reduce(to_reduce, 0, lax.bitwise_xor, (0,))
    return hash_