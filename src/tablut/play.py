from pathlib import Path

import jax
from flax import nnx
import jax.numpy as jnp
import orbax.checkpoint as ocp

from .tablut import Tablut
from .tablut_jax import GameState, Action, BOARD_EDGE
from .ui import TablutUI
from ..mcts import run_mcts
from ..model import TablutZeroNet
from ..utils import policy_value_by_player

FILE_LETTERS = 'abcdefghijk'


class PlayTablut:
    def __init__(self, ai_color=-1, mcts_sims=300):
        self.env = Tablut()
        self.seed = 42
        self.rngs: nnx.Rngs = nnx.Rngs(self.seed)
        self.mcts_sims = mcts_sims
        root_dir = self.root = Path(__file__).resolve().parents[2]
        checkpoint_path = root_dir / 'inference' / 'model'
        self.model = self.load_model(checkpoint_path)
        self.ai_color = ai_color
        self.key_env = jax.random.PRNGKey(self.seed + 1)
        batched_state = jax.jit(jax.vmap(self.env.init))(
            jax.random.split(self.key_env, 1)
        )
        self.state = jax.tree_util.tree_map(lambda x: x[0], batched_state)
        self.step_fn = jax.jit(self.env.step)
        self.game_state: GameState = self.state.game_state
        self.board_edge = BOARD_EDGE
        self.columns = {FILE_LETTERS[i]: i for i in range(BOARD_EDGE)}

        # Warm up JAX MCTS compilation in the background to avoid JIT lag on the first move
        def warm_up():
            try:
                dummy_key = jax.random.PRNGKey(self.seed)
                g_def, m_state = nnx.split(self.model)
                _ = run_mcts(
                    graph_def=g_def,
                    model_state=m_state,
                    env_state=self.state,
                    rng_key=dummy_key,
                    num_simulations=self.mcts_sims,
                    env=self.env
                )
            except Exception:
                pass
        import threading
        threading.Thread(target=warm_up, daemon=True).start()

        # Define JIT-compiled state evaluator to run fast neural network value inference
        @jax.jit
        def eval_state_fn(observation, color):
            role = (color + 1) // 2
            g_def, m_state = nnx.split(self.model)
            local_model = nnx.merge(g_def, m_state)
            obs_batched = jnp.expand_dims(observation, axis=0)
            role_batched = jnp.expand_dims(role, axis=0)
            _, val = policy_value_by_player(local_model(obs_batched, train=False), role_batched)
            return val[0]
            
        self.eval_state_fn = eval_state_fn

        # Define JIT-compiled move probability evaluator to compute raw policy logits
        @jax.jit
        def get_move_probs_fn(observation, color, legal_action_mask):
            role = (color + 1) // 2
            g_def, m_state = nnx.split(self.model)
            local_model = nnx.merge(g_def, m_state)
            obs_batched = jnp.expand_dims(observation, axis=0)
            role_batched = jnp.expand_dims(role, axis=0)
            logits, _ = policy_value_by_player(local_model(obs_batched, train=False), role_batched)
            # Mask illegal actions
            masked_logits = jnp.where(legal_action_mask, logits[0], -1e9)
            # Softmax
            probs = jax.nn.softmax(masked_logits)
            return probs
            
        self.get_move_probs_fn = get_move_probs_fn

    def get_evaluation(self, state):
        gstate = state.game_state if hasattr(state, 'game_state') else state
        val = self.eval_state_fn(state.observation, gstate.color)
        val = float(val)
        # Convert to absolute attacker perspective (Attacker color -1, Defender color 1)
        if int(gstate.color) == 1:
            val = -val
        return val

    def evaluate_user_move(self, state, action_label):
        gstate = state.game_state if hasattr(state, 'game_state') else state
        probs = self.get_move_probs_fn(state.observation, gstate.color, gstate.legal_action_mask)
        probs_list = jnp.array(probs).tolist()
        
        # Get legal actions and their probabilities
        legal_indices = [i for i, is_legal in enumerate(gstate.legal_action_mask.tolist()) if is_legal]
        if not legal_indices:
            return "Good Move! :)", (100, 215, 125)
            
        # Pair indices with probabilities and sort descending
        legal_probs = [(idx, probs_list[idx]) for idx in legal_indices]
        legal_probs.sort(key=lambda x: x[1], reverse=True)
        
        best_action, best_prob = legal_probs[0]
        user_prob = probs_list[action_label]
        
        # Find user's rank (0-indexed)
        user_rank = -1
        for rank, (idx, p) in enumerate(legal_probs):
            if idx == action_label:
                user_rank = rank
                break
                
        # Classify the move using ratio relative to best move
        ratio = user_prob / max(1e-5, best_prob)
        
        if user_rank == 0:
            return "Best Move! :D", (212, 175, 55)  # GOLD
        elif user_rank <= 3:  # 2nd to 4th best
            if ratio >= 0.25:
                return "Good Move! :)", (100, 215, 125)  # GREEN
            elif ratio >= 0.05:
                return "Inaccurate Move :|", (230, 150, 50)  # ORANGE
            else:
                return "Mistake :(", (220, 80, 80)  # RED
        else:
            if ratio >= 0.05:
                return "Inaccurate Move :|", (230, 150, 50)  # ORANGE
            else:
                return "Mistake :(", (220, 80, 80)  # RED


    def load_model(self, checkpoint_path: Path):
        model = TablutZeroNet(
            depth=8,
            filter_count=128,
            rngs=self.rngs
        )
        model.eval()

        if checkpoint_path and Path(checkpoint_path).exists():
            print(f"Loading checkpoint...")
            checkpointer = ocp.StandardCheckpointer()
            graph_def, abstract_state = nnx.split(model)
            abstract_checkpoint = {
                'model': abstract_state,
            }
            restored = checkpointer.restore(checkpoint_path, abstract_checkpoint)
            model = nnx.merge(graph_def, restored['model'])
        else:
            print("No checkpoint loaded. Playing against randomly initialized network.")

        return model

    def make_ai_move(self):
        print("AI is thinking...")
        self.key_env, search_key = jax.random.split(self.key_env)
        graph_def, model_state = nnx.split(self.model)

        mcts_output = run_mcts(
            graph_def=graph_def,
            model_state=model_state,
            env_state=self.state,
            rng_key=search_key,
            num_simulations=self.mcts_sims,
            env=self.env
        )

        action_label = mcts_output.action[0]

        self.key_env, step_key = jax.random.split(self.key_env)
        self.state = self.step_fn(self.state, action_label, step_key)
        self.game_state = self.state.game_state

        action_obj = Action.from_label(action_label)
        uci_move = self._sq_to_uci(int(action_obj.from_sq)) + self._sq_to_uci(int(action_obj.to_sq))
        print(f"AI played: {uci_move}")
        return uci_move

    def reset(self):
        batched_state = jax.jit(jax.vmap(self.env.init))(
            jax.random.split(self.key_env, 1)
        )
        self.state = jax.tree_util.tree_map(lambda x: x[0], batched_state)
        self.game_state: GameState = self.state.game_state

    def print_board(self):
        board = self.game_state.board
        board = board.reshape((self.board_edge, self.board_edge))
        pieces = [' ', 'O', 'K', 'X']
        player = int(self.game_state.color)

        border = '   ' + self.board_edge * '──────'

        board_strings = [border]
        for row in range(self.board_edge):
            row_str = f'{self.board_edge - row}  |  '
            for column in range(self.board_edge):
                row_str += f'{pieces[int(board[row, column] * player)]}  |  '
            board_strings.append(row_str)
            board_strings.append(border)
        board_strings.append('      ' + '     '.join(list(self.columns.keys())))
        print('\n'.join(row for row in board_strings))

    def _sq_to_uci(self, sq):
        row = sq // self.board_edge
        col = sq % self.board_edge
        rank_str = str(row + 1)
        file_str = FILE_LETTERS[col]
        return file_str + rank_str

    def uci_to_action(self, uci: str):
        from_char = uci[0]

        to_char_idx = -1
        for i, char in enumerate(uci[1:], start=1):
            if char.isalpha():
                to_char_idx = i
                break

        if to_char_idx == -1:
            raise ValueError("Invalid UCI format")

        from_rank_str = uci[1:to_char_idx]
        to_char = uci[to_char_idx]
        to_rank_str = uci[to_char_idx + 1:]

        from_col = self.columns[from_char]
        from_row = self.board_edge - int(from_rank_str)

        to_col = self.columns[to_char]
        to_row = self.board_edge - int(to_rank_str)

        from_sq = from_row * self.board_edge + from_col
        to_sq = to_row * self.board_edge + to_col

        action = Action(from_sq, to_sq)
        label = action.to_label()

        if not self.state.legal_action_mask[label]:
            print(f"Debug: Move {uci} (Indices {from_sq}->{to_sq}) is masked out.")
            raise ValueError("Illegal move")

        return label

    def show_legal_moves(self):
        legal_action_mask = self.state.legal_action_mask
        valid_indices = jnp.where(legal_action_mask)[0]

        moves_uci = []
        for label_idx in valid_indices.tolist():
            action = Action.from_label(label_idx)
            f_sq = int(action.from_sq)
            t_sq = int(action.to_sq)
            uci = self._sq_to_uci(f_sq) + self._sq_to_uci(t_sq)
            moves_uci.append(uci)
        moves_uci.sort()
        print(f"Legal Moves ({len(moves_uci)}):", ", ".join(moves_uci))

    def make_move(self, move):
        try:
            action = self.uci_to_action(move)
            print(action)
            self.state = self.step_fn(self.state, action)
            self.game_state = self.state.game_state
        except Exception as e:
            print('Wrong format or illegal move: {}', e)

    def game_loop(self):
        self.print_board()
        while True:
            current_color = int(self.game_state.color)
            if current_color == self.ai_color:
                self.make_ai_move()
            else:
                self.show_legal_moves()
                print('Attacker to move' if current_color < 0 else 'Defender to move')
                move = input('Your move: ')
                self.make_move(move)

            self.print_board()

            if self.env.game.is_terminal(self.game_state):
                rewards = self.env.game.rewards(self.game_state)
                if rewards[0] > 0:
                    print('Attackers won')
                elif rewards[1] > 0:
                    print('Defenders won')
                else:
                    print('Draw')
                break

    def play_ui(self):
        ui = TablutUI(self)
        ui.run()


if __name__ == '__main__':
    game = PlayTablut(ai_color=-1)
    game.play_ui()