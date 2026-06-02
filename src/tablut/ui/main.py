import sys
import threading
import jax
import jax.numpy as jnp
from .constants import *
from .assets import Assets
from .renderer import Renderer
from .tutorial import get_tutorial_steps
from .animation import MoveAnimation
from ..tablut_jax import Action, Game

class TablutUI:
    def __init__(self, logic_engine):
        pygame.init()
        self.clock = pygame.time.Clock()
        self.engine = logic_engine

        self.screen = pygame.display.set_mode((1, 1))
        self.assets = Assets()
        
        self.fonts = {
            'title': pygame.font.SysFont("Georgia", int(28 * UI_SCALE), bold=True),
            'btn': pygame.font.SysFont("Georgia", int(18 * UI_SCALE)),
            'small': pygame.font.SysFont("Georgia", int(14 * UI_SCALE)),
            'coord': pygame.font.SysFont("Consolas", int(12 * UI_SCALE)),
            'status': pygame.font.SysFont("Georgia", int(16 * UI_SCALE), bold=True),
            'over': pygame.font.SysFont("Georgia", int(36 * UI_SCALE), bold=True),
            'timer': pygame.font.SysFont("Consolas", int(22 * UI_SCALE), bold=True)
        }

        self.renderer = Renderer(self.assets, self.fonts)
        
        self.LM = int(34 * UI_SCALE)
        self.base_w = self.renderer.base_w + self.LM
        self.base_h = self.renderer.base_h
        
        pygame.display.set_icon(pygame.transform.smoothscale(self.assets.img_knot_raw, (32, 32)))
        pygame.display.set_caption("Hnefatafl · Viking Board Game")
        self.screen = pygame.display.set_mode((self.base_w, self.base_h), pygame.RESIZABLE)
        self.render_surf = pygame.Surface((self.base_w, self.base_h))
        self.ox, self.oy = 0, 0

        self.timer_initial = TIMER_INITIAL_DEFAULT
        self.attacker_time = self.timer_initial
        self.defender_time = self.timer_initial
        self.selected_sq = None
        self.valid_moves_for_selected = []
        self.move_history = []
        self.state_history = []
        self.view_index = -1
        self.running = True
        self.game_over = False
        self.game_over_msg = ""
        self.mode = None
        self.ai_thinking = False
        self.ai_move_result = None

        self.tutorial_idx = 0
        self.tutorial_steps = get_tutorial_steps()
        
        self.animations = [] # List of MoveAnimation
        self.tutorial_anim = None
        self.tutorial_delay = 0.0
        self.tutorial_seq_idx = 0
        self.tutorial_show_win = False

        self._reset_game()

    def _reset_game(self):
        if hasattr(self.engine, 'reset'): self.engine.reset()
        self.attacker_time = self.timer_initial
        self.defender_time = self.timer_initial
        self.game_over = False; self.game_over_msg = ""
        self.move_history = []
        self.state_history = [(self.engine.state, None)]
        self.view_index = 0
        self.selected_sq = None; self.valid_moves_for_selected = []
        self.animations = []
        self.tutorial_delay = 0.0
        self.tutorial_seq_idx = 0
        self.tutorial_show_win = False
        self.ai_moves_made = 0
        self.ai_start_time = None
        self.eval_cache = {}
        self.move_feedback = None
        self.move_feedback_timer = 0.0

    def _to_menu(self):
        self._reset_game()
        self.mode = None

    def _update_offset(self):
        win_w, win_h = self.screen.get_size()
        self.ox = max(0, (win_w - self.base_w) // 2)
        self.oy = max(0, (win_h - self.base_h) // 2)

    def _to_base_coords(self, pos):
        return pos[0] - self.ox, pos[1] - self.oy

    def draw(self, dt=0.0):
        self.screen.fill(BG_DARK)
        self.render_surf.fill(BG_DARK)
        self._update_offset()

        if self.mode is None:
            self._draw_menu()
            self.screen.blit(self.render_surf, (self.ox, self.oy))
            return

        BO = self.renderer.border_w + BOARD_GAP
        S = self.render_surf
        S.blit(self.renderer.board_surf, (BO + self.LM, BO))

        if self.mode == "tutorial":
            curr_step = self.tutorial_steps[self.tutorial_idx]
            current_state, v_last_move = curr_step["state"]
            
            # Keep a persistent gstate for the active tutorial step to support sequences and intermediate changes
            if not hasattr(self, 'tutorial_gstate') or self.tutorial_gstate is None or getattr(self, 'tutorial_gstate_idx', -1) != self.tutorial_idx:
                self.tutorial_gstate = current_state.game_state if hasattr(current_state, 'game_state') else current_state
                self.tutorial_gstate_idx = self.tutorial_idx
            
            gstate = self.tutorial_gstate
            
            # Handle tutorial animation with delay on repeat
            if self.tutorial_delay > 0:
                self.tutorial_delay -= dt
                if self.tutorial_delay <= 0:
                    self.tutorial_anim = None
                    self.tutorial_gstate = None # Reset persistent state on new cycle
            elif curr_step.get("animate_move"):
                f, t = curr_step["animate_move"]
                if self.tutorial_anim is None or self.tutorial_anim.is_finished:
                    if self.tutorial_anim and self.tutorial_anim.is_finished:
                        self.tutorial_delay = 3.0 # Pause 3s before repeating
                        self.tutorial_show_win = False # Hide win during delay
                        if curr_step.get("show_win"): self.tutorial_show_win = True
                    else:
                        self.tutorial_show_win = False # Clear win when starting new anim
                        p = int(gstate.board[f])
                        self.tutorial_anim = MoveAnimation(f, t, p, turn=int(gstate.color), duration=0.6)
                if self.tutorial_anim: self.tutorial_anim.update()
            elif curr_step.get("animate_move_seq"):
                seq = curr_step["animate_move_seq"]
                if self.tutorial_anim is None or self.tutorial_anim.is_finished:
                    if self.tutorial_anim and self.tutorial_anim.is_finished:
                        # Update the persistent state to reflect the completed move
                        f_idx = self.tutorial_anim.from_idx
                        t_idx = self.tutorial_anim.to_idx
                        p_val = int(self.tutorial_gstate.board[f_idx])
                        
                        # Move the piece
                        self.tutorial_gstate = self.tutorial_gstate._replace(
                            board=self.tutorial_gstate.board.at[f_idx].set(0).at[t_idx].set(p_val)
                        )
                        # Flip perspective
                        self.tutorial_gstate = self.tutorial_gstate._replace(
                            board=-self.tutorial_gstate.board,
                            color=-self.tutorial_gstate.color
                        )
                        gstate = self.tutorial_gstate

                        self.tutorial_seq_idx += 1
                        if self.tutorial_seq_idx >= len(seq):
                            self.tutorial_delay = 4.0 # Pause 4s after sequence
                            self.tutorial_seq_idx = 0
                            self.tutorial_show_win = False
                            if curr_step.get("show_win"): self.tutorial_show_win = True
                        else:
                            self.tutorial_anim = None
                    else:
                        if self.tutorial_seq_idx == 0: self.tutorial_show_win = False
                        f, t = seq[self.tutorial_seq_idx]
                        p = int(gstate.board[f])
                        self.tutorial_anim = MoveAnimation(f, t, p, turn=int(gstate.color), duration=0.6)
                if self.tutorial_anim: self.tutorial_anim.update()

            # Convert JAX array to standard Python list once to avoid massive device-to-host sync overhead
            board_list = gstate.board.tolist()

            # For capture slides, we need to hide the captured piece in board_list
            # if the animation has reached its destination.
            if self.mode == "tutorial" and self.tutorial_anim:
                captures = curr_step.get("captures", [])
                for c_idx in captures:
                    if abs(board_list[c_idx]) == 2:
                        continue
                    # Remove the captured piece only when the sliding piece is adjacent (progress >= 0.85)
                    if self.tutorial_anim.progress >= 0.85:
                        board_list[c_idx] = 0
        else:
            current_state, v_last_move = self.state_history[self.view_index]
            gstate = current_state.game_state if hasattr(current_state, 'game_state') else current_state
            board_list = gstate.board.tolist()
            
            # Ensure captured pieces remain on the board until the sliding piece is adjacent (progress >= 0.85)
            active_anim = None
            for a in self.animations:
                if not a.is_finished and a.progress < 0.85:
                    active_anim = a
                    break
            
            if active_anim is not None and self.view_index > 0:
                prev_state, _ = self.state_history[self.view_index - 1]
                prev_gs = prev_state.game_state if hasattr(prev_state, 'game_state') else prev_state
                prev_board = prev_gs.board.tolist()
                # Find any square that was occupied in the previous state, is now empty,
                # and is NOT the sliding piece's source or destination
                for sq in range(BOARD_SIZE):
                    if sq != active_anim.from_idx and sq != active_anim.to_idx:
                        prev_p = prev_board[sq]
                        curr_p = board_list[sq]
                        if prev_p != 0 and curr_p == 0:
                            # Put the captured piece back on the board temporarily for drawing!
                            # We negate prev_p because the board perspective is flipped on turn change!
                            board_list[sq] = -prev_p
            
            # Ensure King stays on the board if captured (not present on current board)
            has_king = any(abs(x) == 2 for x in board_list)
            if not has_king:
                # Find the last position of the King in the state history
                king_sq = None
                king_val = None
                for hist_state, _ in reversed(self.state_history[:self.view_index + 1]):
                    hist_gs = hist_state.game_state if hasattr(hist_state, 'game_state') else hist_state
                    hist_board = hist_gs.board.tolist()
                    try:
                        king_sq = next(i for i, x in enumerate(hist_board) if abs(x) == 2)
                        king_val = hist_board[king_sq]
                        break
                    except StopIteration:
                        continue
                if king_sq is not None and king_val is not None:
                    # Put the King back on the board list for rendering
                    board_list[king_sq] = king_val

        # Draw last move highlights
        if v_last_move:
            self._draw_highlights(S, v_last_move, (255, 195, 50, 30), BO)

        # Draw selection and valid moves
        if self.mode != "tutorial":
            if self.view_index == len(self.state_history) - 1 and self.selected_sq is not None:
                self._draw_selection(S, self.selected_sq, BO)
                self._draw_valid_moves(S, self.valid_moves_for_selected, BO)
        else:
            v_sq = curr_step.get("show_valid_sq")
            if v_sq is not None:
                self._draw_selection(S, v_sq, BO)
                self._draw_valid_moves(S, self._get_legal_dests_for_state(v_sq, current_state), BO)

        if self.mode == "tutorial":
            hsq = curr_step.get("highlight_sq")
            if hsq is not None: self._draw_highlight_sq(S, hsq, GOLD, BO)

        # Update and filter animations
        self.animations = [a for a in self.animations if not a.is_finished]
        for a in self.animations: a.update()

        # Draw pieces
        animating_to_indices = [a.to_idx for a in self.animations]
        
        # In tutorial, if we are in delay/pause period AFTER an animation, 
        # we want to show the piece at the DESTINATION (to_idx), so we DON'T exclude it.
        # But during active animation, we do exclude it.
        if self.tutorial_anim and not self.tutorial_anim.is_finished:
            animating_to_indices.append(self.tutorial_anim.to_idx)
        
        # turn color from gstate
        turn = int(gstate.color)
        for idx in range(BOARD_SIZE):
            if idx in animating_to_indices: continue
            
            # Special case for tutorial: if we finished an animation but are in delay, 
            # hide the piece at from_idx and show at to_idx (if not already there)
            if self.mode == "tutorial" and self.tutorial_anim and self.tutorial_anim.is_finished:
                if idx == self.tutorial_anim.from_idx: continue
            
            piece = board_list[idx]
            if piece == 0: continue
            
            # If this is tutorial, and we are animating the piece FROM this idx
            if self.mode == "tutorial" and self.tutorial_anim and not self.tutorial_anim.is_finished and idx == self.tutorial_anim.from_idx:
                continue

            row, col = divmod(idx, BOARD_EDGE)
            ui_r = BOARD_EDGE - 1 - row
            pos = (BO + col * CELL + CELL // 2 + self.LM, BO + ui_r * CELL + CELL // 2)
            # Use actual board turn to correctly map Attacker vs Defender
            self.renderer.draw_piece(S, pos, piece, turn)

        # Draw animated pieces
        for a in self.animations:
            ax, ay = a.get_pos(BOARD_EDGE, CELL)
            # Use stored turn of the animation if available, else fall back to current turn or piece sign
            a_turn = a.turn if a.turn is not None else (-1 if a.piece < 0 else 1)
            self.renderer.draw_piece(S, (BO + ax + CELL // 2 + self.LM, BO + ay + CELL // 2), a.piece, a_turn)
            
        if self.mode == "tutorial" and self.tutorial_anim:
            a_turn = self.tutorial_anim.turn if self.tutorial_anim.turn is not None else (-1 if self.tutorial_anim.piece < 0 else 1)
            if not self.tutorial_anim.is_finished:
                ax, ay = self.tutorial_anim.get_pos(BOARD_EDGE, CELL)
                self.renderer.draw_piece(S, (BO + ax + CELL // 2 + self.LM, BO + ay + CELL // 2), self.tutorial_anim.piece, a_turn)
            else:
                # Still show it at destination during delay
                to_idx = self.tutorial_anim.to_idx
                row, col = divmod(to_idx, BOARD_EDGE)
                ui_r = BOARD_EDGE - 1 - row
                pos = (BO + col * CELL + CELL // 2 + self.LM, BO + ui_r * CELL + CELL // 2)
                self.renderer.draw_piece(S, pos, self.tutorial_anim.piece, a_turn)

        S.blit(self.renderer.border_surf, (self.LM, 0))

        # Draw Eval Bar on the far left side, outside of the Celtic border frame (drawn on top to prevent border overlapping)
        if self.mode != "tutorial":
            # 1. Fetch current evaluation value from cache to avoid blocking JAX thread during AI search
            if not hasattr(self, 'eval_cache'):
                self.eval_cache = {}
            if self.view_index not in self.eval_cache:
                current_state = self.state_history[self.view_index][0]
                self.eval_cache[self.view_index] = self.engine.get_evaluation(current_state)
            val = self.eval_cache[self.view_index]
            
            # Smoothly transition the eval bar display value
            if not hasattr(self, 'current_eval_smooth'):
                self.current_eval_smooth = val
            self.current_eval_smooth += (val - self.current_eval_smooth) * 0.1
            
            # Dimensions
            eb_w = int(10 * UI_SCALE)
            eb_h = BOARD_PX
            # Place on the far left (with a 18-pixel margin), to the left of the Celtic frame (which starts at self.LM)
            eb_x = int(18 * UI_SCALE)
            eb_y = BO
            
            # Draw outer dark slot and border
            pygame.draw.rect(S, (22, 19, 18), (eb_x, eb_y, eb_w, eb_h), border_radius=4)
            pygame.draw.rect(S, GOLD_DIM, (eb_x, eb_y, eb_w, eb_h), 1, border_radius=4)
            
            # Draw Defender area (entire bar as background) - Red representing Defender
            pygame.draw.rect(S, STATUS_DEF, (eb_x + 1, eb_y + 1, eb_w - 2, eb_h - 2), border_radius=3)
            
            # Draw Attacker area (top portion based on attacker advantage) - Blue representing Attacker
            frac = (self.current_eval_smooth + 1.0) / 2.0
            frac = max(0.0, min(1.0, frac))
            
            atk_h = int((eb_h - 2) * frac)
            if atk_h > 0:
                pygame.draw.rect(S, STATUS_ATK, (eb_x + 1, eb_y + 1, eb_w - 2, atk_h), border_radius=3)
                
            # Draw central gold divider line (0.0 evaluation)
            pygame.draw.line(S, GOLD, (eb_x, eb_y + eb_h // 2), (eb_x + eb_w - 1, eb_y + eb_h // 2), 1)
            
            # Display text evaluation (e.g. "+0.34" or "-0.15") centered above the bar, with a guard to prevent clipping at the left edge
            eval_text = f"{'+' if val >= 0 else ''}{val:.2f}"
            eval_lbl = self.fonts['coord'].render(eval_text, True, GOLD if val >= 0 else CREAM)
            text_x = max(2, eb_x + eb_w // 2 - eval_lbl.get_width() // 2)
            S.blit(eval_lbl, (text_x, eb_y - int(16 * UI_SCALE)))
        
        if self.mode == "tutorial":
            self._draw_tutorial_panel()
        else:
            self._draw_sidebar()
            
            # Draw floating move feedback inside sidepanel (drawn on S, only in game mode, not tutorial)
            if getattr(self, 'move_feedback_timer', 0.0) > 0.0:
                self.move_feedback_timer -= dt
                
                # Dimensions to fit sidebar
                tw_w = SIDEBAR_W - int(40 * UI_SCALE)
                tw_h = int(34 * UI_SCALE)
                
                sx = self.renderer.total_board + SIDEBAR_GAP + self.LM
                tx = sx + int(20 * UI_SCALE)
                ty = int(220 * UI_SCALE)
                
                # Calculate alpha fade based on remaining time
                alpha = 255
                if self.move_feedback_timer < 0.5:
                    alpha = int(255 * (self.move_feedback_timer / 0.5))
                alpha = max(0, min(255, alpha))
                
                # Draw semi-transparent rounded card directly on S
                toast_surf = pygame.Surface((tw_w, tw_h), pygame.SRCALPHA)
                
                # Deep rich dark brown background
                pygame.draw.rect(toast_surf, (22, 19, 18, int(min(240, alpha))), (0, 0, tw_w, tw_h), border_radius=8)
                # Border using feedback color
                border_color = (int(self.move_feedback_color[0]), int(self.move_feedback_color[1]), int(self.move_feedback_color[2]), int(alpha))
                pygame.draw.rect(toast_surf, border_color, (0, 0, tw_w, tw_h), 2, border_radius=8)
                
                # Text label
                lbl = self.fonts['status'].render(self.move_feedback, True, (int(self.move_feedback_color[0]), int(self.move_feedback_color[1]), int(self.move_feedback_color[2])))
                lbl.set_alpha(alpha)
                toast_surf.blit(lbl, (tw_w // 2 - lbl.get_width() // 2, tw_h // 2 - lbl.get_height() // 2))
                
                # Blit toast onto S
                S.blit(toast_surf, (tx, ty))

        self.screen.blit(self.render_surf, (self.ox, self.oy))
        if self.game_over: self._draw_game_over()

    def _draw_highlights(self, S, sqs, color, BO):
        arm = max(6, CELL // 7)
        surf = pygame.Surface((CELL, CELL), pygame.SRCALPHA)
        surf.fill(color)
        bc = (color[0], color[1], color[2], 150)
        for (px, py), dx, dy in [((1, 1), 1, 1), ((CELL-2, 1), -1, 1), ((1, CELL-2), 1, -1), ((CELL-2, CELL-2), -1, -1)]:
            pygame.draw.line(surf, bc, (px, py), (px + dx * arm, py), 2)
            pygame.draw.line(surf, bc, (px, py), (px, py + dy * arm), 2)
        for sq in sqs:
            r, c = divmod(sq, BOARD_EDGE)
            S.blit(surf, (BO + c * CELL + self.LM, BO + (BOARD_EDGE - 1 - r) * CELL))

    def _draw_selection(self, S, sq, BO):
        arm = max(8, CELL // 6)
        surf = pygame.Surface((CELL, CELL), pygame.SRCALPHA)
        surf.fill((255, 210, 60, 45))
        pygame.draw.rect(surf, (255, 220, 70, 160), surf.get_rect(), 2)
        sc = (255, 235, 90, 245)
        for (px, py), dx, dy in [((1, 1), 1, 1), ((CELL-2, 1), -1, 1), ((1, CELL-2), 1, -1), ((CELL-2, CELL-2), -1, -1)]:
            pygame.draw.line(surf, sc, (px, py), (px + dx * arm, py), 3)
            pygame.draw.line(surf, sc, (px, py), (px, py + dy * arm), 3)
        r, c = divmod(sq, BOARD_EDGE)
        S.blit(surf, (BO + c * CELL + self.LM, BO + (BOARD_EDGE - 1 - r) * CELL))

    def _draw_valid_moves(self, S, moves, BO):
        half = max(10, CELL // 6)
        surf = pygame.Surface((CELL, CELL), pygame.SRCALPHA)
        ccx, ccy = CELL // 2, CELL // 2
        pts = [(ccx, ccy - half), (ccx + half, ccy), (ccx, ccy + half), (ccx - half, ccy)]
        pygame.draw.polygon(surf, (100, 215, 125, 210), pts, 2)
        pygame.draw.circle(surf, (100, 215, 125, 230), (ccx, ccy), 3)
        for dest in moves:
            dr, dc = divmod(dest, BOARD_EDGE)
            S.blit(surf, (BO + dc * CELL + self.LM, BO + (BOARD_EDGE - 1 - dr) * CELL))

    def _draw_highlight_sq(self, S, sq, color, BO):
        arm = max(10, CELL // 5)
        surf = pygame.Surface((CELL, CELL), pygame.SRCALPHA)
        for (px, py), dx, dy in [((1, 1), 1, 1), ((CELL-2, 1), -1, 1), ((1, CELL-2), 1, -1), ((CELL-2, CELL-2), -1, -1)]:
            pygame.draw.line(surf, color, (px, py), (px + dx * arm, py), 4)
            pygame.draw.line(surf, color, (px, py), (px, py + dy * arm), 4)
        r, c = divmod(sq, BOARD_EDGE)
        S.blit(surf, (BO + c * CELL + self.LM, BO + (BOARD_EDGE - 1 - r) * CELL))

    def _draw_menu(self):
        S = self.render_surf; cx, cy = self.base_w // 2, self.base_h // 2
        
        # 1. Draw two elegant, highly visible flanking Celtic knots (Left & Right) to frame the menu controls
        ks = int(160 * UI_SCALE)
        kimg = pygame.transform.smoothscale(self.assets.img_knot_raw, (ks, ks))
        arr = pygame.surfarray.pixels3d(kimg)
        for c, v in enumerate(GOLD_DIM):
            arr[:, :, c] = (arr[:, :, c].astype(int) * v // 255).clip(0, 255)
        del arr
        kimg.set_alpha(110)
        
        # Left flanking knot
        S.blit(kimg, (cx - int(300 * UI_SCALE) - ks // 2, cy - ks // 2))
        # Right flanking knot
        S.blit(kimg, (cx + int(300 * UI_SCALE) - ks // 2, cy - ks // 2))

        # Title and styled separator line
        title = self.fonts['title'].render("TABLUT", True, GOLD)
        S.blit(title, (cx - title.get_width() // 2, cy - int(200 * UI_SCALE)))
        pygame.draw.line(S, GOLD_DIM, (cx - 120, cy - int(162 * UI_SCALE)), (cx + 120, cy - int(162 * UI_SCALE)), 1)
        
        # 2. Premium Time Dial Card (Glassmorphism & Gold Capsule)
        pw, ph = int(300 * UI_SCALE), int(76 * UI_SCALE)
        px, py = cx - pw // 2, cy - int(155 * UI_SCALE)
        
        # Transparent surface for capsule card
        card_surf = pygame.Surface((pw, ph), pygame.SRCALPHA)
        pygame.draw.rect(card_surf, (22, 19, 18, 200), (0, 0, pw, ph), border_radius=12)
        pygame.draw.rect(card_surf, GOLD_DIM, (0, 0, pw, ph), 1, border_radius=12)
        S.blit(card_surf, (px, py))
        
        # Header label inside card
        header_lbl = self.fonts['small'].render("INITIAL TIME LIMIT", True, GOLD_DIM)
        S.blit(header_lbl, (cx - header_lbl.get_width() // 2, py + int(8 * UI_SCALE)))
        
        # Large centered time limit text
        time_val_text = f"{int(self.timer_initial // 60)}:00"
        time_lbl = self.fonts['timer'].render(time_val_text, True, CREAM)
        unit_lbl = self.fonts['small'].render("min", True, GOLD_DIM)
        
        # Draw them together centered
        total_w = time_lbl.get_width() + 4 + unit_lbl.get_width()
        start_x = cx - total_w // 2
        S.blit(time_lbl, (start_x, py + int(32 * UI_SCALE)))
        S.blit(unit_lbl, (start_x + time_lbl.get_width() + 4, py + int(40 * UI_SCALE)))
        
        # Plus and minus buttons as beautiful anti-aliased circles
        btn_r = int(17 * UI_SCALE)
        minus_center = (px + int(32 * UI_SCALE), py + int(45 * UI_SCALE))
        plus_center = (px + pw - int(32 * UI_SCALE), py + int(45 * UI_SCALE))
        
        self.menu_timer_btns = {
            "minus": pygame.Rect(minus_center[0] - btn_r, minus_center[1] - btn_r, btn_r * 2, btn_r * 2),
            "plus": pygame.Rect(plus_center[0] - btn_r, plus_center[1] - btn_r, btn_r * 2, btn_r * 2)
        }
        
        for key, rect in self.menu_timer_btns.items():
            center = minus_center if key == "minus" else plus_center
            is_hover = rect.collidepoint(self._to_base_coords(pygame.mouse.get_pos()))
            color = MENU_BTN_HOV if is_hover else MENU_BTN
            
            # Draw beautiful button circle and gold border
            pygame.draw.circle(S, color, center, btn_r)
            pygame.draw.circle(S, GOLD, center, btn_r, 1)
            
            # Draw clean anti-aliased mathematical symbols
            line_w = max(2, int(2 * UI_SCALE))
            arm = int(6 * UI_SCALE)
            if key == "minus":
                pygame.draw.line(S, CREAM, (center[0] - arm, center[1]), (center[0] + arm, center[1]), line_w)
            else:
                pygame.draw.line(S, CREAM, (center[0] - arm, center[1]), (center[0] + arm, center[1]), line_w)
                pygame.draw.line(S, CREAM, (center[0], center[1] - arm), (center[0], center[1] + arm), line_w)

        # 3. Premium styled menu buttons
        self.menu_btn_rects = {}
        btn_data = [
            ("Play vs AI (Attacker)", "ai_atk"),
            ("Play vs AI (Defender)", "ai_def"),
            ("Player vs Player", "pvp"),
            ("Tutorial", "tutorial"),
            ("Quit Game", "quit")
        ]
        btn_y = cy - int(65 * UI_SCALE)
        for label, key in btn_data:
            rect = pygame.Rect(cx - int(130 * UI_SCALE), btn_y, int(260 * UI_SCALE), int(36 * UI_SCALE))
            self.menu_btn_rects[key] = rect
            
            is_hover = rect.collidepoint(self._to_base_coords(pygame.mouse.get_pos()))
            
            if is_hover:
                color = MENU_BTN_HOV
                border_color = GOLD
                text_color = GOLD if key != "quit" else (255, 100, 100)
            else:
                color = MENU_BTN
                border_color = GOLD_DIM
                text_color = MENU_TEXT if key != "quit" else (210, 80, 80)
                
            pygame.draw.rect(S, color, rect, border_radius=8)
            pygame.draw.rect(S, border_color, rect, 1, border_radius=8)
            
            lbl = self.fonts['btn'].render(label, True, text_color)
            S.blit(lbl, (rect.centerx - lbl.get_width() // 2, rect.centery - lbl.get_height() // 2))
            btn_y += int(48 * UI_SCALE)

    def _draw_sidebar(self):
        S = self.render_surf; sx = self.renderer.total_board + SIDEBAR_GAP + self.LM
        pygame.draw.line(S, GOLD_DIM, (sx, 0), (sx, self.base_h), 2)
        title = self.fonts['title'].render("TABLUT", True, GOLD)
        S.blit(title, (sx + (SIDEBAR_W - title.get_width()) // 2, 20))
        turn = int(self.engine.game_state.color)
        lbl = self.fonts['status'].render("Attacker's Turn" if turn == -1 else "Defender's Turn", True, STATUS_ATK if turn == -1 else STATUS_DEF)
        S.blit(lbl, (sx + (SIDEBAR_W - lbl.get_width()) // 2, int(80 * UI_SCALE)))
        def fmt(t): m, s = divmod(int(t), 60); return f"{m:02d}:{s:02d}"
        atk_t = self.fonts['timer'].render(fmt(self.attacker_time), True, STATUS_ATK)
        def_t = self.fonts['timer'].render(fmt(self.defender_time), True, STATUS_DEF)
        S.blit(atk_t, (sx + 30, int(115 * UI_SCALE))); S.blit(def_t, (sx + SIDEBAR_W - 30 - def_t.get_width(), int(115 * UI_SCALE)))
        
        # AI Thinking Progress Bar
        if self.ai_thinking:
            if not hasattr(self, 'ai_start_time') or self.ai_start_time is None:
                self.ai_start_time = pygame.time.get_ticks()
                self.ai_current_progress = 0.0
            elapsed = (pygame.time.get_ticks() - self.ai_start_time) / 1000.0
            
            # Predict expected time: use the duration of the last AI move if available, otherwise a sensible default
            if hasattr(self, 'last_ai_duration'):
                expected_time = self.last_ai_duration
            else:
                sims = getattr(self.engine, 'mcts_sims', 200)
                expected_time = 0.5 + sims * 0.008  # ~2.9s for 300 sims
            
            target_progress = min(0.98, elapsed / max(0.1, expected_time))
            
            if not hasattr(self, 'ai_current_progress'):
                self.ai_current_progress = 0.0
            # Interpolate faster (0.2 instead of 0.05) to eliminate the 3-second hang at "almost full"
            self.ai_current_progress += (target_progress - self.ai_current_progress) * 0.2
            progress = self.ai_current_progress
            
            pb_w = SIDEBAR_W - 60
            pb_h = int(8 * UI_SCALE)
            pb_x = sx + 30
            pb_y = int(172 * UI_SCALE)
            
            pb_lbl = self.fonts['small'].render(f"AI Thinking ({int(progress * 100)}%)", True, GOLD)
            S.blit(pb_lbl, (sx + (SIDEBAR_W - pb_lbl.get_width()) // 2, pb_y - int(18 * UI_SCALE)))
            
            pygame.draw.rect(S, (22, 19, 18), (pb_x, pb_y, pb_w, pb_h), border_radius=4)
            pygame.draw.rect(S, GOLD_DIM, (pb_x, pb_y, pb_w, pb_h), 1, border_radius=4)
            if progress > 0.01:
                pygame.draw.rect(S, GOLD, (pb_x + 1, pb_y + 1, int((pb_w - 2) * progress), pb_h - 2), border_radius=3)
        
        # Decorative Celtic knot in the center of the sidebar
        ks = int(100 * UI_SCALE)
        kimg = pygame.transform.smoothscale(self.assets.img_knot_raw, (ks, ks))
        arr = pygame.surfarray.pixels3d(kimg)
        for c, v in enumerate(BOARD_LINE):
            arr[:, :, c] = (arr[:, :, c].astype(int) * v // 255).clip(0, 255)
        del arr
        kimg.set_alpha(80)
        S.blit(kimg, (sx + SIDEBAR_W // 2 - ks // 2, self.base_h // 2 - ks // 2))

        # Bottom buttons
        self.btn_rects = {}
        btn_y = self.base_h - int(155 * UI_SCALE)
        for label, key in [("Reset Game", "reset"), ("Back to Menu", "menu")]:
            rect = pygame.Rect(sx + int(20 * UI_SCALE), btn_y, SIDEBAR_W - int(40 * UI_SCALE), int(38 * UI_SCALE)); self.btn_rects[key] = rect
            color = MENU_BTN_HOV if rect.collidepoint(self._to_base_coords(pygame.mouse.get_pos())) else MENU_BTN
            pygame.draw.rect(S, color, rect, border_radius=6); pygame.draw.rect(S, GOLD_DIM, rect, 1, border_radius=6)
            lbl = self.fonts['btn'].render(label, True, MENU_TEXT); S.blit(lbl, (rect.centerx - lbl.get_width() // 2, rect.centery - lbl.get_height() // 2))
            btn_y += int(50 * UI_SCALE)
            
        nav_y = self.base_h - int(45 * UI_SCALE)
        nav_w = (SIDEBAR_W - int(50 * UI_SCALE)) // 4
        for i, (label, key) in enumerate([("|<", "start"), ("<", "prev"), (">", "next"), (">|", "end")]):
            rect = pygame.Rect(sx + int(25 * UI_SCALE) + i * nav_w, nav_y, nav_w - 4, int(30 * UI_SCALE)); self.btn_rects[key] = rect
            
            # Check if button is disabled (pressing it has no effect)
            disabled = False
            if key in ("start", "prev") and self.view_index == 0:
                disabled = True
            elif key in ("next", "end") and self.view_index >= len(self.state_history) - 1:
                disabled = True
                
            if disabled:
                color = (38, 34, 30)  # Muted, blended with BG
                border_color = (68, 62, 52)  # Muted border
                text_color = (100, 90, 80)  # Dimmed text
            else:
                color = MENU_BTN_HOV if rect.collidepoint(self._to_base_coords(pygame.mouse.get_pos())) else MENU_BTN
                border_color = GOLD_DIM
                text_color = MENU_TEXT
                
            pygame.draw.rect(S, color, rect, border_radius=4); pygame.draw.rect(S, border_color, rect, 1, border_radius=4)
            lbl = self.fonts['small'].render(label, True, text_color); S.blit(lbl, (rect.centerx - lbl.get_width() // 2, rect.centery - lbl.get_height() // 2))

    def _draw_tutorial_panel(self):
        S = self.render_surf; sx = self.renderer.total_board + SIDEBAR_GAP + self.LM
        BO = self.renderer.border_w + BOARD_GAP
        
        # Initialize button rects and register sidebar Quit button
        self.tutorial_btn_rects = {}
        quit_w = int(80 * UI_SCALE)
        quit_h = int(28 * UI_SCALE)
        quit_rect = pygame.Rect(sx + SIDEBAR_W // 2 - quit_w // 2, self.base_h - int(105 * UI_SCALE), quit_w, quit_h)
        self.tutorial_btn_rects["quit"] = quit_rect
        
        # 1. Draw a sleek sidebar in tutorial mode
        pygame.draw.line(S, GOLD_DIM, (sx, 0), (sx, self.base_h), 2)
        
        # Title banner in the sidebar
        lbl_mode = self.fonts['status'].render("TUTORIAL", True, GOLD)
        S.blit(lbl_mode, (sx + SIDEBAR_W // 2 - lbl_mode.get_width() // 2, 40))
        
        step = self.tutorial_steps[self.tutorial_idx]
        
        # Decorative Celtic knot in the center of the sidebar (always drawn behind)
        ks = int(100 * UI_SCALE)
        kimg = pygame.transform.smoothscale(self.assets.img_knot_raw, (ks, ks))
        arr = pygame.surfarray.pixels3d(kimg)
        for c, v in enumerate(BOARD_LINE):
            arr[:, :, c] = (arr[:, :, c].astype(int) * v // 255).clip(0, 255)
        del arr
        kimg.set_alpha(80)
        S.blit(kimg, (sx + SIDEBAR_W // 2 - ks // 2, self.base_h // 2 - ks // 2))
        
        if self.tutorial_show_win:
            # Draw beautiful, premium victory card in the sidebar on top of the Celtic knot
            card_w = int(180 * UI_SCALE)
            card_h = int(50 * UI_SCALE)
            card_x = sx + SIDEBAR_W // 2 - card_w // 2
            card_y = self.base_h // 2 - card_h // 2
            
            card_surf = pygame.Surface((card_w, card_h), pygame.SRCALPHA)
            # Semi-transparent dark background
            pygame.draw.rect(card_surf, (22, 19, 18, 240), (0, 0, card_w, card_h), border_radius=12)
            pygame.draw.rect(card_surf, GOLD, (0, 0, card_w, card_h), 2, border_radius=12)
            
            # Winner label
            win_msg = step.get("show_win", "")
            lbl_win = self.fonts['status'].render(win_msg, True, STATUS_ATK if "ATTACKER" in win_msg else STATUS_DEF)
            card_surf.blit(lbl_win, (card_w // 2 - lbl_win.get_width() // 2, card_h // 2 - lbl_win.get_height() // 2))
            
            S.blit(card_surf, (card_x, card_y))
            
        # A small info text at the bottom of the sidebar
        step_lbl = self.fonts['small'].render(f"Step {self.tutorial_idx + 1} of {len(self.tutorial_steps)}", True, CREAM)
        S.blit(step_lbl, (sx + SIDEBAR_W // 2 - step_lbl.get_width() // 2, self.base_h - 50))
        
        # 2. Draw the popup window on the board itself
        # Word wrap the message text to fit inside the popup width
        popup_w = int(240 * UI_SCALE)
        msg_text = step["msg"]
        paragraphs = msg_text.split('\n')
        lines = []
        for paragraph in paragraphs:
            if paragraph.strip() == "":
                lines.append("")
                continue
            words = paragraph.split(' ')
            curr_line = ""
            for w in words:
                test = curr_line + w + " "
                if self.fonts['small'].size(test)[0] < popup_w - 40:
                    curr_line = test
                else:
                    lines.append(curr_line)
                    curr_line = w + " "
            lines.append(curr_line)
            
        title_lbl = self.fonts['status'].render(step["title"], True, GOLD)
        title_h = title_lbl.get_height()
        title_y = int(12 * UI_SCALE)
        sep_y = title_y + title_h + int(6 * UI_SCALE)
        ly_start = sep_y + int(8 * UI_SCALE)
        
        # Dynamically calculate the popup height
        line_h = int(14 * UI_SCALE)
        btn_h = int(28 * UI_SCALE)
        popup_h = ly_start + len(lines) * line_h + int(15 * UI_SCALE) + btn_h + int(15 * UI_SCALE)
        
        # Calculate popup top-left base coordinates on the board
        p_row, p_col = step["popup_pos"]
        cell_x = p_col * CELL
        cell_ui_r = (BOARD_EDGE - 1 - p_row) * CELL
        ccx, ccy = cell_x + CELL // 2, cell_ui_r + CELL // 2
        px = ccx - popup_w // 2
        py = ccy - popup_h // 2
        
        # Restrict popup fully inside the board area
        px = max(15, min(BOARD_PX - popup_w - 15, px))
        py = max(15, min(BOARD_PX - popup_h - 15, py))
        
        # Create a semi-transparent surface for the popup bubble
        popup_surf = pygame.Surface((popup_w, popup_h), pygame.SRCALPHA)
        # Deep dark rich brown-charcoal background with high opacity
        pygame.draw.rect(popup_surf, (22, 19, 18, 240), (0, 0, popup_w, popup_h), border_radius=12)
        # Elegant gold border
        pygame.draw.rect(popup_surf, GOLD, (0, 0, popup_w, popup_h), 2, border_radius=12)
        
        # Blit Title inside popup
        popup_surf.blit(title_lbl, (20, title_y))
        pygame.draw.line(popup_surf, (*GOLD_DIM, 120), (20, sep_y), (popup_w - 20, sep_y), 1)
        
        # Blit wrapped lines inside popup
        ly = ly_start
        for line in lines:
            if line.strip() != "":
                txt = self.fonts['small'].render(line.strip(), True, CREAM)
                popup_surf.blit(txt, (20, ly))
            ly += line_h
            
        # Draw the popup surf onto the board surface S
        S.blit(popup_surf, (BO + px + self.LM, BO + py))
        
        # 3. Handle buttons (drawn on top of S to enable mouse hovers/clicks easily)
        btn_w = int(75 * UI_SCALE)
        local_y = popup_h - btn_h - int(15 * UI_SCALE)
        base_x = BO + px + self.LM
        base_y = BO + py
        
        if self.tutorial_idx < len(self.tutorial_steps) - 1:
            r = pygame.Rect(base_x + popup_w - btn_w - 20, base_y + local_y, btn_w, btn_h)
            self.tutorial_btn_rects["next"] = r
        else:
            r = pygame.Rect(base_x + popup_w - btn_w - 20, base_y + local_y, btn_w, btn_h)
            self.tutorial_btn_rects["finish"] = r
            
        if self.tutorial_idx > 0:
            r = pygame.Rect(base_x + 20, base_y + local_y, btn_w, btn_h)
            self.tutorial_btn_rects["prev"] = r
            
        for key, rect in self.tutorial_btn_rects.items():
            color = MENU_BTN_HOV if rect.collidepoint(self._to_base_coords(pygame.mouse.get_pos())) else MENU_BTN
            pygame.draw.rect(S, color, rect, border_radius=6)
            pygame.draw.rect(S, GOLD_DIM, rect, 1, border_radius=6)
            lbl = self.fonts['btn'].render("Finish" if key=="finish" else key.capitalize(), True, MENU_TEXT)
            S.blit(lbl, (rect.centerx - lbl.get_width() // 2, rect.centery - lbl.get_height() // 2))

    def _draw_game_over(self):
        sw, sh = self.screen.get_size()
        
        # Dimensions matching the elegant tutorial popup styling
        popup_w = int(260 * UI_SCALE)
        popup_h = int(140 * UI_SCALE)
        
        # Coordinates in center of screen
        px = sw // 2 - popup_w // 2
        py = sh // 2 - popup_h // 2
        
        # Create a semi-transparent surface for the popup bubble
        popup_surf = pygame.Surface((popup_w, popup_h), pygame.SRCALPHA)
        # Deep dark rich brown-charcoal background with high opacity (240)
        pygame.draw.rect(popup_surf, (22, 19, 18, 240), (0, 0, popup_w, popup_h), border_radius=12)
        # Elegant gold border
        pygame.draw.rect(popup_surf, GOLD, (0, 0, popup_w, popup_h), 2, border_radius=12)
        
        # 1. Title "GAME OVER"
        title_lbl = self.fonts['status'].render("GAME OVER", True, GOLD)
        title_y = int(12 * UI_SCALE)
        popup_surf.blit(title_lbl, (popup_w // 2 - title_lbl.get_width() // 2, title_y))
        
        # Separator line
        sep_y = title_y + title_lbl.get_height() + int(6 * UI_SCALE)
        pygame.draw.line(popup_surf, (*GOLD_DIM, 120), (20, sep_y), (popup_w - 20, sep_y), 1)
        
        # 2. Winning message
        msg_lbl = self.fonts['over'].render(self.game_over_msg, True, CREAM)
        msg_y = sep_y + int(15 * UI_SCALE)
        # Scale if too wide
        if msg_lbl.get_width() > popup_w - 40:
            msg_lbl = pygame.transform.smoothscale(msg_lbl, (popup_w - 40, int(msg_lbl.get_height() * (popup_w - 40) / msg_lbl.get_width())))
        popup_surf.blit(msg_lbl, (popup_w // 2 - msg_lbl.get_width() // 2, msg_y))
        
        # Blit popup surface to screen
        self.screen.blit(popup_surf, (px, py))

    def handle_board_click(self, pos):
        if self.game_over or self.mode is None or self.view_index != len(self.state_history) - 1: return
        bx, by = self._to_base_coords(pos); BO = self.renderer.border_w + BOARD_GAP
        bx -= BO + self.LM; by -= BO
        if bx < 0 or by < 0 or bx >= BOARD_PX or by >= BOARD_PX: return
        col, ui_row = int(bx) // CELL, int(by) // CELL; row = BOARD_EDGE - 1 - ui_row; idx = row * BOARD_EDGE + col
        if not (0 <= idx < BOARD_SIZE): return
        
        state = self.state_history[self.view_index][0]
        gstate = state.game_state if hasattr(state, 'game_state') else state
        
        if self.selected_sq is not None and idx in self.valid_moves_for_selected:
            self._execute_move(self.selected_sq, idx); self.selected_sq = None; self.valid_moves_for_selected = []
            return
        p = int(gstate.board[idx])
        if p > 0:
            self.selected_sq = idx; self.valid_moves_for_selected = self._get_legal_dests(idx)
        else: self.selected_sq = None; self.valid_moves_for_selected = []

    def _get_legal_dests(self, from_sq):
        state, _ = self.state_history[self.view_index]
        return self._get_legal_dests_for_state(from_sq, state)

    def _get_legal_dests_for_state(self, from_sq, state):
        import numpy as np
        # Handle both pgx State and GameState
        gstate = state.game_state if hasattr(state, 'game_state') else state
        
        mask = np.array(gstate.legal_action_mask)
        dests = []
        indices = np.where(mask)[0]
        for l_idx in indices.tolist():
            f_sq = l_idx // 32 # 32 = ACTION_PLANES for 9x9
            if f_sq == from_sq:
                # Use FROM_PLANE for decoding
                from ..tablut_jax import FROM_PLANE
                t_sq = int(FROM_PLANE[f_sq, l_idx % 32])
                dests.append(t_sq)
        return dests

    def _execute_move(self, from_sq, to_sq):
        state = self.state_history[self.view_index][0]
        gstate = state.game_state if hasattr(state, 'game_state') else state
        valid = jnp.where(gstate.legal_action_mask)[0]
        action_label = -1
        for l in valid.tolist():
            a = Action.from_label(l)
            if int(a.from_sq) == from_sq and int(a.to_sq) == to_sq: action_label = l; break
        if action_label != -1:
            # Evaluate user move and store feedback
            txt, color = self.engine.evaluate_user_move(state, action_label)
            self.move_feedback = txt
            self.move_feedback_color = color
            self.move_feedback_timer = 2.5  # Display for 2.5 seconds
            
            # Animation
            p = int(gstate.board[from_sq])
            self.animations.append(MoveAnimation(from_sq, to_sq, p, turn=int(gstate.color)))
            
            step_key, self.engine.key_env = jax.random.split(self.engine.key_env)
            self.engine.state = self.engine.step_fn(self.engine.state, action_label, step_key)
            self.engine.game_state = self.engine.state.game_state
            uci = self.engine._sq_to_uci(from_sq) + self.engine._sq_to_uci(to_sq)
            self.move_history.append(uci)
            self.state_history.append((self.engine.state, (from_sq, to_sq)))
            self.view_index = len(self.state_history) - 1
            self._check_game_over()

    def _check_game_over(self):
        if not self.game_over and Game.is_terminal(self.engine.game_state):
            self.game_over = True; rew = Game.rewards(self.engine.game_state)
            if rew[0] > 0: self.game_over_msg = "ATTACKERS WIN!"
            elif rew[1] > 0: self.game_over_msg = "DEFENDERS WIN!"
            else: self.game_over_msg = "DRAW"

    def run(self):
        try:
            while self.running:
                dt = self.clock.tick(FPS) / 1000.0
                if self.mode and self.mode != "tutorial" and not self.game_over:
                    ai_c = getattr(self.engine, 'ai_color', 0)
                    if self.ai_thinking:
                        if ai_c == -1: self.attacker_time -= dt
                        else: self.defender_time -= dt
                    else:
                        turn = int(self.engine.game_state.color)
                        if turn == -1: self.attacker_time -= dt
                        else: self.defender_time -= dt
                    if self.attacker_time <= 0: self.attacker_time = 0; self.game_over = True; self.game_over_msg = "DEFENDERS WIN (Timeout)!"
                    elif self.defender_time <= 0: self.defender_time = 0; self.game_over = True; self.game_over_msg = "ATTACKERS WIN (Timeout)!"

                if self.mode == "ai" and not self.game_over:
                    turn = int(self.engine.game_state.color); ai_c = getattr(self.engine, 'ai_color', 0)
                    if turn == ai_c and not self.ai_thinking:
                        self.ai_thinking = True
                        self.ai_start_time = pygame.time.get_ticks()
                        def ai_fn():
                            try:
                                uci = self.engine.make_ai_move(); self.ai_move_result = uci; self.ai_thinking = False
                            except Exception as e:
                                print(f"AI ERROR: {e}")
                                self.ai_thinking = False
                        threading.Thread(target=ai_fn, daemon=True).start()
                    if self.ai_move_result:
                        self.ai_moves_made = getattr(self, 'ai_moves_made', 0) + 1
                        if getattr(self, 'ai_start_time', None) is not None:
                            self.last_ai_duration = (pygame.time.get_ticks() - self.ai_start_time) / 1000.0
                        self.ai_start_time = None
                        uci = self.ai_move_result; self.ai_move_result = None
                        def uci_to_idx(s): return (int(s[1:]) - 1) * BOARD_EDGE + (ord(s[0]) - ord('a'))
                        f, t = uci_to_idx(uci[:2]), uci_to_idx(uci[2:])
                        # Animation
                        state = self.state_history[self.view_index][0]
                        gstate = state.game_state if hasattr(state, 'game_state') else state
                        p = int(gstate.board[f])
                        self.animations.append(MoveAnimation(f, t, p, turn=int(gstate.color)))
                        self.move_history.append(uci); self.state_history.append((self.engine.state, (f, t)))
                        if self.view_index == len(self.state_history) - 2: self.view_index = len(self.state_history) - 1
                        self._check_game_over()

                for event in pygame.event.get():
                    if event.type == pygame.QUIT: self.running = False
                    elif event.type == pygame.VIDEORESIZE: self.screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
                    elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        bx, by = self._to_base_coords(event.pos)
                        if self.mode is None:
                            for key, rect in self.menu_timer_btns.items():
                                if rect.collidepoint(bx, by):
                                    if key == "plus": self.timer_initial += 60
                                    elif key == "minus" and self.timer_initial > 60: self.timer_initial -= 60
                            for key, rect in self.menu_btn_rects.items():
                                if rect.collidepoint(bx, by):
                                    if key == "ai_atk": self.engine.ai_color = 1; self.mode = "ai"; self._reset_game()
                                    elif key == "ai_def": self.engine.ai_color = -1; self.mode = "ai"; self._reset_game()
                                    elif key == "pvp": self.mode = "pvp"; self._reset_game()
                                    elif key == "tutorial": self.mode = "tutorial"; self.tutorial_idx = 0; self.tutorial_anim = None; self.tutorial_delay = 0; self.tutorial_seq_idx = 0; self.tutorial_show_win = False
                                    elif key == "quit": self.running = False
                        elif self.mode == "tutorial":
                            for key, rect in self.tutorial_btn_rects.items():
                                if rect.collidepoint(bx, by):
                                    if key == "next": 
                                        self.tutorial_idx += 1; self.tutorial_anim = None; self.tutorial_delay = 0; self.tutorial_seq_idx = 0; self.tutorial_show_win = False
                                    elif key == "prev": 
                                        self.tutorial_idx -= 1; self.tutorial_anim = None; self.tutorial_delay = 0; self.tutorial_seq_idx = 0; self.tutorial_show_win = False
                                    elif key in ("finish", "quit"): 
                                        self._to_menu()
                                    break
                        else:
                            found_btn = False
                            for key, rect in getattr(self, 'btn_rects', {}).items():
                                if rect.collidepoint(bx, by):
                                    found_btn = True
                                    if key == "reset": self._reset_game()
                                    elif key == "menu": self._to_menu()
                                    elif key == "start": self.view_index = 0
                                    elif key == "prev": self.view_index = max(0, self.view_index - 1)
                                    elif key == "next": self.view_index = min(len(self.state_history) - 1, self.view_index + 1)
                                    elif key == "end": self.view_index = len(self.state_history) - 1
                                    break
                            if not found_btn:
                                if self.mode == "pvp" or (self.mode == "ai" and int(self.state_history[self.view_index][0].game_state.color) != getattr(self.engine, 'ai_color', 0)):
                                    self.handle_board_click(event.pos)
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_r: self._reset_game()
                        elif event.key == pygame.K_ESCAPE: self._to_menu()
                        elif event.key == pygame.K_LEFT:
                            if self.mode == "tutorial": 
                                self.tutorial_idx = max(0, self.tutorial_idx - 1); self.tutorial_anim = None; self.tutorial_delay = 0; self.tutorial_seq_idx = 0; self.tutorial_show_win = False
                            else: self.view_index = max(0, self.view_index - 1)
                        elif event.key == pygame.K_RIGHT:
                            if self.mode == "tutorial": 
                                self.tutorial_idx = min(len(self.tutorial_steps) - 1, self.tutorial_idx + 1); self.tutorial_anim = None; self.tutorial_delay = 0; self.tutorial_seq_idx = 0; self.tutorial_show_win = False
                            else: self.view_index = min(len(self.state_history) - 1, self.view_index + 1)
                self.draw(dt); pygame.display.flip()
        except Exception as e:
            print(f"CRITICAL UI ERROR: {e}")
            import traceback
            traceback.print_exc()
        finally:
            pygame.quit()
            sys.exit()
