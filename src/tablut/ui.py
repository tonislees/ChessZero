import pygame
import sys
import os
import jax
import jax.numpy as jnp
import threading
from .tablut_jax import Action, BOARD_EDGE, BOARD_SIZE, THRONE

# ─── Layout ───────────────────────────────────────────────────────────
UI_SCALE = 1.5

SIDEBAR_W = int(220 * UI_SCALE)
SIDEBAR_GAP = 10        # gap between Celtic border right edge and sidebar separator
CELL = 90
CELLS_PER_PATTERN = 2
PATTERN_H = CELL * CELLS_PER_PATTERN
BOARD_PX = CELL * BOARD_EDGE
FPS = 60

SHADOW_OFFSET = 5
SHADOW_ALPHA = 60
BOARD_GAP = 10          # gap between Celtic border and board edge
BOARD_CORNER_R = 10    # board corner radius in pixels

TIMER_INITIAL = 300.0   # 5 minutes in seconds

# ─── Nordic palette ──────────────────────────────────────────────────
BG_DARK       = (28, 25, 23)
BOARD_LIGHT   = (193, 165, 126)
BOARD_DARK    = (162, 132, 94)
BOARD_LINE    = (120, 90, 55)
GOLD          = (212, 175, 55)
GOLD_DIM      = (160, 130, 40)
BORDER_TINT   = (75, 68, 55)   # dim muted tone for celtic border pattern
CREAM         = (235, 225, 205)
ATK_TINT      = (235, 225, 205)
DEF_TINT      = (45, 45, 50)
KING_TINT     = (45, 45, 50)
HIGHLIGHT     = (90, 180, 90, 120)
SELECTED_CLR  = (255, 210, 60, 180)
MENU_BTN      = (58, 50, 42)
MENU_BTN_HOV  = (80, 68, 55)
MENU_TEXT      = (235, 225, 205)
STATUS_ATK    = (100, 160, 210)
STATUS_DEF    = (190, 80, 70)
LAST_MOVE_CLR = (255, 210, 60, 50)

CORNERS = [0, BOARD_EDGE - 1, BOARD_SIZE - BOARD_EDGE, BOARD_SIZE - 1]
SPECIAL_SQUARES = CORNERS + [THRONE]


def _asset_path(name):
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
    return os.path.join(base, name)


def _load_img(path, size):
    img = pygame.image.load(path).convert_alpha()
    w, h = img.get_size()
    scale = size / max(w, h)
    img = pygame.transform.smoothscale(img, (int(w * scale), int(h * scale)))
    return img


def _make_shadow(src):
    w, h = src.get_size()
    shadow = pygame.Surface((w, h), pygame.SRCALPHA)
    shadow.blit(src, (0, 0))
    arr_rgb = pygame.surfarray.pixels3d(shadow)
    arr_rgb[:, :, :] = 0
    del arr_rgb
    arr_a = pygame.surfarray.pixels_alpha(shadow)
    arr_a[:, :] = (arr_a[:, :].astype(int) * SHADOW_ALPHA // 255).clip(0, 255)
    del arr_a
    return shadow


class TablutUI:
    def __init__(self, logic_engine):
        pygame.init()
        self.clock = pygame.time.Clock()
        self.engine = logic_engine

        # ── Warmup JAX Engine ────────────────────────────────────────
        # We run a dummy AI move to ensure all JIT (including MCTS/Model) is compiled
        # before the user starts their game.
        print("Initializing JAX engine & pre-compiling MCTS/Model...")
        self.engine.make_ai_move() 
        self.engine.reset() # Reset to clean state after warmup
        print("Warmup complete.")

        # Temporary display for convert_alpha()
        self.screen = pygame.display.set_mode((1, 1))

        # ── Load assets ──────────────────────────────────────────────
        piece_size = int(CELL * 0.85)
        self.img_attacker = _load_img(_asset_path("Tablut_attacker.png"), piece_size)
        self.img_defender = _load_img(_asset_path("Tablut_defender.png"), piece_size)
        self.img_king     = _load_img(_asset_path("Tablut_king.png"), piece_size)
        self.shadow_attacker = _make_shadow(self.img_attacker)
        self.shadow_defender = _make_shadow(self.img_defender)
        self.shadow_king     = _make_shadow(self.img_king)

        knot = pygame.image.load(_asset_path("Celtic_knot.png")).convert_alpha()
        self.img_knot_raw = knot
        knot_board = pygame.transform.smoothscale(knot, (CELL - 4, CELL - 4))
        knot_board.set_alpha(45)
        self.img_knot = knot_board

        mid = pygame.image.load(_asset_path("pattern_middle.png")).convert_alpha()
        tip = pygame.image.load(_asset_path("pattern_tip.png")).convert_alpha()
        for img in [mid, tip]:
            arr = pygame.surfarray.pixels3d(img)
            for c, v in enumerate(BORDER_TINT):
                arr[:, :, c] = (arr[:, :, c].astype(int) * v // 255).clip(0, 255)
            del arr
        self.border_mid = mid
        self.border_tip = tip

        # ── Layout from pattern ──────────────────────────────────────
        img_w, img_h = self.border_mid.get_size()
        self.border_w = round(img_w * PATTERN_H / img_h)
        self.total_board = BOARD_PX + 2 * self.border_w + 2 * BOARD_GAP
        self.base_w = self.total_board + SIDEBAR_GAP + SIDEBAR_W
        self.base_h = self.total_board

        # ── Window ───────────────────────────────────────────────────
        icon = pygame.transform.smoothscale(self.img_knot_raw, (32, 32))
        pygame.display.set_icon(icon)
        pygame.display.set_caption("Hnefatafl · Viking Board Game")
        self.screen = pygame.display.set_mode(
            (self.base_w, self.base_h), pygame.RESIZABLE)

        # Fixed-res render target — blitted to screen at offset, never scaled
        self.render_surf = pygame.Surface((self.base_w, self.base_h))
        self.ox = 0
        self.oy = 0

        # Fonts
        self.font_title  = pygame.font.SysFont("Georgia", int(28 * UI_SCALE), bold=True)
        self.font_btn    = pygame.font.SysFont("Georgia", int(18 * UI_SCALE))
        self.font_small  = pygame.font.SysFont("Georgia", int(14 * UI_SCALE))
        self.font_coord  = pygame.font.SysFont("Consolas", int(12 * UI_SCALE))
        self.font_status = pygame.font.SysFont("Georgia", int(16 * UI_SCALE), bold=True)
        self.font_over   = pygame.font.SysFont("Georgia", int(36 * UI_SCALE), bold=True)
        self.font_timer  = pygame.font.SysFont("Consolas", int(22 * UI_SCALE), bold=True)

        # ── State ────────────────────────────────────────────────────
        self.attacker_time = TIMER_INITIAL
        self.defender_time = TIMER_INITIAL
        self.selected_sq = None
        self.valid_moves_for_selected = []
        self.move_history = []
        self.state_history = []  # List of (env_state, last_move_tuple)
        self.view_index = -1
        self.last_move = None
        self.running = True
        self.game_over = False
        self.game_over_msg = ""
        self.mode = None
        self.ai_thinking = False
        self.ai_move_result = None

        self._reset_game()
        self._render_board_base()

    # ─── Board + border pre-render ───────────────────────────────────
    def _render_board_base(self):
        self.board_surf = pygame.Surface((BOARD_PX, BOARD_PX), pygame.SRCALPHA)
        self.board_surf.fill((*BOARD_LIGHT, 255))

        for r in range(BOARD_EDGE):
            for c in range(BOARD_EDGE):
                if (r + c) % 2 == 0:
                    pygame.draw.rect(self.board_surf, BOARD_DARK,
                                     (c * CELL, r * CELL, CELL, CELL))

        # ── Throne & corner square decorations ───────────────────────────
        for sq in SPECIAL_SQUARES:
            row, col = divmod(sq, BOARD_EDGE)
            ui_r = BOARD_EDGE - 1 - row
            x, y = col * CELL, ui_r * CELL
            ccx, ccy = x + CELL // 2, y + CELL // 2

            # ── Special Square Style ─────────────────────────────────────────
            # Both Throne and Corners now share a base dark square look
            ov = pygame.Surface((CELL, CELL), pygame.SRCALPHA)
            ov.fill((*BOARD_DARK, 160)) 
            self.board_surf.blit(ov, (x, y))
            
            # Common "Etched" Patterns (Dark Brown)
            # Double rings
            pygame.draw.circle(self.board_surf, (*BOARD_LINE, 200), (ccx, ccy), CELL // 2 - 8, 1)
            pygame.draw.circle(self.board_surf, (*BOARD_LINE, 120), (ccx, ccy), CELL // 2 - 14, 1)
            
            # Cardinal markers
            for angle in range(0, 360, 90):
                import math
                rad = math.radians(angle)
                dist = CELL // 2 - 8
                px, py = ccx + math.cos(rad) * dist, ccy + math.sin(rad) * dist
                pygame.draw.circle(self.board_surf, (*BOARD_LINE, 220), (int(px), int(py)), 3)

            # Central knot (tinted to dark brown)
            ks = CELL // 2 + 10
            kimg = pygame.transform.smoothscale(self.img_knot_raw, (ks, ks))
            arr = pygame.surfarray.pixels3d(kimg)
            for c, v in enumerate(BOARD_LINE):
                arr[:, :, c] = (arr[:, :, c].astype(int) * v // 255).clip(0, 255)
            del arr
            kimg.set_alpha(180) 
            self.board_surf.blit(kimg, (ccx - ks // 2, ccy - ks // 2))

            if sq != THRONE:
                # ── Additional Corner Marking ─────────────────────────
                # Sharp corner brackets to distinguish escape squares
                # Added a small margin from the edge for better visibility
                margin = 1
                arm = CELL // 4
                cx0 = (x + margin) if col == 0 else (x + CELL - 1 - margin)
                cy0 = (y + margin) if ui_r == 0 else (y + CELL - 1 - margin)
                ddx = 1 if col == 0 else -1
                ddy = 1 if ui_r == 0 else -1
                
                pygame.draw.line(self.board_surf, (*BOARD_LINE, 255), (cx0, cy0), (cx0 + ddx * arm, cy0), 2)
                pygame.draw.line(self.board_surf, (*BOARD_LINE, 255), (cx0, cy0), (cx0, cy0 + ddy * arm), 2)

        for i in range(BOARD_EDGE + 1):
            pygame.draw.line(self.board_surf, BOARD_LINE,
                             (0, i * CELL), (BOARD_PX, i * CELL), 1)
            pygame.draw.line(self.board_surf, BOARD_LINE,
                             (i * CELL, 0), (i * CELL, BOARD_PX), 1)

        # Round board corners via alpha mask
        _clip = pygame.Surface((BOARD_PX, BOARD_PX), pygame.SRCALPHA)
        _clip.fill((0, 0, 0, 0))
        pygame.draw.rect(_clip, (255, 255, 255, 255),
                         _clip.get_rect(), border_radius=BOARD_CORNER_R)
        _ba = pygame.surfarray.pixels_alpha(self.board_surf)
        _ma = pygame.surfarray.pixels_alpha(_clip)
        _ba[:] = (_ba.astype(int) * _ma // 255).clip(0, 255).astype(_ba.dtype)
        del _ba, _ma

        file_letters = "abcdefghi"
        for c in range(BOARD_EDGE):
            color = BOARD_LINE
            lbl = self.font_coord.render(file_letters[c], True, color)
            # Bottom-left of each square in the bottom row (ui_r = BOARD_EDGE-1)
            self.board_surf.blit(lbl, (c * CELL + 3, (BOARD_EDGE - 1) * CELL + CELL - lbl.get_height() - 3))
        for r in range(BOARD_EDGE):
            color = BOARD_LINE
            ui_r = BOARD_EDGE - 1 - r
            lbl = self.font_coord.render(str(r + 1), True, color)
            self.board_surf.blit(lbl, (3, ui_r * CELL + 2))

        # ── Border frame ─────────────────────────────────────────────
        B = self.border_w
        tw, th = self.border_tip.get_size()
        tip_h = round(th * B / tw)
        tip_v = pygame.transform.smoothscale(self.border_tip, (B, tip_h))

        tip_h_rot = pygame.transform.rotate(tip_v, 90)
        tip_h_len = tip_h_rot.get_width()   # == tip_h for a square border

        # Scale mid-tile so exactly N whole tiles fit between the two tips —
        # no half-tile cut-off at either end.
        raw_strip = BOARD_PX + 2 * BOARD_GAP - 2 * tip_h
        n_tiles = max(1, round(raw_strip / PATTERN_H))
        tile_size = round(raw_strip / n_tiles)

        mid_v = pygame.transform.smoothscale(self.border_mid, (B, tile_size))
        mid_h = pygame.transform.rotate(mid_v, -90)

        vert_strip_len = n_tiles * tile_size
        horiz_strip_len = vert_strip_len

        def tile_vert(tile, length):
            s = pygame.Surface((B, length), pygame.SRCALPHA)
            y = 0
            while y < length:
                s.blit(tile, (0, y))
                y += tile.get_height()
            return s

        def tile_horiz(tile, length):
            s = pygame.Surface((length, B), pygame.SRCALPHA)
            x = 0
            while x < length:
                s.blit(tile, (x, 0))
                x += tile.get_width()
            return s

        self.border_surf = pygame.Surface(
            (self.total_board, self.total_board), pygame.SRCALPHA)

        # Left edge: tip ─ strip ─ tip(180°)
        self.border_surf.blit(tip_v, (0, B))
        self.border_surf.blit(tile_vert(mid_v, vert_strip_len), (0, B + tip_h))
        self.border_surf.blit(
            pygame.transform.flip(tip_v, True, True),
            (0, self.total_board - B - tip_h))

        # Right edge (mirrored)
        mid_v_r = pygame.transform.flip(mid_v, True, False)
        tip_v_r = pygame.transform.flip(tip_v, True, False)
        self.border_surf.blit(tip_v_r, (self.total_board - B, B))
        self.border_surf.blit(tile_vert(mid_v_r, vert_strip_len),
                              (self.total_board - B, B + tip_h))
        self.border_surf.blit(
            pygame.transform.flip(tip_v_r, True, True),
            (self.total_board - B, self.total_board - B - tip_h))

        # Top edge: tip ─ strip ─ tip(180°)
        self.border_surf.blit(tip_h_rot, (B, 0))
        self.border_surf.blit(tile_horiz(mid_h, horiz_strip_len),
                              (B + tip_h_len, 0))
        self.border_surf.blit(
            pygame.transform.flip(tip_h_rot, True, True),
            (self.total_board - B - tip_h_len, 0))

        # Bottom edge (flipped)
        mid_h_b = pygame.transform.flip(mid_h, False, True)
        tip_h_bot_l = pygame.transform.flip(tip_h_rot, False, True)
        self.border_surf.blit(tip_h_bot_l, (B, self.total_board - B))
        self.border_surf.blit(tile_horiz(mid_h_b, horiz_strip_len),
                              (B + tip_h_len, self.total_board - B))
        self.border_surf.blit(
            pygame.transform.flip(tip_h_rot, True, False),
            (self.total_board - B - tip_h_len, self.total_board - B))

        # Corner squares
        for cx, cy in [(0, 0), (self.total_board - B, 0),
                       (0, self.total_board - B),
                       (self.total_board - B, self.total_board - B)]:
            pygame.draw.rect(self.border_surf, (*BG_DARK, 255), (cx, cy, B, B))

    # ─── Helpers ─────────────────────────────────────────────────────
    def get_piece_at(self, idx):
        state, _ = self.state_history[self.view_index]
        return int(state.game_state.board[idx])

    def get_legal_destinations(self, from_sq):
        state, _ = self.state_history[self.view_index]
        legal_mask = state.legal_action_mask
        destinations = []
        for label_idx in jnp.where(legal_mask)[0].tolist():
            action = Action.from_label(label_idx)
            if int(action.from_sq) == from_sq:
                destinations.append(int(action.to_sq))
        return destinations

    def _to_base_coords(self, screen_pos):
        return screen_pos[0] - self.ox, screen_pos[1] - self.oy

    def _update_offset(self):
        win_w, win_h = self.screen.get_size()
        self.ox = max(0, (win_w - self.base_w) // 2)
        self.oy = max(0, (win_h - self.base_h) // 2)

    # ─── Drawing ─────────────────────────────────────────────────────
    def draw(self):
        self.screen.fill(BG_DARK)
        self.render_surf.fill(BG_DARK)
        self._update_offset()

        if self.mode is None:
            self._draw_menu()
            self.screen.blit(self.render_surf, (self.ox, self.oy))
            return

        B = self.border_w
        BO = B + BOARD_GAP   # board origin (border + gap)
        S = self.render_surf

        S.blit(self.board_surf, (BO, BO))

        current_state, v_last_move = self.state_history[self.view_index]

        # ── Last-move brackets ───────────────────────────────────────────
        if v_last_move:
            arm = max(6, CELL // 7)
            lm_surf = pygame.Surface((CELL, CELL), pygame.SRCALPHA)
            lm_surf.fill((255, 195, 50, 30))
            bc = (255, 195, 50, 150)
            for (px, py), dx, dy in [
                ((1, 1), 1, 1), ((CELL - 2, 1), -1, 1),
                ((1, CELL - 2), 1, -1), ((CELL - 2, CELL - 2), -1, -1)
            ]:
                pygame.draw.line(lm_surf, bc, (px, py), (px + dx * arm, py), 2)
                pygame.draw.line(lm_surf, bc, (px, py), (px, py + dy * arm), 2)
            for sq in v_last_move:
                row, col = divmod(sq, BOARD_EDGE)
                ui_r = BOARD_EDGE - 1 - row
                S.blit(lm_surf, (BO + col * CELL, BO + ui_r * CELL))

        # ── Selected-square brackets ─────────────────────────────────────
        # Only show selection if we are at the latest state
        if self.view_index == len(self.state_history) - 1 and self.selected_sq is not None:
            arm = max(8, CELL // 6)
            sel_surf = pygame.Surface((CELL, CELL), pygame.SRCALPHA)
            sel_surf.fill((255, 210, 60, 45))
            pygame.draw.rect(sel_surf, (255, 220, 70, 160), sel_surf.get_rect(), 2)
            sc = (255, 235, 90, 245)
            for (px, py), dx, dy in [
                ((1, 1), 1, 1), ((CELL - 2, 1), -1, 1),
                ((1, CELL - 2), 1, -1), ((CELL - 2, CELL - 2), -1, -1)
            ]:
                pygame.draw.line(sel_surf, sc, (px, py), (px + dx * arm, py), 3)
                pygame.draw.line(sel_surf, sc, (px, py), (px, py + dy * arm), 3)
            row, col = divmod(self.selected_sq, BOARD_EDGE)
            ui_r = BOARD_EDGE - 1 - row
            S.blit(sel_surf, (BO + col * CELL, BO + ui_r * CELL))

            # ── Valid-move diamonds ───────────────────────────────────────
            half = max(10, CELL // 6)
            dot_surf = pygame.Surface((CELL, CELL), pygame.SRCALPHA)
            ccx, ccy = CELL // 2, CELL // 2
            pts = [(ccx, ccy - half), (ccx + half, ccy),
                   (ccx, ccy + half), (ccx - half, ccy)]
            pygame.draw.polygon(dot_surf, (100, 215, 125, 210), pts, 2)
            pygame.draw.circle(dot_surf, (100, 215, 125, 230), (ccx, ccy), 3)
            for dest in self.valid_moves_for_selected:
                dr, dc = divmod(dest, BOARD_EDGE)
                dui_r = BOARD_EDGE - 1 - dr
                S.blit(dot_surf, (BO + dc * CELL, BO + dui_r * CELL))

        # Pieces
        current_turn = int(current_state.game_state.color)
        for idx in range(BOARD_SIZE):
            piece = int(current_state.game_state.board[idx])
            if piece == 0:
                continue
            row, col = divmod(idx, BOARD_EDGE)
            ui_r = BOARD_EDGE - 1 - row
            cx = BO + col * CELL + CELL // 2
            cy = BO + ui_r * CELL + CELL // 2

            is_attacker = ((piece > 0 and current_turn == -1)
                           or (piece < 0 and current_turn == 1))
            if abs(piece) == 2:
                img, shadow = self.img_king, self.shadow_king
            elif is_attacker:
                img, shadow = self.img_attacker, self.shadow_attacker
            else:
                img, shadow = self.img_defender, self.shadow_defender

            S.blit(shadow, (cx - shadow.get_width() // 2 + SHADOW_OFFSET,
                            cy - shadow.get_height() // 2 + SHADOW_OFFSET))
            S.blit(img, (cx - img.get_width() // 2, cy - img.get_height() // 2))

        S.blit(self.border_surf, (0, 0))
        self._draw_sidebar()

        self.screen.blit(self.render_surf, (self.ox, self.oy))

        if self.game_over:
            self._draw_game_over()

    def _draw_sidebar(self):
        S = self.render_surf
        sx = self.total_board + SIDEBAR_GAP
        pygame.draw.line(S, GOLD_DIM, (sx, 0), (sx, self.base_h), 2)

        title = self.font_title.render("TABLUT", True, GOLD)
        S.blit(title, (sx + (SIDEBAR_W - title.get_width()) // 2, 20))

        current_turn = int(self.engine.game_state.color)
        turn_text = "Attacker's Turn" if current_turn == -1 else "Defender's Turn"
        turn_color = STATUS_ATK if current_turn == -1 else STATUS_DEF
        turn_lbl = self.font_status.render(turn_text, True, turn_color)
        S.blit(turn_lbl, (sx + (SIDEBAR_W - turn_lbl.get_width()) // 2, int(80 * UI_SCALE)))

        # ── Timers ───────────────────────────────────────────────────
        def fmt_time(t):
            m, s = divmod(int(t), 60)
            return f"{m:02d}:{s:02d}"

        atk_t = self.font_timer.render(fmt_time(self.attacker_time), True, STATUS_ATK)
        def_t = self.font_timer.render(fmt_time(self.defender_time), True, STATUS_DEF)
        
        S.blit(atk_t, (sx + 30, int(115 * UI_SCALE)))
        S.blit(def_t, (sx + SIDEBAR_W - 30 - def_t.get_width(), int(115 * UI_SCALE)))

        mode_text = "vs AI" if self.mode == "ai" else "PvP"
        mode_lbl = self.font_small.render(f"Mode: {mode_text}", True, (140, 130, 115))
        S.blit(mode_lbl, (sx + (SIDEBAR_W - mode_lbl.get_width()) // 2, int(150 * UI_SCALE)))

        mc = self.font_small.render(f"Moves: {len(self.move_history)}", True, (140, 130, 115))
        S.blit(mc, (sx + (SIDEBAR_W - mc.get_width()) // 2, int(172 * UI_SCALE)))

        self.btn_rects = {}
        btn_y = int(210 * UI_SCALE)
        for label, key in [("Reset Game", "reset"), ("Back to Menu", "menu")]:
            rect = pygame.Rect(sx + int(20 * UI_SCALE), btn_y, SIDEBAR_W - int(40 * UI_SCALE), int(38 * UI_SCALE))
            self.btn_rects[key] = rect
            mx, my = self._to_base_coords(pygame.mouse.get_pos())
            color = MENU_BTN_HOV if rect.collidepoint(mx, my) else MENU_BTN
            pygame.draw.rect(S, color, rect, border_radius=6)
            pygame.draw.rect(S, GOLD_DIM, rect, 1, border_radius=6)
            lbl = self.font_btn.render(label, True, MENU_TEXT)
            S.blit(lbl, (rect.centerx - lbl.get_width() // 2,
                         rect.centery - lbl.get_height() // 2))
            btn_y += int(50 * UI_SCALE)

        # ── Navigation Buttons ───────────────────────────────────────
        nav_y = btn_y + int(10 * UI_SCALE)
        nav_w = (SIDEBAR_W - int(50 * UI_SCALE)) // 4
        nav_btns = [("|<", "start"), ("<", "prev"), (">", "next"), (">|", "end")]
        for i, (label, key) in enumerate(nav_btns):
            rect = pygame.Rect(sx + int(25 * UI_SCALE) + i * nav_w, nav_y, nav_w - 4, int(30 * UI_SCALE))
            self.btn_rects[key] = rect
            mx, my = self._to_base_coords(pygame.mouse.get_pos())
            color = MENU_BTN_HOV if rect.collidepoint(mx, my) else MENU_BTN
            pygame.draw.rect(S, color, rect, border_radius=4)
            pygame.draw.rect(S, GOLD_DIM, rect, 1, border_radius=4)
            lbl = self.font_small.render(label, True, MENU_TEXT)
            S.blit(lbl, (rect.centerx - lbl.get_width() // 2,
                         rect.centery - lbl.get_height() // 2))
        
        hist_y = nav_y + int(45 * UI_SCALE)
        hist_title = self.font_small.render("─ Move Log ─", True, GOLD_DIM)
        S.blit(hist_title, (sx + (SIDEBAR_W - hist_title.get_width()) // 2, hist_y))
        hist_y += int(25 * UI_SCALE)

        # ── Scrollable Move Log ──────────────────────────────────────
        # Show a window of moves around the current view_index
        log_height = 10
        total_items = len(self.move_history) + 1
        start_idx = max(0, min(self.view_index - log_height // 2, total_items - log_height))
        end_idx = min(start_idx + log_height, total_items)
        
        for i in range(start_idx, end_idx):
            is_current = (i == self.view_index)
            color = GOLD if is_current else (130, 120, 105)
            prefix = ">" if is_current else " "
            
            if i == 0:
                mv_text = "(Initial State)"
            else:
                mv_text = self.move_history[i - 1]
                
            txt = self.font_small.render(f"{prefix}{i:>3}. {mv_text}", True, color)
            S.blit(txt, (sx + int(25 * UI_SCALE), hist_y))
            hist_y += int(18 * UI_SCALE)

        hint = self.font_small.render("R = Reset", True, (90, 80, 70))
        S.blit(hint, (sx + (SIDEBAR_W - hint.get_width()) // 2, self.base_h - int(25 * UI_SCALE)))

    def _draw_menu(self):
        S = self.render_surf
        cx, cy = self.base_w // 2, self.base_h // 2
        
        # Title
        title = self.font_title.render("Tablut", True, GOLD)
        S.blit(title, (cx - title.get_width() // 2, cy - int(200 * UI_SCALE)))
        pygame.draw.line(S, GOLD_DIM, (cx - 120, cy - int(162 * UI_SCALE)), (cx + 120, cy - int(162 * UI_SCALE)), 1)

        # ── Timer Setup Panel ────────────────────────────────────────
        # A centered panel for time adjustment
        pw, ph = int(300 * UI_SCALE), int(60 * UI_SCALE)
        px, py = cx - pw // 2, cy - int(145 * UI_SCALE)
        
        # Panel Background
        pygame.draw.rect(S, (20, 15, 10), (px, py, pw, ph), border_radius=10)
        pygame.draw.rect(S, GOLD_DIM, (px, py, pw, ph), 1, border_radius=10)
        
        time_text = f"Time: {int(TIMER_INITIAL // 60)} min"
        lbl = self.font_status.render(time_text, True, GOLD_DIM)
        S.blit(lbl, (cx - lbl.get_width() // 2, py + (ph - lbl.get_height()) // 2))
        
        self.menu_timer_btns = {}
        # Minus Button
        m_rect = pygame.Rect(px + 15, py + (ph - int(34 * UI_SCALE)) // 2, int(40 * UI_SCALE), int(34 * UI_SCALE))
        self.menu_timer_btns["minus"] = m_rect
        # Plus Button
        p_rect = pygame.Rect(px + pw - int(40 * UI_SCALE) - 15, py + (ph - int(34 * UI_SCALE)) // 2, int(40 * UI_SCALE), int(34 * UI_SCALE))
        self.menu_timer_btns["plus"] = p_rect
        
        for key, rect in self.menu_timer_btns.items():
            mx, my = self._to_base_coords(pygame.mouse.get_pos())
            color = MENU_BTN_HOV if rect.collidepoint(mx, my) else MENU_BTN
            pygame.draw.rect(S, color, rect, border_radius=6)
            pygame.draw.rect(S, GOLD_DIM, rect, 1, border_radius=6)
            txt = "-" if key == "minus" else "+"
            btn_lbl = self.font_btn.render(txt, True, MENU_TEXT)
            S.blit(btn_lbl, (rect.centerx - btn_lbl.get_width() // 2, rect.centery - btn_lbl.get_height() // 2))

        # ── Game Mode Buttons ────────────────────────────────────────
        self.menu_btn_rects = {}
        btn_data = [
            ("Play vs AI (Attacker)", "ai_atk"),
            ("Play vs AI (Defender)", "ai_def"),
            ("Player vs Player", "pvp"),
        ]
        btn_y = cy - int(60 * UI_SCALE)
        for label, key in btn_data:
            rect = pygame.Rect(cx - int(140 * UI_SCALE), btn_y, int(280 * UI_SCALE), int(44 * UI_SCALE))
            self.menu_btn_rects[key] = rect
            mx, my = self._to_base_coords(pygame.mouse.get_pos())
            color = MENU_BTN_HOV if rect.collidepoint(mx, my) else MENU_BTN
            pygame.draw.rect(S, color, rect, border_radius=8)
            pygame.draw.rect(S, GOLD_DIM, rect, 1, border_radius=8)
            lbl = self.font_btn.render(label, True, MENU_TEXT)
            S.blit(lbl, (rect.centerx - lbl.get_width() // 2,
                         rect.centery - lbl.get_height() // 2))
            btn_y += int(56 * UI_SCALE)

    def _draw_game_over(self):
        sw, sh = self.screen.get_size()
        overlay = pygame.Surface((sw, sh), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 140))
        self.screen.blit(overlay, (0, 0))
        
        text = self.font_over.render(self.game_over_msg, True, GOLD)
        rect = text.get_rect(center=(sw // 2, sh // 2))
        bg_rect = rect.inflate(40, 20)
        pygame.draw.rect(self.screen, BG_DARK, bg_rect, border_radius=10)
        pygame.draw.rect(self.screen, GOLD_DIM, bg_rect, 2, border_radius=10)
        self.screen.blit(text, rect)
        
        hint = self.font_small.render("Press R to reset  ·  ESC for menu", True, CREAM)
        self.screen.blit(hint, (sw // 2 - hint.get_width() // 2,
                      sh // 2 + int(35 * UI_SCALE)))

    # ─── Game logic ──────────────────────────────────────────────────
    def check_game_over(self):
        if not self.game_over and self.engine.env.game.is_terminal(self.engine.game_state):
            self.game_over = True
            rewards = self.engine.env.game.rewards(self.engine.game_state)
            if rewards[0] > 0:
                self.game_over_msg = "ATTACKERS WIN!"
            elif rewards[1] > 0:
                self.game_over_msg = "DEFENDERS WIN!"
            else:
                self.game_over_msg = "DRAW"

    def handle_board_click(self, pos):
        if self.game_over or self.mode is None:
            return
        if self.view_index != len(self.state_history) - 1:
            return

        x, y = self._to_base_coords(pos)
        x -= self.border_w + BOARD_GAP
        y -= self.border_w + BOARD_GAP
        if x < 0 or y < 0 or x >= BOARD_PX or y >= BOARD_PX:
            return

        col = int(x) // CELL
        ui_row = int(y) // CELL
        row = BOARD_EDGE - 1 - ui_row
        idx = row * BOARD_EDGE + col
        if not (0 <= idx < BOARD_SIZE):
            return

        if self.selected_sq is not None and idx in self.valid_moves_for_selected:
            self._execute_move(self.selected_sq, idx)
            self.selected_sq = None
            self.valid_moves_for_selected = []
            return

        piece = self.get_piece_at(idx)
        if piece > 0:
            self.selected_sq = idx
            self.valid_moves_for_selected = self.get_legal_destinations(idx)
        else:
            self.selected_sq = None
            self.valid_moves_for_selected = []

    def _execute_move(self, from_sq, to_sq):
        legal_mask = self.state_history[self.view_index][0].legal_action_mask
        valid_indices = jnp.where(legal_mask)[0]
        action_label = -1
        for label_idx in valid_indices.tolist():
            a = Action.from_label(label_idx)
            if int(a.from_sq) == from_sq and int(a.to_sq) == to_sq:
                action_label = label_idx
                break

        if action_label != -1:
            uci = self.engine._sq_to_uci(from_sq) + self.engine._sq_to_uci(to_sq)
            self.move_history.append(uci)
            
            step_key, self.engine.key_env = jax.random.split(self.engine.key_env)
            # Apply move to the LATEST state in engine
            self.engine.state = self.engine.step_fn(self.engine.state, action_label, step_key)
            self.engine.game_state = self.engine.state.game_state
            
            self.state_history.append((self.engine.state, (from_sq, to_sq)))
            self.view_index = len(self.state_history) - 1
            self.check_game_over()

    def _reset_game(self):
        if hasattr(self.engine, 'reset'):
            self.engine.reset()
        self.attacker_time = TIMER_INITIAL
        self.defender_time = TIMER_INITIAL
        self.game_over = False
        self.game_over_msg = ""
        self.move_history = []
        self.state_history = [(self.engine.state, None)]
        self.view_index = 0
        self.selected_sq = None
        self.valid_moves_for_selected = []

    def _to_menu(self):
        self._reset_game()
        self.mode = None

    # ─── Main loop ───────────────────────────────────────────────────
    def run(self):
        while self.running:
            dt_ms = self.clock.tick(FPS)
            dt = dt_ms / 1000.0

            if self.mode is not None and not self.game_over:
                # Decide whose clock to run
                ai_color = getattr(self.engine, 'ai_color', 0)
                if self.ai_thinking:
                    # AI is thinking, run AI clock
                    if ai_color == -1: self.attacker_time -= dt
                    else: self.defender_time -= dt
                else:
                    # No AI thinking, run current turn's clock (latest engine state)
                    current_turn = int(self.engine.game_state.color)
                    if current_turn == -1: self.attacker_time -= dt
                    else: self.defender_time -= dt
                
                if self.attacker_time <= 0:
                    self.attacker_time = 0
                    self.game_over = True
                    self.game_over_msg = "DEFENDERS WIN (Timeout)!"
                elif self.defender_time <= 0:
                    self.defender_time = 0
                    self.game_over = True
                    self.game_over_msg = "ATTACKERS WIN (Timeout)!"

            if self.mode == "ai" and not self.game_over:
                engine_turn = int(self.engine.game_state.color)
                ai_color = getattr(self.engine, 'ai_color', 0)
                
                # Start thinking if it is AI turn and not already thinking
                if engine_turn == ai_color and not self.ai_thinking:
                    self.ai_thinking = True
                    def ai_thread_fn():
                        uci = self.engine.make_ai_move()
                        self.ai_move_result = uci
                        self.ai_thinking = False
                    
                    threading.Thread(target=ai_thread_fn, daemon=True).start()
                
                # Process result whenever it arrives, regardless of turn or view
                if self.ai_move_result:
                    uci_move = self.ai_move_result
                    self.ai_move_result = None
                    
                    def uci_to_idx(s):
                        col = ord(s[0]) - ord('a')
                        row = int(s[1:]) - 1
                        return row * BOARD_EDGE + col
                    
                    lm_tuple = (uci_to_idx(uci_move[:2]), uci_to_idx(uci_move[2:]))
                    
                    self.move_history.append(uci_move)
                    self.state_history.append((self.engine.state, lm_tuple))
                    
                    # Only jump the view if we were already watching the latest state
                    if self.view_index == len(self.state_history) - 2:
                        self.view_index = len(self.state_history) - 1
                    
                    self.check_game_over()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.VIDEORESIZE:
                    self.screen = pygame.display.set_mode(
                        (event.w, event.h), pygame.RESIZABLE)
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    bx, by = self._to_base_coords(event.pos)
                    if self.mode is None:
                        # Timer adjustments
                        for key, rect in getattr(self, 'menu_timer_btns', {}).items():
                            if rect.collidepoint(bx, by):
                                global TIMER_INITIAL
                                if key == "plus":
                                    TIMER_INITIAL += 60
                                elif key == "minus" and TIMER_INITIAL > 60:
                                    TIMER_INITIAL -= 60
                        
                        # Menu buttons
                        for key, rect in getattr(self, 'menu_btn_rects', {}).items():
                            if rect.collidepoint(bx, by):
                                if key == "ai_atk":
                                    self.engine.ai_color = 1
                                    self.mode = "ai"
                                    self._reset_game()
                                elif key == "ai_def":
                                    self.engine.ai_color = -1
                                    self.mode = "ai"
                                    self._reset_game()
                                elif key == "pvp":
                                    self.mode = "pvp"
                                    self._reset_game()
                    else:
                        found_btn = False
                        for key, rect in getattr(self, 'btn_rects', {}).items():
                            if rect.collidepoint(bx, by):
                                found_btn = True
                                if key == "reset":
                                    self._reset_game()
                                elif key == "menu":
                                    self._to_menu()
                                elif key == "start":
                                    self.view_index = 0
                                elif key == "prev":
                                    self.view_index = max(0, self.view_index - 1)
                                elif key == "next":
                                    self.view_index = min(len(self.state_history) - 1, self.view_index + 1)
                                elif key == "end":
                                    self.view_index = len(self.state_history) - 1
                                break
                        
                        if not found_btn:
                            if self.mode == "pvp" or (
                                self.mode == "ai"
                                and int(self.state_history[self.view_index][0].game_state.color)
                                    != getattr(self.engine, 'ai_color', 0)
                            ):
                                self.handle_board_click(event.pos)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        self._reset_game()
                    elif event.key == pygame.K_ESCAPE:
                        self._to_menu()
                    elif event.key == pygame.K_LEFT:
                        self.view_index = max(0, self.view_index - 1)
                    elif event.key == pygame.K_RIGHT:
                        self.view_index = min(len(self.state_history) - 1, self.view_index + 1)
                    elif event.key == pygame.K_HOME:
                        self.view_index = 0
                    elif event.key == pygame.K_END:
                        self.view_index = len(self.state_history) - 1

            self.draw()
            pygame.display.flip()

        pygame.quit()
        sys.exit()