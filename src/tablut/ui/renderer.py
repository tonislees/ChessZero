import pygame
import math
from .constants import *

class Renderer:
    def __init__(self, assets, fonts):
        self.assets = assets
        self.fonts = fonts
        self.border_w = self._calc_border_w()
        self.total_board = BOARD_PX + 2 * self.border_w + 2 * BOARD_GAP
        self.base_w = self.total_board + SIDEBAR_GAP + SIDEBAR_W
        self.base_h = self.total_board
        self.board_surf = self._render_board_base()
        self.border_surf = self._render_border_frame()

    def _calc_border_w(self):
        img_w, img_h = self.assets.border_mid.get_size()
        return round(img_w * PATTERN_H / img_h)

    def _render_board_base(self):
        surf = pygame.Surface((BOARD_PX, BOARD_PX), pygame.SRCALPHA)
        surf.fill((*BOARD_LIGHT, 255))
        for r in range(BOARD_EDGE):
            for c in range(BOARD_EDGE):
                if (r + c) % 2 == 0:
                    pygame.draw.rect(surf, BOARD_DARK, (c * CELL, r * CELL, CELL, CELL))
        
        for sq in SPECIAL_SQUARES:
            row, col = divmod(sq, BOARD_EDGE)
            ui_r = BOARD_EDGE - 1 - row
            x, y = col * CELL, ui_r * CELL
            ccx, ccy = x + CELL // 2, y + CELL // 2
            ov = pygame.Surface((CELL, CELL), pygame.SRCALPHA)
            ov.fill((*BOARD_DARK, 160)) 
            surf.blit(ov, (x, y))
            pygame.draw.circle(surf, (*BOARD_LINE, 200), (ccx, ccy), CELL // 2 - 8, 1)
            pygame.draw.circle(surf, (*BOARD_LINE, 120), (ccx, ccy), CELL // 2 - 14, 1)
            for angle in range(0, 360, 90):
                rad = math.radians(angle)
                dist = CELL // 2 - 8
                px, py = ccx + math.cos(rad) * dist, ccy + math.sin(rad) * dist
                pygame.draw.circle(surf, (*BOARD_LINE, 220), (int(px), int(py)), 3)
            ks = CELL // 2 + 10
            kimg = pygame.transform.smoothscale(self.assets.img_knot_raw, (ks, ks))
            arr = pygame.surfarray.pixels3d(kimg)
            for c, v in enumerate(BOARD_LINE):
                arr[:, :, c] = (arr[:, :, c].astype(int) * v // 255).clip(0, 255)
            del arr
            kimg.set_alpha(180) 
            surf.blit(kimg, (ccx - ks // 2, ccy - ks // 2))
            if sq != THRONE:
                margin = 1
                arm = CELL // 4
                cx0 = (x + margin) if col == 0 else (x + CELL - 1 - margin)
                cy0 = (y + margin) if ui_r == 0 else (y + CELL - 1 - margin)
                ddx = 1 if col == 0 else -1
                ddy = 1 if ui_r == 0 else -1
                pygame.draw.line(surf, (*BOARD_LINE, 255), (cx0, cy0), (cx0 + ddx * arm, cy0), 2)
                pygame.draw.line(surf, (*BOARD_LINE, 255), (cx0, cy0), (cx0, cy0 + ddy * arm), 2)
        for i in range(BOARD_EDGE + 1):
            pygame.draw.line(surf, BOARD_LINE, (0, i * CELL), (BOARD_PX, i * CELL), 1)
            pygame.draw.line(surf, BOARD_LINE, (i * CELL, 0), (i * CELL, BOARD_PX), 1)
        _clip = pygame.Surface((BOARD_PX, BOARD_PX), pygame.SRCALPHA)
        _clip.fill((0, 0, 0, 0))
        pygame.draw.rect(_clip, (255, 255, 255, 255), _clip.get_rect(), border_radius=BOARD_CORNER_R)
        _ba = pygame.surfarray.pixels_alpha(surf); _ma = pygame.surfarray.pixels_alpha(_clip)
        _ba[:] = (_ba.astype(int) * _ma // 255).clip(0, 255).astype(_ba.dtype); del _ba, _ma
        file_letters = "abcdefghi"
        for c in range(BOARD_EDGE):
            lbl = self.fonts['coord'].render(file_letters[c], True, BOARD_LINE)
            surf.blit(lbl, (c * CELL + 3, (BOARD_EDGE - 1) * CELL + CELL - lbl.get_height() - 3))
        for r in range(BOARD_EDGE):
            ui_r = BOARD_EDGE - 1 - r
            lbl = self.fonts['coord'].render(str(r + 1), True, BOARD_LINE)
            surf.blit(lbl, (3, ui_r * CELL + 2))
        return surf

    def _render_border_frame(self):
        B = self.border_w
        tw, th = self.assets.border_tip.get_size()
        tip_h = round(th * B / tw)
        tip_v = pygame.transform.smoothscale(self.assets.border_tip, (B, tip_h))
        tip_h_rot = pygame.transform.rotate(tip_v, 90)
        tip_h_len = tip_h_rot.get_width()
        raw_strip = BOARD_PX + 2 * BOARD_GAP - 2 * tip_h
        n_tiles = max(1, round(raw_strip / PATTERN_H))
        tile_size = round(raw_strip / n_tiles)
        mid_v = pygame.transform.smoothscale(self.assets.border_mid, (B, tile_size))
        mid_h = pygame.transform.rotate(mid_v, -90)
        vert_strip_len = n_tiles * tile_size; horiz_strip_len = vert_strip_len
        def tile_v_fn(tile, length):
            s = pygame.Surface((B, length), pygame.SRCALPHA); y = 0
            while y < length: s.blit(tile, (0, y)); y += tile.get_height()
            return s
        def tile_h_fn(tile, length):
            s = pygame.Surface((length, B), pygame.SRCALPHA); x = 0
            while x < length: s.blit(tile, (x, 0)); x += tile.get_width()
            return s
        surf = pygame.Surface((self.total_board, self.total_board), pygame.SRCALPHA)
        surf.blit(tip_v, (0, B))
        surf.blit(tile_v_fn(mid_v, vert_strip_len), (0, B + tip_h))
        surf.blit(pygame.transform.flip(tip_v, True, True), (0, self.total_board - B - tip_h))
        mid_v_r = pygame.transform.flip(mid_v, True, False); tip_v_r = pygame.transform.flip(tip_v, True, False)
        surf.blit(tip_v_r, (self.total_board - B, B))
        surf.blit(tile_v_fn(mid_v_r, vert_strip_len), (self.total_board - B, B + tip_h))
        surf.blit(pygame.transform.flip(tip_v_r, True, True), (self.total_board - B, self.total_board - B - tip_h))
        surf.blit(tip_h_rot, (B, 0))
        surf.blit(tile_h_fn(mid_h, horiz_strip_len), (B + tip_h_len, 0))
        surf.blit(pygame.transform.flip(tip_h_rot, True, True), (self.total_board - B - tip_h_len, 0))
        mid_h_b = pygame.transform.flip(mid_h, False, True); tip_h_bot_l = pygame.transform.flip(tip_h_rot, False, True)
        surf.blit(tip_h_bot_l, (B, self.total_board - B))
        surf.blit(tile_h_fn(mid_h_b, horiz_strip_len), (B + tip_h_len, self.total_board - B))
        surf.blit(pygame.transform.flip(tip_h_rot, True, False), (self.total_board - B - tip_h_len, self.total_board - B))
        for cx, cy in [(0, 0), (self.total_board - B, 0), (0, self.total_board - B), (self.total_board - B, self.total_board - B)]:
            pygame.draw.rect(surf, (*BG_DARK, 255), (cx, cy, B, B))
        return surf

    def draw_piece(self, S, pos, piece, turn):
        cx, cy = pos
        is_attacker = ((piece > 0 and turn == -1) or (piece < 0 and turn == 1))
        if abs(piece) == 2:
            img, shadow = self.assets.img_king, self.assets.shadow_king
        elif is_attacker:
            img, shadow = self.assets.img_attacker, self.assets.shadow_attacker
        else:
            img, shadow = self.assets.img_defender, self.assets.shadow_defender
        S.blit(shadow, (cx - shadow.get_width() // 2 + SHADOW_OFFSET, cy - shadow.get_height() // 2 + SHADOW_OFFSET))
        S.blit(img, (cx - img.get_width() // 2, cy - img.get_height() // 2))
