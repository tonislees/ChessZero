import pygame
import os
from .constants import CELL, SHADOW_ALPHA, BORDER_TINT

def _asset_path(name):
    base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
    return os.path.join(base, name)

def _load_img(path, size):
    img = pygame.image.load(path).convert_alpha()
    w, h = img.get_size()
    scale = size / max(w, h)
    img = pygame.transform.smoothscale(img, (int(w * scale), int(h * scale)))
    return img

def _make_shadow(src, shadow_alpha):
    w, h = src.get_size()
    shadow = pygame.Surface((w, h), pygame.SRCALPHA)
    shadow.blit(src, (0, 0))
    arr_rgb = pygame.surfarray.pixels3d(shadow)
    arr_rgb[:, :, :] = 0
    del arr_rgb
    arr_a = pygame.surfarray.pixels_alpha(shadow)
    arr_a[:, :] = (arr_a[:, :].astype(int) * shadow_alpha // 255).clip(0, 255)
    del arr_a
    return shadow

class Assets:
    def __init__(self):
        piece_size = int(CELL * 0.85)
        self.img_attacker = _load_img(_asset_path("Tablut_attacker.png"), piece_size)
        self.img_defender = _load_img(_asset_path("Tablut_defender.png"), piece_size)
        self.img_king     = _load_img(_asset_path("Tablut_king.png"), piece_size)
        
        from .constants import SHADOW_ALPHA
        self.shadow_attacker = _make_shadow(self.img_attacker, SHADOW_ALPHA)
        self.shadow_defender = _make_shadow(self.img_defender, SHADOW_ALPHA)
        self.shadow_king     = _make_shadow(self.img_king, SHADOW_ALPHA)

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
