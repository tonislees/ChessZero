import pygame
from ..tablut_jax import BOARD_EDGE, BOARD_SIZE, THRONE

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

TIMER_INITIAL_DEFAULT = 300.0   # 5 minutes in seconds

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
