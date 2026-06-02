import pygame

class MoveAnimation:
    def __init__(self, from_idx, to_idx, piece, turn=None, duration=0.2):
        self.from_idx = from_idx
        self.to_idx = to_idx
        self.piece = piece
        self.duration = duration
        self.start_ticks = pygame.time.get_ticks()
        self.progress = 0.0
        self.is_finished = False
        self.turn = turn

    def update(self):
        elapsed = (pygame.time.get_ticks() - self.start_ticks) / 1000.0
        self.progress = min(1.0, elapsed / self.duration)
        if self.progress >= 1.0:
            self.is_finished = True

    def get_pos(self, board_edge, cell_size):
        from_r, from_c = divmod(self.from_idx, board_edge)
        to_r, to_c = divmod(self.to_idx, board_edge)
        
        # UI row is inverted
        ui_from_r = board_edge - 1 - from_r
        ui_to_r = board_edge - 1 - to_r
        
        curr_ui_r = ui_from_r + (ui_to_r - ui_from_r) * self.progress
        curr_c = from_c + (to_c - from_c) * self.progress
        
        return curr_c * cell_size, curr_ui_r * cell_size
