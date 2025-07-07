from typing import Dict, Any, Optional, Tuple
import pygame
import numpy as np
from src.game.colour import Colour
from src.game.tile_colour import TileBGColour, TileFontColour
from src.game.action import Action
from src.logger import logger


class UI:
    def __init__(self,
                 ui_config: Dict[str, Any],
                 board: np.ndarray,
                 episode: int,
                 high_score: Optional[int] = 0,
                 last_action: Optional[Action] = None):
        pygame.init()
        self.ui_config = ui_config
        self.board_width: int = self.ui_config["BOARD_DIM"]
        self.board_height: int = self.ui_config["BOARD_DIM"]
        self.cell_size: int = self.ui_config["CELL_SIZE_IN_PIXELS"]
        self.board: np.ndarray = board
        self.score = np.max(board)
        self.high_score: int = high_score
        self.last_action = last_action
        self.episode = episode
        
        self.padding = self.ui_config["BOARD"]["PADDING"]
        self.board_pixel_width = self.board_width * self.cell_size
        self.board_pixel_height = self.board_height * self.cell_size
        
        self.title_height = self.ui_config["TITLE"]["HEIGHT"]
        self.info_band_height = self.ui_config["INFO_BAND"]["HEIGHT"]
        self.extra_height = self.title_height + self.info_band_height + 2 * self.padding
        
        self.window_width = self.board_pixel_width + 2 * self.padding
        self.window_height = self.board_pixel_height + self.extra_height
        
        self._initialize_fonts()
        self.screen = None
        self._initialize_display()
        
        logger.debug(f"UI initialized with window size: {self.window_width}x{self.window_height}")
        
        # Initialize UI elements
        self.title_rect = None
        self.info_band_rect = None
        self.board_rect = None
        self.game_over_rect = None
    
    def _initialize_fonts(self) -> None:
        self.font_title = pygame.font.SysFont(
            self.ui_config["TITLE"]["FONT"]["NAME"],
            self.ui_config["TITLE"]["FONT"]["SIZE"],
            bold=True if self.ui_config["TITLE"]["FONT"].get("STYLE", "") == "bold" else False
        )
        
        self.font_info = pygame.font.SysFont(
            self.ui_config["INFO_BAND"]["SECTIONS"]["SCORE"]["FONT"]["NAME"],
            self.ui_config["INFO_BAND"]["SECTIONS"]["SCORE"]["FONT"]["SIZE"],
            bold=True if self.ui_config["INFO_BAND"]["SECTIONS"]["SCORE"]["FONT"].get("STYLE", "") == "bold" else False
        )
        
        self.font_tile = pygame.font.SysFont(
            self.ui_config["TILE"]["FONT"]["NAME"],
            self.ui_config["TILE"]["FONT"]["SIZE"],
            bold=True if self.ui_config["TILE"]["FONT"].get("STYLE", "") == "bold" else False
        )
        
        self.font_game_over = pygame.font.SysFont(
            self.ui_config["GAME_OVER_LABEL"]["FONT"]["NAME"],
            self.ui_config["GAME_OVER_LABEL"]["FONT"]["SIZE"],
            bold=True if self.ui_config["GAME_OVER_LABEL"]["FONT"].get("STYLE", "") == "bold" else False
        )
        
        logger.debug("Fonts initialized")
    
    def _initialize_display(self) -> None:
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption(self.ui_config["TITLE"]["TEXT"])
        self._is_initialized = True
        logger.debug(f"Display initialized with dimensions: {self.window_width}x{self.window_height}")
    
    def _draw_title(self, surface: pygame.Surface) -> Tuple[pygame.Surface, pygame.Rect]:
        title_height = self.ui_config['TITLE']['HEIGHT']
        title_section = pygame.Rect(0, 0, self.window_width, title_height)
        
        pygame.draw.rect(surface, Colour[self.ui_config['BG_COLOUR']].value, title_section)
        
        pygame.draw.rect(surface, Colour[self.ui_config['BOARD']['BORDER']['FILL']].value,
                     title_section, self.ui_config['BOARD']['BORDER']['THICKNESS'] // 2)
        
        title_text = self.font_title.render(f"{self.ui_config['TITLE']['TEXT']} - Episode: {self.episode}",
                                       True, Colour[self.ui_config['TITLE']['COLOUR']].value)
        
        title_rect = title_text.get_rect(center=(self.window_width // 2, title_section.height // 2))
        surface.blit(title_text, title_rect)
        
        return surface, title_section
    
    def _draw_info_band(self, surface: pygame.Surface) -> Tuple[pygame.Surface, pygame.Rect]:
        title_height = self.ui_config['TITLE']['HEIGHT']
        info_band_height = self.ui_config['INFO_BAND']['HEIGHT']
        info_band_rect = pygame.Rect(0, title_height, self.window_width, info_band_height)
        
        pygame.draw.rect(
            surface,
            Colour[self.ui_config['BG_COLOUR']].value,
            info_band_rect
        )
        
        pygame.draw.rect(
            surface,
            Colour[self.ui_config['BOARD']['BORDER']['FILL']].value,
            info_band_rect,
            self.ui_config['BOARD']['BORDER']['THICKNESS'] // 2
        )
        
        section_width = self.window_width // 3
        
        # Left section to show score
        score_section_rect = pygame.Rect(0, title_height, section_width, info_band_height)
        score_text = self.font_info.render(
            f"Score: {self.score}",
            True,
            Colour[self.ui_config['INFO_BAND']['SECTIONS']['SCORE']['COLOUR']].value
        )
        score_rect = score_text.get_rect(
            center=(section_width // 2, title_height + info_band_height // 2)
        )
        surface.blit(score_text, score_rect)
        
        # Center section to show last action
        action_section_rect = pygame.Rect(
            section_width, title_height, section_width, info_band_height
        )
        action_text = self.font_info.render(
            f"Action: {self.last_action.name if self.last_action else '-'}",
            True,
            Colour[self.ui_config['INFO_BAND']['SECTIONS']['ACTION']['COLOUR']].value
        )
        action_rect = action_text.get_rect(
            center=(section_width + section_width // 2, title_height + info_band_height // 2)
        )
        surface.blit(action_text, action_rect)
        
        # Right section to show high score
        high_score_section_rect = pygame.Rect(
            2 * section_width, title_height, section_width, info_band_height
        )
        high_score_text = self.font_info.render(
            f"High: {self.high_score}",
            True,
            Colour[self.ui_config['INFO_BAND']['SECTIONS']['HIGH_SCORE']['COLOUR']].value
        )
        high_score_rect = high_score_text.get_rect(
            center=(2 * section_width + section_width // 2, title_height + info_band_height // 2)
        )
        surface.blit(high_score_text, high_score_rect)
        
        self.score_section_rect = score_section_rect
        self.action_section_rect = action_section_rect
        self.high_score_section_rect = high_score_section_rect
        
        return surface, info_band_rect
    
    def _draw_board(self, surface: pygame.Surface, info_band_rect: pygame.Rect) -> Tuple[pygame.Surface, pygame.Rect]:
        pad = self.ui_config['BOARD']['PADDING']
        
        board_topleft_x = pad
        board_topleft_y = info_band_rect.bottom + pad
        
        board_width = self.board_width * self.cell_size
        board_height = self.board_height * self.cell_size
        
        board_bottomright_x = board_topleft_x + board_width
        board_bottomright_y = board_topleft_y + board_height
        
        board_rect = pygame.Rect(
            board_topleft_x, 
            board_topleft_y, 
            board_width, 
            board_height
        )
        
        pygame.draw.rect(
            surface,
            Colour[self.ui_config['BOARD']['FILL']].value,
            board_rect
        )
        
        pygame.draw.rect(
            surface,
            Colour[self.ui_config['BOARD']['BORDER']['FILL']].value,
            board_rect,
            self.ui_config['BOARD']['BORDER']['THICKNESS']
        )
        
        for i in range(1, self.board_width):
            x_pos = board_topleft_x + i * self.cell_size
            pygame.draw.line(
                surface,
                Colour[self.ui_config['BOARD']['GRID']['FILL']].value,
                (x_pos, board_topleft_y),
                (x_pos, board_bottomright_y),
                self.ui_config['BOARD']['GRID']['THICKNESS']
            )
        
        for i in range(1, self.board_height):
            y_pos = board_topleft_y + i * self.cell_size
            pygame.draw.line(
                surface,
                Colour[self.ui_config['BOARD']['GRID']['FILL']].value,
                (board_topleft_x, y_pos),
                (board_bottomright_x, y_pos),
                self.ui_config['BOARD']['GRID']['THICKNESS']
            )
        
        self._draw_tiles(surface, board_rect)
        
        self.board_rect = board_rect
        return surface, board_rect
    
    def _draw_tiles(self, surface: pygame.Surface, board_rect: pygame.Rect) -> None:
        tile_padding = self.ui_config['TILE']['PADDING']
        
        for row in range(self.board_height):
            for col in range(self.board_width):
                tile_x = board_rect.left + col * self.cell_size + tile_padding
                tile_y = board_rect.top + row * self.cell_size + tile_padding
                tile_size = self.cell_size - 2 * tile_padding
                
                tile_value = int(self.board[row][col])
                
                tile_rect = pygame.Rect(tile_x, tile_y, tile_size, tile_size)
                
                tile_bg_color_name = TileBGColour.get_color_for_value(tile_value)
                tile_color = Colour[tile_bg_color_name].value
                
                tile_font_color_name = TileFontColour.get_color_for_value(tile_value)
                tile_font_color = Colour[tile_font_color_name].value
                
                pygame.draw.rect(surface, tile_color, tile_rect)
                
                if tile_value > 0:
                    font_size_adjustment = 0
                    if tile_value >= 1000:
                        font_size_adjustment = -6
                    elif tile_value >= 100:
                        font_size_adjustment = -2
                    
                    if font_size_adjustment != 0:
                        adjusted_font = pygame.font.SysFont(
                            self.ui_config["TILE"]["FONT"]["NAME"],
                            self.ui_config["TILE"]["FONT"]["SIZE"] + font_size_adjustment,
                            bold=True if self.ui_config["TILE"]["FONT"].get("STYLE", "") == "bold" else False
                        )
                        font_to_use = adjusted_font
                    else:
                        font_to_use = self.font_tile
                    
                    tile_text = font_to_use.render(str(tile_value), True, tile_font_color)
                    text_rect = tile_text.get_rect(center=tile_rect.center)
                    surface.blit(tile_text, text_rect)
    
    def _game_over_screen(self, surface: pygame.Surface, board_rect: pygame.Rect) -> Tuple[pygame.Surface, pygame.Rect]:
        overlay = pygame.Surface((self.window_width, self.window_height), pygame.SRCALPHA)
        overlay.fill(Colour.TRANSPARENT_GREY.value)
        surface.blit(overlay, (0, 0))
        
        game_over_text = self.font_game_over.render(
            self.ui_config['GAME_OVER_LABEL']['TEXT'],
            True,
            Colour[self.ui_config['GAME_OVER_LABEL']['FONT']['COLOUR']].value
        )
        
        game_over_rect = game_over_text.get_rect(center=board_rect.center)
        surface.blit(game_over_text, game_over_rect)
        
        score_text = self.font_info.render(
            f"Final Score: {self.score}",
            True,
            Colour[self.ui_config['GAME_OVER_LABEL']['FONT']['COLOUR']].value
        )
        
        score_rect = score_text.get_rect(
            center=(board_rect.centerx, game_over_rect.bottom + 20)
        )
        surface.blit(score_text, score_rect)
        
        self.game_over_rect = game_over_rect
        return surface, game_over_rect
        
    def update_state(self, board: np.ndarray, high_score: int, last_action: Action) -> None:
        self.board = board
        self.score = np.max(board)
        self.high_score = high_score
        self.last_action = last_action
        logger.debug(f"Updated game state: score={self.score}, high_score={self.high_score}, action={last_action}")
    
    def render(self, is_game_over: bool = False) -> bool:
        """Render the current game state to the screen.
        
        Args:
            is_game_over: Whether to show the game over screen.
            
        Returns:
            bool: True if the game should continue, False if the window was closed.
        """
        if self.screen is None:
            self._initialize_display()
            
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        
        self.screen.fill(Colour[self.ui_config['BG_COLOUR']].value)
        
        _, self.title_rect = self._draw_title(self.screen, self.episode)
        _, self.info_band_rect = self._draw_info_band(self.screen)
        _, self.board_rect = self._draw_board(self.screen, self.info_band_rect)
        
        if is_game_over:
            _, self.game_over_rect = self._game_over_screen(self.screen, self.board_rect)
        
        pygame.display.flip()
        
        return True
    
    def _cleanup_ui(self) -> None:
        if self.screen is not None:
            pygame.display.quit()
            self.screen = None
        
        return True

    def close(self) -> None:
        logger.debug("Closing pygame UI resources")
        self._cleanup_ui()
        pygame.quit()
    