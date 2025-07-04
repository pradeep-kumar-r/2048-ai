import random
from uuid import uuid4
from datetime import datetime
from typing import Optional, Tuple, Dict, Any
import numpy as np
from numba import jit
from src.game.action import Action
from src.game.direction import Direction
from src.logger import logger


class Game:
    def __init__(self, game_config: Optional[Dict[str, Any]]):
        self.game_config = game_config
        self.game_dim = self.game_config["TILES_PER_ROW"]
        self._2_tile_probability = self.game_config["2_TILE_PROBABILITY"]
        self.board: np.ndarray = np.zeros((self.game_dim, self.game_dim))
        self.is_game_over: bool = False
        self.score: int = np.max(self.board)
        self.steps_elapsed: int = 0
        self.state: Optional[Dict[str, int]] = None
        self.start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._start_new_game()

    def reset(self) -> None:
        self.board = np.zeros((self.game_dim, self.game_dim))
        self.is_game_over = False
        self.score = np.max(self.board)
        self.steps_elapsed += 1
        self.start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._start_new_game()

    def _generate_tile_value(self) -> int:
        if random.random() < self._2_tile_probability:
            return 2
        else:
            return 4

    def _get_random_empty_position(self) -> Tuple[int, int]:
        empty_positions = np.argwhere(self.board == 0)
        if empty_positions.size == 0:
            return None
        new_random_empty_position = random.choice(empty_positions)
        return tuple(new_random_empty_position)

    def _generate_tile(self) -> bool:
        pos = self._get_random_empty_position()
        if pos is None:
            self.is_game_over = True
            logger.info("Game Over")
            return False
        else:
            self.board[pos] = self._generate_tile_value()
            return True

    @jit(nopython=True)
    def _start_new_game(self) -> None:
        for _ in range(2):
            assert self._generate_tile()

    @jit(nopython=True)
    def _check_merge(self, direction: Direction) -> bool:
        for i in range(self.game_dim):
            for j in range(self.game_dim):
                new_i = i + direction.value[0]
                new_j = j + direction.value[1]
                if self.board[i][j] == self.board[new_i][new_j]:
                    return True
        return False

    def _transform_board(self, direction: Direction) -> np.ndarray:
        board = self.board.copy()
        if direction == Direction.LEFT:
            return board
        elif direction == Direction.RIGHT:
            return np.fliplr(board)
        elif direction == Direction.UP:
            return np.rot90(board, k=1)
        else:
            return np.rot90(board, k=-1)

    def _reverse_transform_board(self, 
                                 board: np.ndarray,
                                 direction: Direction) -> np.ndarray:
        if direction == Direction.LEFT:
            return board
        elif direction == Direction.RIGHT:
            return np.fliplr(board)
        elif direction == Direction.UP:
            return np.rot90(board, k=-1)
        else:
            return np.rot90(board, k=1)

    def _slide_non_zeros(self, line: np.ndarray) -> np.ndarray:
        zeros = line == 0
        non_zeros = ~zeros
        return np.concatenate([line[non_zeros], line[zeros]])

    @jit(nopython=True)
    def step(self, action: Action) -> Tuple[int, bool, int, bool]:
        if self.is_game_over:
            logger.info("Game Over")
            return (self.steps_elapsed,
                    self.is_game_over,
                    self.score,
                    False)
        
        if np.all(self.board != 0) and not self._check_merge(action):
            self.is_game_over = True
            logger.info("Can't move, Game Over")
            return (self.steps_elapsed,
                    self.is_game_over,
                    self.score,
                    False)
        
        self.steps_elapsed += 1
    
        # Step 1 - transform the board as per the direction of the action
        direction = Direction[action.name]
        transformed_board = self._transform_board(direction)
        has_merged = False
        
        for i in range(self.game_dim):
            line = transformed_board[i]
            
            # Step 2 - Slide all tiles to the left
            slided_line = self._slide_non_zeros(line)
            
            # Step 3 - Merge tiles
            merged_line = slided_line.copy()
            for j in range(len(slided_line)-1):
                if slided_line[j] == slided_line[j+1]:
                    merged_line[j] *= 2
                    merged_line[j+1] = 0
                    
            # Step 4 - Slide all tiles to the left again
            final_line = self._slide_non_zeros(merged_line)
            
            transformed_board[i] = final_line
        
        self.board = self._reverse_transform_board(transformed_board, direction)    
        
        new_score = np.max(self.board)
        if new_score > self.score:
            has_merged = True
            self.score = new_score
            
        return (self.steps_elapsed,
                self.is_game_over,
                self.score,
                has_merged)

    def get_final_game_state(self) -> Dict[Any, Any]:
        if self.is_game_over:
            final_game_state = {
                "game_id": str(uuid4()),
                "is_game_over": self.is_game_over,
                "created_at": self.start_time,
                "ended_at": datetime.now(),
                "steps_elapsed": self.steps_elapsed, 
                "score": self.score,
                "is_game_won": self.score >= 2048,
                "board": self.board,
            }
            return final_game_state
        else:
            return dict()