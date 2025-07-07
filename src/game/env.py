import os
from typing import Optional, Dict, Any, Tuple
import gym
from gym import spaces
import numpy as np
import pandas as pd
from src.game.game import Game
from src.game.action import Action
from src.game.ui import UI
from src.config import ConfigManager
from src.logger import logger


class _2048Env(gym.Env):
    def __init__(self, app_config: ConfigManager):
        super().__init__()
        self.game_env_config = app_config.get_game_env_config()
        self.ui_config = app_config.get_ui_config()
        self.model_training_config = app_config.get_model_training_config()
        self.file_folder_config = app_config.get_file_folder_config()
        
        logger.info("Initializing Env with config settings")

        self.game = Game(game_config=self.game_env_config)
        self.last_action: Optional[Action] = None
        
        self.episodes_count: int = -1
        self.rewards: float = 0.0
        
        self.headless: bool = True
        self.ui: Optional[UI] = None
        
        self.score_path = self.file_folder_config["SCORES_FILE_PATH"]
        self._set_high_score()
        
        # Action space: 0:RIGHT, 1:DOWN, 2:LEFT, 3:UP
        self.action_space = spaces.Discrete(self.model_training_config["NUM_ACTIONS"])
        
        # Observation space: Board as a flattened numpy array
        self.observation_space = spaces.Box(
            low=0,
            high=131072,
            shape=(self.game.board.size ** 2,),
            dtype=np.uint32
        )

    def _set_high_score(self):
        try:
            scores_df = pd.read_csv(self.score_path)
            self.high_score = scores_df['score'].max()
        except FileNotFoundError:
            self.high_score = 0
            
    
    def _get_obs(self) -> np.ndarray:
        return self.game.board.flatten()

    def _get_info(self) -> Dict[str, Any]:
        state = self.game.get_game_state()
        state["rewards"] = self.rewards
        return state

    def reset(self, 
              seed: Optional[int] = None, 
              options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self.episodes_count += 1
        self.rewards = 0.0
        
        episodes_per_render = self.game_env_config.get("EPISODES_PER_RENDER", 1)
        self.headless = all([self.episodes_count % episodes_per_render != 0,
                             self.episodes_count > 2])
        
        if self.ui is not None and self.headless:
            self.cleanup_ui()
        
        self.game.reset()
        logger.debug(f"Environment reset for episode {self.episodes_count}")
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action: Action) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Takes an action and transitions the environment to the next state.
        
        Args:
            action: The action to take, represented as an Action enum.
        
        Returns:
            A tuple containing:
                observation: The new state of the environment, represented as a flattened numpy array.
                reward: The reward obtained from taking the action.
                terminated: A boolean indicating whether the episode has terminated.
                truncated: A boolean indicating whether the episode has been truncated.
                info: A dictionary containing additional information about the state of the environment.
        """
        self.last_action = action
        reward = 0
        
        logger.debug(f"Taking action {action}")
        _, terminated, new_score, has_merged = self.game.step(action)
        
        # Update rewards accumulated
        if action not in list(Action.__members__.keys()):
            r = self.model_training_config["REWARDS"]["INVALID_ACTION"]
            reward += r
            logger.debug(f"Invalid action, reward: {r}")
        
        if has_merged:
            r = self.model_training_config["REWARDS"]["MERGE"]
            reward += r
            logger.debug(f"Tiles merged, reward: {r}")
        
        if terminated:
            if new_score >= 2048:
                if new_score > self.high_score:
                    r = self.model_training_config["REWARDS"]["WIN_WITH_NEW_HIGH_SCORE"]
                else:
                    r = self.model_training_config["REWARDS"]["WIN"]
            else:
                r = self.model_training_config["REWARDS"]["LOSE"]
            reward += r
            logger.debug(f"Game terminated, you {'win' if new_score >= 2048 else 'lose'}, reward: {r}")
            self._record_final_score()
            
        r = self.model_training_config["REWARDS"]["ACTION"]
        reward += r
        logger.debug(f"Action taken, adding action reward: {r}")
        
        self.rewards += reward
        observation = self._get_obs()
        info = self._get_info()
        truncated = False
        return observation, reward, terminated, truncated, info

    def render(self) -> bool:
        if self.ui is None:
            self.ui = UI(ui_config=self.ui_config, board=self.game.board)
        else:
            self.ui.update_state(board=self.game.board, 
                               high_score=self.high_score, 
                               last_action=self.last_action)
        
        if not self.headless and self.ui:
            return self.ui.render(is_game_over=self.game.is_game_over)
        return True

    def cleanup_ui(self) -> None:
        if self.ui:
            logger.debug("Cleaning up UI resources")
            self.ui.close()
            self.ui = None
            
    def _record_final_score(self) -> None:
        if not self.game.is_game_over:
            return
        
        final_state = self.game.get_game_state()
        final_state["board"] = str(final_state["board"].tolist())
        final_state["reward"] = self.rewards
        
        record = {
            "game_id": [final_state["game_id"]],
            "is_game_over": [final_state["is_game_over"]],
            "created_at": [final_state["created_at"]],
            "updated_at": [final_state["updated_at"]],
            "steps_elapsed": [final_state["steps_elapsed"]],
            "score": [final_state["score"]],
            "is_game_won": [final_state["is_game_won"]],
            "board": [final_state["board"]],
            "reward": [final_state["reward"]]
        }
        
        final_state_df = pd.DataFrame(record)
        
        try:
            scores_df = pd.read_csv(self.score_path)
            scores_df = pd.concat([scores_df, final_state_df], ignore_index=True)
        except FileNotFoundError:
            scores_df = final_state_df
        
        scores_df.to_csv(self.score_path, index=False)