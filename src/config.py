from pathlib import Path
import yaml
import numpy as np
from src.game.direction import Direction


class ConfigManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            with open("config.yml", "r", encoding="utf-8") as file:
                cls.config = yaml.safe_load(file)
                cls._set_config()
        return cls._instance

    @classmethod
    def _set_config(cls):
        cls.game_env_config = cls.config["GAME_ENV_CONFIG"]
        cls.file_folder_config = cls.config["FILE_FOLDER_CONFIG"]
        cls.model_training_config = cls.config["MODEL_TRAINING_CONFIG"]
        cls.ui_config = cls.config["UI_CONFIG"]
        
        cls.file_folder_config["LOGS_FOLDER_PATH"] = Path(cls.file_folder_config["LOGS_FOLDER_PATH"])
        cls.file_folder_config["MODEL_FOLDER_PATH"] = Path(cls.file_folder_config["MODEL_FOLDER_PATH"])
        cls.file_folder_config["GAME_DATA_FOLDER_PATH"] = Path(cls.file_folder_config["GAME_DATA_FOLDER_PATH"])
        cls.file_folder_config["SCORES_FILE_PATH"] = Path(cls.file_folder_config["SCORES_FILE_PATH"])
    
    @classmethod
    def get_model_training_config(cls):
        return cls.model_training_config
    
    @classmethod
    def get_file_folder_config(cls):
        return cls.file_folder_config
    
    @classmethod
    def get_game_env_config(cls):
        return cls.game_env_config
    
    @classmethod
    def get_ui_config(cls):
        return cls.ui_config

config = ConfigManager()


# Testing configs
if __name__ == "__main__":
    game_env_config = config.get_game_env_config()
    print(game_env_config)
    print(game_env_config["2_TILE_PROBABILITY"])
    