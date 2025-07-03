import sys
import os
from loguru import logger
from src.config import config

log_path = config.get_logs_config()["LOGS_FOLDER_PATH"]
os.makedirs(log_path, exist_ok=True)

logger.remove()

logger.add(sys.stdout, level="INFO", 
          format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")

logger.add(f"{log_path}/training_log.log", level="INFO",
          format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}")

logger.add(f"{log_path}/debug_log.log", level="DEBUG",
          format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}")

# Test the logger
if __name__ == "__main__":
    logger.info("Testing whether logging works correctly")
    logger.debug("DEBUG - Testing whether logging works correctly")
