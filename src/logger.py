import sys
import os
from loguru import logger
from src.config import config

# Check & create log folder & files
log_path = config.get_file_folder_config()["LOGS_FOLDER_PATH"]
os.makedirs(log_path, exist_ok=True)

training_log_path = log_path / "training_log.log"
debug_log_path = log_path / "debug_log.log"

for log_file in [training_log_path, debug_log_path]:
    if not log_file.exists():
        log_file.touch()
        

logger.remove()

logger.add(sys.stdout, level="INFO", 
          format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")

logger.add(training_log_path, level="INFO",
          format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}")

logger.add(debug_log_path, level="DEBUG",
          format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}")


# Test the logger
if __name__ == "__main__":
    logger.info("Testing whether logging works correctly")
    logger.debug("DEBUG - Testing whether logging works correctly")
