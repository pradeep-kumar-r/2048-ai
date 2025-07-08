import os
import glob
from src.config import config as app_config
from src.logger import logger
from src.game.env import _2048Env
from src.ai.agents import DQNAgent


def cleanup_data_log_files():
    file_folder_config = app_config.get_file_folder_config()
    
    data_dir = file_folder_config.get('DATA_DIR', 'data')
    csv_files = glob.glob(os.path.join(data_dir, '*.csv'))
    for csv_file in csv_files:
        try:
            os.remove(csv_file)
            logger.debug(f"Removed CSV file: {csv_file}")
        except OSError as e:
            logger.warning(f"Error removing {csv_file}: {e}")
    
    logs_dir = file_folder_config.get('LOGS_DIR', 'logs')
    log_files = glob.glob(os.path.join(logs_dir, '*.log'))
    for log_file in log_files:
        try:
            os.remove(log_file)
            logger.debug(f"Removed log file: {log_file}")
        except OSError as e:
            logger.warning(f"Error removing {log_file}: {e}")
    
    models_dir = file_folder_config.get('MODEL_DIR', 'models')
    model_files = glob.glob(os.path.join(models_dir, '*.pt'))
    for model_file in model_files:
        try:
            os.remove(model_file)
            logger.debug(f"Removed model file: {model_file}")
        except OSError as e:
            logger.warning(f"Error removing {model_file}: {e}")
    
    logger.info("Cleanup completed")


def main(cleanup_files: bool = False):
    # MAIN TRAINING LOOP
    if cleanup_files:
        cleanup_data_log_files()
    
    # Get configuration
    model_training_config = app_config.get_model_training_config()
    
    # agent = RandomAgent(app_config)
    agent = DQNAgent(app_config)
    env = _2048Env(app_config)
    
    max_episodes = model_training_config["MAX_TRAINING_EPISODES"]
    max_steps_per_episode = model_training_config["MAX_TIMESTEPS_PER_EPISODE"]
    episodes_per_checkpoint = model_training_config["EPISODES_PER_CHECKPOINT"]
    
    logger.info(f"Starting the training loop for {max_episodes} episodes")
    
    try:
        running = True
        for _ in range(max_episodes):
            if not running:
                break
                
            obs, info = env.reset()
            logger.info(f"Episode {env.episodes_count}/{max_episodes}")
            logger.debug(f"Initial state: {info}")
            
            for step in range(max_steps_per_episode):
                action_taken = agent.select_action(obs, env.episodes_count)
                obs, _, terminated, truncated, info = env.step(action_taken)
                loss = f"{agent.loss:.4f}" if agent.loss else "inf"
                if step % model_training_config["PRINT_LOSS_EVERY"] == 0:
                    logger.info(f"Step {step}, Loss: {loss}")
                
                if not env.render():
                    running = False
                    break
                
                if terminated or truncated:
                    break
            
            if not running:
                break
                
            logger.info(f"Episode {env.episodes_count}/{max_episodes} finished after {step + 1} steps, with score {env.game.score}")
            logger.info(f"Total reward: {env.rewards}")
            logger.info(f"High score: {env.high_score}")
            logger.debug(f"Final state: {info}")
                
            if env.episodes_count % episodes_per_checkpoint == 0:
                logger.info(f"Saving checkpoint at episode {env.episodes_count}")
                agent.save(env.episodes_count)
    
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Error during training: {e}")
    finally:
        env.cleanup_ui()


if __name__ == "__main__":
    main()
    # mock_main()
    
    
    
    
    

    
# def mock_main():
#     import numpy as np
#     from src.game.game import Game
#     from src.game.action import Action
#     _2048_game = Game(game_config=app_config.get_game_env_config())
#     for episode in range(2):
#         _2048_game.reset()
#         print(f"Episode {episode+1}")
#         _2048_game.board = np.array([
#                 [2, 0, 0, 0],
#                 [2, 0, 0, 0],
#                 [2, 0, 0, 0],
#                 [2, 0, 0, 0],
#             ])
#         for step in range(10):
#             print(f"\n\nStep: {step}\nBoard:\n")
#             print(_2048_game.board)
#             user_input = input("Enter action: (RIGHT, LEFT, UP, DOWN):\n")
#             while user_input not in Action.__members__:
#                 user_input = input("Invalid Input!! Enter action: (RIGHT, LEFT, UP, DOWN):\n")
#             _, game_over, score, has_merged, board_sequence = _2048_game.step(Action[user_input])
#             print("Has merged" if has_merged else "")
#             print(f"Score: {score}")
#             print("Board Sequence:")
#             for temp_board in board_sequence:
#                 print(temp_board)
#             if game_over:
#                 break
