from src.config import config as app_config
from src.logger import logger
from src.game.env import _2048Env
from src.ai.agents import DQNAgent
import time


def main():
    # MAIN TRAINING LOOP
    
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
        for episode in range(max_episodes):
            if not running:
                break
                
            obs, info = env.reset()
            logger.info(f"Episode {episode+1}/{max_episodes}")
            logger.debug(f"Initial state: {info}")
            
            for step in range(1, max_steps_per_episode + 1):
                action_taken = agent.select_action(obs, episode)
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
                
            logger.info(f"Episode {episode+1}/{max_episodes} finished after {step} steps")
            logger.info(f"Total reward: {env.rewards}")
            logger.debug(f"Final state: {info}")
                
            if episode % episodes_per_checkpoint == 0:
                logger.info(f"Saving checkpoint at episode {episode}")
                agent.save(episode)
    
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    # except Exception as e:
    #     logger.error(f"Error during training: {e}")
    finally:
        env.cleanup_ui()


def mock_main():
    from src.game.game import Game
    from src.game.action import Action
    _2048_game = Game(game_config=app_config.get_game_env_config())
    for episode in range(2):
        _2048_game.reset()
        print(f"Episode {episode+1}")
        for step in range(10):
            print(f"\n\nStep: {step}\nBoard:\n")
            print(_2048_game.board)
            user_input = input("Enter action: (RIGHT, LEFT, UP, DOWN):\n")
            while user_input not in Action.__members__:
                user_input = input("Invalid Input!! Enter action: (RIGHT, LEFT, UP, DOWN):\n")
            _2048_game.step(Action[user_input])
            if _2048_game.is_game_over:
                break


if __name__ == "__main__":
    main()