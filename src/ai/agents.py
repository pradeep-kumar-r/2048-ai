import random
import os
from abc import ABC, abstractmethod
from typing import Optional
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from src.ai.models import NNModel
from src.ai.replay_buffer import ReplayBuffer, Transition
from src.game.action import Action
from src.config import ConfigManager
from src.logger import logger


class BaseAgent(ABC):
    """Interface for all gameplaying agents."""
    def __init__(self, app_config: ConfigManager):
        """Initialize the base agent.
        Args:
            app_config: Configuration object with agent parameters
        """
        self.action_space_n = app_config.get_model_training_config()["NUM_ACTIONS"]
        self.app_config = app_config
    
    @abstractmethod
    def select_action(self, state: np.ndarray, episode: int) -> int:
        """Select an action given the current state.
        Args:
            state: Current state/observation from the environment
            episode: Current episode number
        Returns:
            int: Selected action
        """
    
    def on_episode_start(self, episode: int) -> None:
        """Called at the start of each episode.
        Args:
            episode: Current episode number
        """
    
    def on_episode_end(self, episode: int) -> None:
        """Called at the end of each episode.
        Args:
            episode: Current episode number
        """
    
    def on_step(self, 
                state: np.ndarray,
                action: int,
                reward: float,
                next_state: np.ndarray,
                done: bool) -> None:
        """Called after each step in the environment.
        Args:
            state: Previous state
            action: Action taken
            reward: Reward received
            next_state: New state after taking the action
            done: Whether the episode is done
        """
    
    def save(self, path: str, episode: int) -> None:
        """Save the agent's state to disk.
        Args:
            path: Directory to save the model
            episode: Current episode number
        """
    
    def load(self, path: str) -> int:
        """Load the agent's state from disk.
        Args:
            path: Path to the saved model
        Returns:
            int: The episode number of the loaded model, or 0 if starting fresh
        """
        if path or not path:
            return 0


class RandomAgent(BaseAgent):
    """A simple agent that selects actions randomly."""
    def __init__(self, app_config: ConfigManager):
        super().__init__(app_config)
        self.action_space = list(range(self.action_space_n))
    
    def select_action(self, state: np.ndarray, episode: int) -> int:
        action = random.choice(self.action_space)
        return list(Action)[action]


class DQNAgent(BaseAgent):
    """DQN based RL agent."""
    def __init__(self, app_config: ConfigManager):
        super().__init__(app_config)
        
        self.game_env_config = app_config.get_game_env_config()
        self.model_training_config = app_config.get_model_training_config()
        self.file_folder_config = app_config.get_file_folder_config()
        
        self.data_dir = self.file_folder_config["GAME_DATA_FOLDER_PATH"]
        self.metrics_file_path = os.path.join(self.data_dir, "model_metrics.csv")
        self.model_dir = self.file_folder_config["MODEL_FOLDER_PATH"]
        self.model_name_prefix = self.file_folder_config["MODEL_NAME_PREFIX"]
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"DQNAgent using device: {self.device}")
        
        self.run_start_time = datetime.now().strftime("%Y%m%d_%H%M")
        
        self.loss = None
        self.training_metrics = {
            'losses': [],
            'rewards': [],
            'episode_lengths': [],
            'epsilon_values': []
        }
        
        self.policy_net = NNModel(num_classes=self.action_space_n).to(self.device)
        self.target_net = NNModel(num_classes=self.action_space_n).to(self.device)
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.AdamW(
            self.policy_net.parameters(), 
            lr=self.model_training_config["LEARNING_RATE"], 
            amsgrad=True
        )
        
        
        self.memory = ReplayBuffer(self.model_training_config["REPLAY_MEMORY_SIZE"])
        
        self.batch_size = self.model_training_config["BATCH_SIZE"]
        self.gamma = self.model_training_config["GAMMA"]
        self.epsilon_start = self.model_training_config["EPSILON_START"]
        self.epsilon_end = self.model_training_config["EPSILON_END"]
        self.epsilon_decay = self.model_training_config["EPSILON_DECAY"]
        self.target_update_frequency = self.model_training_config["TARGET_UPDATE_FREQUENCY"]
        
        self.steps_done = 0
        self.current_epsilon = self.epsilon_start
    
    def select_action(self, state: np.ndarray, episode: int) -> Action:
        if episode <= 1:
            self.current_epsilon = self.epsilon_start
        else:
            self.current_epsilon = self.current_epsilon * (self.epsilon_decay ** (episode - 1))
            self.current_epsilon = max(self.epsilon_end, self.current_epsilon)
        
        self.steps_done += 1
        
        logger.debug(f"State shape: {state.shape}, Epsilon: {self.current_epsilon:.4f}, Steps: {self.steps_done}")
        
        if random.random() >= self.current_epsilon:
            try:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    q_values = self.policy_net(state_tensor)
                    action = q_values.max(1)[1].item()
                    logger.debug(f"EXPLOIT - Q-values: {q_values} - Action: {action}")
                    return list(Action)[action]
            except Exception as e:
                logger.error(f"Error in action selection: {e}")
                action = random.randrange(self.action_space_n)
                logger.warning(f"Fallback random action: {action}")
                return list(Action)[action]
        else:
            action = random.randrange(self.action_space_n)
            logger.debug(f"EXPLORE - Random - Action: {action}")
            return list(Action)[action]
    
    def on_step(self,
                state: np.ndarray,
                action: int,
                reward: float,
                next_state: np.ndarray,
                done: bool) -> None:
        state_tensor = torch.FloatTensor(state).to(self.device)
        next_state_tensor = torch.FloatTensor(next_state).to(self.device) if not done else None
        action_tensor = torch.tensor([action], device=self.device)
        reward_tensor = torch.tensor([reward], device=self.device, dtype=torch.float32)
        self.memory.push(state_tensor, action_tensor, next_state_tensor, reward_tensor)
        self.loss = self.optimize_model()
    
    def optimize_model(self) -> Optional[float]:
        if len(self.memory) < self.batch_size:
            return None
        
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=self.device,
            dtype=torch.bool
        )
        
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        
        state_batch = torch.cat(batch.state).to(self.device)
        action_batch = torch.cat(batch.action).to(self.device)
        reward_batch = torch.cat(batch.reward).to(self.device)
        
        with torch.no_grad():
            state_action_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
        
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        if non_final_next_states.size(0) > 0:
            with torch.no_grad():
                next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
        
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        
        criterion = nn.SmoothL1Loss()
        loss = criterion(
            state_action_values, 
            expected_state_action_values.unsqueeze(1)
        )
        
        self.optimizer.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), self.model_training_config["CLIP_GRADIENTS"])
        self.optimizer.step()
        
        loss_value = loss.item()
        self.training_metrics['losses'].append(loss_value)
        self.training_metrics['epsilon_values'].append(self.current_epsilon)
        
        return loss_value
    
    def update_target_network(self) -> None:
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def on_episode_end(self, episode: int) -> None:
        if episode % self.target_update_frequency == 0:
        # if self.steps_done % self.target_update_frequency == 0:
            self.update_target_network()
            logger.info(f"Target network updated after {episode} episodes")
    
    def save(self, episode: int) -> None:
        if not hasattr(self, 'run_start_time'):
            self.run_start_time = datetime.now().strftime("%Y%m%d_%H%M")
            
        model_path = os.path.join(self.model_dir, f"{self.model_name_prefix}_{self.run_start_time}_ep_{episode}.pt")
        
        torch.save({
            'episode': episode,
            'run_start_time': self.run_start_time,
            'steps_done': self.steps_done,
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.current_epsilon,
            'training_metrics': getattr(self, 'training_metrics', {})
        }, model_path)
        
        self.save_training_metrics(episode)
        logger.info(f"Model saved to {model_path}")
        
    def save_training_metrics(self, episode: int) -> None:
        metrics_df = pd.DataFrame({
            'episode': episode,
            'steps_done': self.steps_done,
            'created_at': self.run_start_time,
            'updated_at': pd.Timestamp.now(),
            'loss_mean_100': np.mean(self.training_metrics['losses'][-100:]) if self.training_metrics['losses'] else np.nan,
            'reward_mean_10': np.mean(self.training_metrics['rewards'][-10:]) if self.training_metrics['rewards'] else np.nan,
            'episode_length_mean_10': np.mean(self.training_metrics['episode_lengths'][-10:]) if self.training_metrics['episode_lengths'] else np.nan,
            'epsilon': self.current_epsilon,
        }, index=[0])
        
        if os.path.exists(self.metrics_file_path):
            old_metrics_df = pd.read_csv(self.metrics_file_path)
            updated_df = pd.concat([old_metrics_df, metrics_df], ignore_index=True)
            updated_df.to_csv(self.metrics_file_path, index=False)
        else:
            metrics_df.to_csv(self.metrics_file_path, index=False)
            
        logger.info(f"Training metrics saved to {self.metrics_file_path}")
    
    def load(self, path: str) -> int:
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            
            self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            self.steps_done = checkpoint.get('steps_done', 0)
            self.current_epsilon = checkpoint.get('epsilon', self.epsilon_start)
            
            self.policy_net.train()
            self.target_net.eval()
            
            episode = checkpoint.get('episode', 0)
            self.run_start_time = checkpoint.get('run_start_time', datetime.now().strftime("%Y%m%d_%H%M"))
            self.steps_done = checkpoint.get('steps_done', 0)
            self.current_epsilon = checkpoint.get('epsilon', self.epsilon_start)
            self.training_metrics = checkpoint.get('training_metrics', {})
            logger.info(f"Model loaded from {path} (episode {episode})")
            return episode
        else:
            logger.warning(f"No model found at {path}, starting from scratch.")
            return 0
