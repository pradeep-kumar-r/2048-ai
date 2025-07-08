# 2048-AI

This project implements an AI agent that learns to play the game of 2048 using Reinforcement Learning (RL). The agent is built with a Deep Q-Network (DQN) and is trained to maximize its score by making optimal moves. The project includes an interactive UI built with Pygame to visualize the agent's training process in real-time.

## Features

*   **Reinforcement Learning Agent:** A Deep Q-Network (DQN) agent that learns to play 2048.
*   **Interactive UI:** A Pygame-based user interface to visualize the game and the agent's actions.
*   **Configuration Driven:** Easily configure the UI, game environment, and model training parameters through a `config.yml` file.
*   **Logging:** Detailed logging of the training process, including rewards, losses, and high scores.
*   **Model Checkpointing:** Automatically saves model checkpoints during training.

## Repository Structure

```
2048-ai/
├── data/                 # Stores game scores and other data
├── logs/                 # Contains log files for training sessions
├── models/               # Saves trained model checkpoints
├── src/
│   ├── ai/               # Contains the RL agent implementation
│   │   ├── models.py
│   │   ├── replay_buffer.py
│   │   ├── agents.py
│   ├── game/             # Game logic and environment
│   │   ├── game.py
│   │   ├── ui.py
│   │   └── action.py
│   │   └── colour.py
│   │   └── direction.py
│   │   └── tile.py
│   │   └── board.py
│   │   └── env.py
│   │   └── tile_colour.py
│   ├── __init__.py
│   └── utils.py
│   └── logger.py
│   └── config.py
├── tests/                # Unit and integration tests
├── .gitignore
├── config.yml            # Configuration file for the project
├── LICENSE
├── main.py               # Entry point to run the training or play the game
├── pyproject.toml        # Project metadata and dependencies
├── analysis.ipynb        # Analysis notebook
└── README.md             # This file
```

## Key Libraries and Tools

*   **RL Agent**: `PyTorch`
*   **Game Environment**: `Gym`
*   **Game UI**: `Pygame`
*   **Logging**: `Loguru`
*   **Configuration**: `PyYAML`
*   **Data Handling**: `Pandas`
*   **Visualization**: `Plotly`

## Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/2048-ai.git
    cd 2048-ai
    ```

2.  **Create and activate a virtual environment:**

    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install the dependencies:**

    ```bash
    pip install -e .[dev]
    ```

## How to Run

### Train the AI Agent

To start the training process, run the `main.py` script:

```bash
python main.py
```

This will launch the Pygame window and begin the training loop defined by the parameters in `config.yml`. Training progress, including loss and scores, will be printed to the console and saved in the `logs` directory.

### Run in Interactive Mode

The project also includes a `mock_main` function within `main.py` that allows you to play the game manually. To use this, you will need to modify the `if __name__ == "__main__":` block in `main.py` to call `mock_main()` instead of `main()`.

## Configuration

The project's behavior can be customized by editing the `config.yml` file. The configuration is divided into four main sections:

*   `UI_CONFIG`: Controls the appearance and behavior of the Pygame UI.
*   `GAME_ENV_CONFIG`: Configures the 2048 game environment.
*   `FILE_FOLDER_CONFIG`: Specifies the paths for saving models, data, and logs.
*   `MODEL_TRAINING_CONFIG`: Defines the hyperparameters for the DQN agent and the training process.

## License

This project is licensed under the Apache License 2.0. See the `LICENSE` file for more details.
