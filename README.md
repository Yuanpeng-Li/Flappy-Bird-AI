# FlappyBird-AI
This project is an AI agent that learns to play Flappy Bird. The game is built using the Pygame library. The AI agent is built on PPO(Proximal Policy Optimization) algorithm. The agent is trained using the OpenAI Gym environment.
## File Structure
```bash
Flappy-Bird-AI
├───logs # Stores the logs of the training process
├───main
│   ├───evaluate.py # Evaluates the model
│   ├───human_flappy_bird.py # Play the game manually
│   ├───test.py # Test the model with visual output
│   └───train.py # Train the model
├───trained_models # Stores the trained model
├───requirements.txt 
└───README.md
```
The trained model is stored in the `trained_models/` folder. The logs are stored in the `logs/` folder. The `main/` folder contains the code for training, testing and evaluating the model. The `human_flappy_bird.py` file is used to play the game manually.
## Running Guide
This project is based on the Python programming language and primarily utilizes standard libraries like [OpenAI Gymnasium](https://gymnasium.farama.org/) and [Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/). The Python version used is 3.9.18, and it is recommended to use [Anaconda](https://www.anaconda.com) to configure the Python environment. The following setup process has been tested on Windows 11. Below are console/terminal/shell commands.
### Environment Setup
```bash
# Create a new environment
conda create -n flappybird python=3.9.18
conda activate flappybird

# Install the required libraries
cd [parent_directory_of_project]/Flappy-Bird-AI
pip install -r requirements.txt
```

### Training the Model
If you want to train your own model, you can run `train.py` in the `main/` folder. The model will be saved in the `trained_models/` folder. The training process will be logged in the `logs/` folder. The training process can be visualized using Tensorboard. The following command can be used to run Tensorboard.
```bash
cd [parent_directory_of_project]/Flappy-Bird-AI
tensorboard --logdir=logs/
```
- **Notes**: As the reward is intrinsically linked to how many steps the game takes, consider increasing the `n_steps` parameter for potentially improved performance. The final model is trained in two stages with different `n_steps` and `batch_size`. The final model is the model obtained after the second stages.
### Testing the Model
If you want to test the model with visual output, you can run `test.py` in the `main/` folder.
### Evaluating the Model
If you want to evaluate the model with average points of the game, you can run `evaluate.py` in the `main/` folder. The model will be loaded from the `trained_models/` folder. 
- **Notes**: The performance of the final model is so high that the evaluation time will be very long and may never be completed.
### Playing the Game Manually

If you want to play the game manually:

- **Running the Game**: Execute `human_flappy_bird.py` located in the `main/` folder.
  
- **Game Controls**
  | Action          | Key       |
  |-----------------|-----------|
  | Fly             | `space`   |
  | Pause/Unpause   | `p`       |
  | Restart         | `r`       |
  | Exit            | `q`       |


## Acknowledgements
This project would not have been possible without the contributions of the open-source community. We'd like to express our gratitude to the developers of the following libraries and projects:

- [OpenAI Gymnasium](https://gymnasium.farama.org/): A powerful library for developing and comparing reinforcement learning algorithms.

- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/): A collection of high-quality implementations of reinforcement learning algorithms.

- [Flappy Bird Gymnasium](https://github.com/markub3327/flappy-bird-gymnasium): The environment used in this project, which simulates the Flappy Bird game for reinforcement learning experiments.

- [Street Fighter AI](https://github.com/linyiLYi/street-fighter-ai): A source of inspiration and reference for our project, which explores AI in the context of fighting games.

We are deeply thankful for the hard work and dedication of these developers to the open-source community. Their contributions have been instrumental in the success of this project.
