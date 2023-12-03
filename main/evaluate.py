import os
import flappy_bird_gymnasium
import gymnasium
from stable_baselines3 import PPO

# Number of episodes to evaluate
NUM = 10
# Name of the model to evaluate
MODEL_NAME = "FlappyBird_stage_final"
# Directory where the model is saved
MODEL_DIR = "trained_models"

env = gymnasium.make("FlappyBird-v0", render_mode="rgb_array")
model = PPO.load(path=os.path.join(MODEL_DIR, MODEL_NAME), env=env)
average_score = 0

for _ in range(NUM):
    obs, _ = env.reset()
    while True:
        # Next action:
        # (feed the observation to your agent here)
        action, _ = model.predict(obs)
        # Processing:
        obs, reward, terminated, _, info = env.step(action)

        # Checking if the player is still alive
        if terminated:
            break
    average_score += info["score"]
average_score /= NUM
print("Average score: ", average_score)
env.close()
