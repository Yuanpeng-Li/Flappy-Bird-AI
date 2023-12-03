import os
import flappy_bird_gymnasium
import gymnasium
from stable_baselines3 import PPO

# Name of the model to evaluate
MODEL_NAME = "FlappyBird_stage_final"
# Directory where the model is saved
MODEL_DIR = "trained_models"

env = gymnasium.make("FlappyBird-v0", render_mode="human")
model = PPO.load(path=os.path.join(MODEL_DIR, MODEL_NAME), env=env)

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
print("Score: ", info["score"])
os.system("pause")

env.close()
