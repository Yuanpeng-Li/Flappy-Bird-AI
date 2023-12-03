import os
import sys
import gymnasium as gym
import flappy_bird_gymnasium
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

NUM_ENVS = 16 # Number of parallel environments, better not to exceed the number of CPU cores available
LOG_DIR = 'logs'

# Linear scheduler
def linear_schedule(initial_value, final_value=0.0):

    if isinstance(initial_value, str):
        initial_value = float(initial_value)
        final_value = float(final_value)
        assert (initial_value > 0.0)

    def scheduler(progress):
        return final_value + progress * (initial_value - final_value)

    return scheduler

def make_env():
    def _init():
        env = gym.make("FlappyBird-v0")
        env = Monitor(env)
        return env
    return _init

# Callback for logging the reward of each step
class RewardLogger(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardLogger, self).__init__(verbose)

    def _on_step(self):
        reward = sum(self.locals['rewards']) # Adapt based on how rewards are stored in 'self.locals'
        self.logger.record('train/reward', reward)
        return True



def main():
    env = SubprocVecEnv([make_env() for _ in range(NUM_ENVS)])
    
    lr_schedule = linear_schedule(2.5e-4, 2.5e-6)
    clip_range_schedule = linear_schedule(0.15, 0.025)
    
    # train for the first time
    model = PPO(
        "MlpPolicy", 
        env,
        ent_coef = 0.001,
        verbose=1,
        n_steps=2048,
        batch_size=256,
        learning_rate=lr_schedule,
        clip_range=clip_range_schedule,
        tensorboard_log="logs"
    )


    # Load the model to continue training
    # model = PPO.load(r"./trained_models/FlappyBird_stage_1.zip", env=env, custom_objects={'n_steps': 16384, 'batch_size': 256})

    save_dir = "trained_models"
    os.makedirs(save_dir, exist_ok=True)

    # Checkpoint callback for saving the model during training 
    checkpoint_interval = 15625 # checkpoint_interval * num_envs = total_steps_per_checkpoint
    checkpoint_callback = CheckpointCallback(save_freq=checkpoint_interval, save_path=save_dir, name_prefix="FlappyBird")
    
    # Reward logger callback
    reward_logger = RewardLogger()
    
    # Writing the training logs from stdout to a file
    original_stdout = sys.stdout
    log_file_path = os.path.join(save_dir, "training_log.txt")
    with open(log_file_path, 'w') as log_file:
        sys.stdout = log_file
    
        model.learn(
            total_timesteps=int(100000000), # total_timesteps = stage_interval * num_envs * num_stages
            callback=[checkpoint_callback, reward_logger] # checkpoint_callback is used to save the model
        )
        env.close()

    # Restore stdout
    sys.stdout = original_stdout

    # Save the final model
    model.save(os.path.join(save_dir, "final.zip"))

if __name__ == "__main__":
    main()