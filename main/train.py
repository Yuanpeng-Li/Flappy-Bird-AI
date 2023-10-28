import os
import sys
import gymnasium as gym
import flappy_bird_gymnasium
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

NUM_ENVS = 8
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

def make_env(rank):
    def _init():
        env = gym.make("FlappyBird-v0")
        env = Monitor(env)
        return env
    return _init

class RewardLogger(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardLogger, self).__init__(verbose)

    def _on_step(self):
        reward = sum(self.locals['rewards']) # Adapt based on how rewards are stored in 'self.locals'
        self.logger.record('train/reward', reward)
        return True



def main():
    env = SubprocVecEnv([make_env(i) for i in range(NUM_ENVS)])
    # env = gym.make("FlappyBird-v0")
    
    lr_schedule = linear_schedule(2.5e-4, 2.5e-6)
    clip_range_schedule = linear_schedule(0.15, 0.025)
    
    # model = PPO(
    #     "MlpPolicy", 
    #     env,
    #     ent_coef = 0.001,
    #     # device="cuda", 
    #     # verbose=1,
    #     # n_steps=512,
    #     # batch_size=512,
    #     # n_epochs=4,
    #     # gamma=0.94,
    #     # learning_rate=lr_schedule,
    #     # clip_range=clip_range_schedule,
    #     tensorboard_log="logs"
    # )
    model = PPO.load("trained_models/FlappyBird_1000000_steps.zip",
                     env = env,
                     custom_objects = {"learning_rate": lr_schedule,
                                       "clip_range": clip_range_schedule}
                    )

    save_dir = "trained_models"
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_interval = 31250 # checkpoint_interval * num_envs = total_steps_per_checkpoint
    checkpoint_callback = CheckpointCallback(save_freq=checkpoint_interval, save_path=save_dir, name_prefix="FlappyBird")
    reward_logger = RewardLogger()
    # Writing the training logs from stdout to a file
    original_stdout = sys.stdout
    log_file_path = os.path.join(save_dir, "training_log.txt")
    with open(log_file_path, 'w') as log_file:
        sys.stdout = log_file
    
        model.learn(
            total_timesteps=int(100000000), # total_timesteps = stage_interval * num_envs * num_stages (1120 rounds)
            callback=[checkpoint_callback, reward_logger]#, stage_increase_callback]
        )
        env.close()

    # Restore stdout
    sys.stdout = original_stdout

    # Save the final model
    model.save(os.path.join(save_dir, "final.zip"))

if __name__ == "__main__":
    main()