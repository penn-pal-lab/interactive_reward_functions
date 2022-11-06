import os
import time
from stable_baselines3 import SAC

from src.env.hand_manipulation_suite.door_v0 import DoorEnvV0
from src.env.hand_manipulation_suite.door_lock_verify_adroit_simple import DoorLockVerifyEnv

if __name__ == "__main__":
    from src.config import argparser

    config, _ = argparser()
    config.jobname = "door_lock_verifier"
    checkpoint_path = os.path.join("checkpoints", config.jobname)
    os.makedirs(checkpoint_path, exist_ok=True)

    env = DoorLockVerifyEnv()
    env.reset()

    model = SAC("MlpPolicy", env, verbose=1)

    # Training/Continue training
    total_timesteps = 2000000
    save_per_timesteps = 10000
    timestep = []
    success_rates = []
    rewards = []
    continue_training = 2000000
    for i in range(int(total_timesteps / save_per_timesteps)):
        print("======================================")
        start = time.time()
        model.learn(total_timesteps=save_per_timesteps)

        # Save model
        model.save(checkpoint_path + "/mfrl_sc_" +
                   str((i + 1) * save_per_timesteps + continue_training))
        print("Model saved after",
              (i + 1) * save_per_timesteps + continue_training, "timesteps")
