import os
import time

from stable_baselines3 import SAC

from robel.dclaw.turn import (DCLAW3_ASSET_PATH, DCLAW4_ASSET_PATH)
from src.env.robel.dclaw_env import DClawScrewTask

if __name__ == "__main__":
    from src.config import argparser

    config, _ = argparser()
    config.jobname = "real_robel_screw_valve4"
    figures_path = os.path.join("figures", config.jobname)
    os.makedirs(figures_path, exist_ok=True)
    checkpoint_path = os.path.join("checkpoints", config.jobname)
    os.makedirs(checkpoint_path, exist_ok=True)

    env = DClawScrewTask(
        asset_path=DCLAW4_ASSET_PATH,
        # frame_skip=80,
        verification_mode=False,
        use_verification_reward=True,
        # device_path='/dev/ttyUSB0',
        # action_mode="fixed_last_joint",
        use_engineered_rew=False,
    )
    env.use_hist_obs = True
    env.reset()

    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        gamma=0.99,
        batch_size=1024,
        target_entropy=-3.0,
    )

    # Training/Continue training
    total_timesteps = 4000000
    save_per_timesteps = 10000

    for i in range(int(total_timesteps / save_per_timesteps)):
        print("======================================")
        start = time.time()
        model.learn(total_timesteps=save_per_timesteps)

        # Save model
        model.save(checkpoint_path + "/mfrl_sc_" +
                   str((i + 1) * save_per_timesteps))
        print("Model saved after", (i + 1) * save_per_timesteps, "timesteps")
