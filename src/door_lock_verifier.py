import os
import numpy as np
import time
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv

from src.env.hand_manipulation_suite.door_v0 import DoorEnvV0
from src.env.hand_manipulation_suite.door_lock_verify_adroit_simple import DoorLockVerifyEnv


def test(env: DoorEnvV0, model, total_trials, test_horizon=100, render=False):
    mean_ep_len = 0.
    successes = 0.
    mean_reward = 0.
    for _ in range(total_trials):
        obs = env.reset()
        if render:
            env.mj_render()
            time.sleep(0.04)
        t = 0
        is_success = 0
        while t < test_horizon:
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, _, info = env.step(action)
            if render:
                env.mj_render()

            t += 1
            if info['goal_achieved']:
                is_success = 1
        mean_ep_len += t
        mean_reward += rewards
        successes += is_success

    mean_ep_len /= float(total_trials)
    successes /= float(total_trials)
    mean_reward /= float(total_trials)
    return mean_ep_len, successes, mean_reward


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
    if not continue_training == 0:
        timestep = list(range(0, continue_training, save_per_timesteps))
        success_rates = list(np.load(checkpoint_path +
                                     '/success_rates.npy'))[:len(timestep)]
        rewards = list(np.load(checkpoint_path +
                               '/rewards.npy'))[:len(timestep)]
    for i in range(int(total_timesteps / save_per_timesteps)):
        print("======================================")
        start = time.time()
        model.learn(total_timesteps=save_per_timesteps)

        # Save model
        model.save(checkpoint_path + "/mfrl_sc_" +
                   str((i + 1) * save_per_timesteps + continue_training))
        print("Model saved after",
              (i + 1) * save_per_timesteps + continue_training, "timesteps")

        # Testing
        mean_ep_len, successes, mean_reward = test(env,
                                                   model,
                                                   total_trials=100)
        print("success rate:", successes)
        print("mean_reward:", mean_reward)
        print("mean ep len:", mean_ep_len)
        print("training time:", time.time() - start)
        success_rates.append(successes)
        rewards.append(mean_reward)
        timestep.append(i * save_per_timesteps + continue_training)

        np.save(checkpoint_path + '/success_rates', np.array(success_rates))
        np.save(checkpoint_path + '/rewards', np.array(rewards))

        plt.figure()
        plt.plot(timestep, rewards, 'ro-', label="reward")
        plt.grid(True)
        plt.legend()
        plt.savefig(checkpoint_path + "/rewards.png")
        plt.close()

        plt.figure()
        plt.plot(timestep, success_rates, 'ro-', label="success rates")
        plt.grid(True)
        plt.legend()
        plt.savefig(checkpoint_path + "/success_rates.png")
        plt.close()

    test(env, model, total_trials=100, test_horizon=400, render=True)
