import os
import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.buffers import ReplayBuffer

from src.env.hand_manipulation_suite.door_v0 import DoorEnvV0
from src.env.hand_manipulation_suite.door_lock_adroit_simple import DoorLockEnvAdroitSimple


def test(env: DoorEnvV0, model, total_trials, test_horizon=101, render=False):
    mean_ep_len = 0.
    successes = 0.
    mean_reward = 0.
    mean_door_closed_rate = 0.
    progress_tqdm = tqdm(range(total_trials))
    for _ in progress_tqdm:
        obs = env.reset()
        t = 0
        is_success = 0
        is_door_closed = 0
        traj_rew = 0.0
        while t < test_horizon:
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, _, info = env.step(action)
            if render:
                env.mj_render()

            t += 1
            traj_rew += rewards
            if info['goal_achieved']:
                is_success = 1
            if info['door_closed']:
                is_door_closed = 1

        mean_ep_len += t
        mean_reward += traj_rew
        successes += is_success
        mean_door_closed_rate += is_door_closed

    mean_ep_len /= float(total_trials)
    successes /= float(total_trials)
    mean_reward /= float(total_trials)
    mean_door_closed_rate /= float(total_trials)
    return mean_ep_len, successes, mean_reward, mean_door_closed_rate


if __name__ == "__main__":
    from src.config import argparser

    config, _ = argparser()
    config.jobname = "door_locker"
    checkpoint_path = os.path.join("checkpoints", config.jobname)
    os.makedirs(checkpoint_path, exist_ok=True)

    env = DoorLockEnvAdroitSimple()
    env.reset()

    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
    )

    # Training/Continue training
    total_timesteps = 4000000
    save_per_timesteps = 10000
    timestep = []
    success_rates = []
    door_closed_rates = []
    rewards = []
    continue_training = 0
    if not continue_training == 0:
        timestep = list(range(0, continue_training, save_per_timesteps))
        success_rates = list(np.load(checkpoint_path +
                                     '/success_rates.npy'))[:len(timestep)]
        max_suc_rate_id = np.argmax(
            np.load(checkpoint_path + '/success_rates.npy'))
        print("Best checkpoint:", max_suc_rate_id)
        print("Best performance:",
              np.load(checkpoint_path + '/success_rates.npy')[max_suc_rate_id])
        rewards = list(np.load(checkpoint_path +
                               '/rewards.npy'))[:len(timestep)]
        door_closed_rates = list(
            np.load(checkpoint_path +
                    '/door_closed_rates.npy'))[:len(timestep)]

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
        mean_ep_len, successes, mean_reward, mean_door_closed_rate = test(
            env, model, total_trials=100, test_horizon=101)
        print("success rate:", successes)
        print("mean_door_closed_rate", mean_door_closed_rate)
        print("mean_reward:", mean_reward)
        print("mean ep len:", mean_ep_len)
        print("training time:", time.time() - start)
        success_rates.append(successes)
        rewards.append(mean_reward)
        timestep.append(i * save_per_timesteps + continue_training)
        door_closed_rates.append(mean_door_closed_rate)

        np.save(checkpoint_path + '/success_rates', np.array(success_rates))
        np.save(checkpoint_path + '/rewards', np.array(rewards))
        np.save(checkpoint_path + '/door_closed_rates',
                np.array(door_closed_rates))

        plt.figure()
        plt.plot(timestep, rewards, 'ro-', label="reward")
        plt.grid(True)
        plt.legend()
        plt.savefig(checkpoint_path + "/rewards.png")
        plt.close()

        plt.figure()
        plt.plot(timestep, success_rates, 'ro-', label="success rates")
        plt.plot(timestep, door_closed_rates, 'bo-', label="door closed rates")
        plt.grid(True)
        plt.legend()
        plt.savefig(checkpoint_path + "/success_rates.png")
        plt.close()

    test(env, model, total_trials=100, test_horizon=101, render=True)
