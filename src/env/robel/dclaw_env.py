import numpy as np
import collections
from typing import Any, Dict, Sequence, Tuple, Optional
from gym import spaces
from stable_baselines3 import PPO, SAC

from transforms3d.euler import euler2quat

from robel.components.robot.dynamixel_robot import DynamixelRobotState
from robel.utils.resources import get_asset_path
from robel.dclaw.base_env import BaseDClawObjectEnv
from robel.dclaw.turn import (DCLAW3_ASSET_PATH, DCLAW4_ASSET_PATH,
                              DEFAULT_OBSERVATION_KEYS, RESET_POSE)


class DClawScrewTask(BaseDClawObjectEnv):
    def __init__(self,
                 asset_path: str = DCLAW4_ASSET_PATH,
                 observation_keys: Sequence[str] = DEFAULT_OBSERVATION_KEYS,
                 frame_skip: int = 40,
                 interactive: bool = False,
                 success_threshold: float = 0.2,
                 verification_mode: bool = False,
                 use_verification_reward: bool = False,
                 action_mode: str = None,
                 use_engineered_rew: bool = False,
                 **kwargs):
        """Initializes the environment.

        Args:
            asset_path: The XML model file to load.
            observation_keys: The keys in `get_obs_dict` to concatenate as the
                observations returned by `step` and `reset`.
            frame_skip: The number of simulation steps per environment step.
            interactive: If True, allows the hardware guide motor to freely
                rotate and its current angle is used as the goal.
            success_threshold: The difference threshold (in radians) of the
                object position and the goal position within which we consider
                as a sucesss.
            action_mode: None, "uniform_3_finger", "only_1_finger", 
                "fixed_last_joint"
        """
        super().__init__(sim_model=get_asset_path(asset_path),
                         observation_keys=observation_keys,
                         frame_skip=frame_skip,
                         **kwargs)

        self.action_mode = action_mode
        if action_mode == "uniform_3_finger" or action_mode == "only_1_finger":
            self._action_space = spaces.Box(low=-1.0,
                                            high=1.0,
                                            dtype=np.float32,
                                            shape=(3, ))
        elif action_mode == "fixed_last_joint":
            self._action_space = spaces.Box(low=-1.0,
                                            high=1.0,
                                            dtype=np.float32,
                                            shape=(6, ))

        self._interactive = interactive
        self._success_threshold = success_threshold
        self._desired_claw_pos = RESET_POSE
        self.use_hist_obs = True
        self.obs_hist_len = 41
        self.obs_list = []

        # should be negative to mimic screwing
        self.target = -np.pi
        self._target_bid = self.model.body_name2id('target')

        # The following are modified (possibly every reset) by subclasses.
        self._initial_object_pos = 0
        self._initial_object_vel = 0
        self._set_target_object_pos(0)  # unbounded=True
        self._aliased_target_obj_pose = 0

        self.max_traj_length = 80
        self.verification_mode = verification_mode
        if self.verification_mode:
            # This mode is used for training the IRF policy
            self.init_locked = None
            # The pi_v is very easy to learn
            self.max_traj_length = 20
        else:
            # This mode is used for training the task policy
            self.use_verification_reward = use_verification_reward
            # TODO(for users): train an IRF policy first and put the checkpoint path below
            self.verifier_checkpoint_path = "checkpoints/robel_screw_verifier"
            if self.use_verification_reward:
                self.pi_v = SAC.load(self.verifier_checkpoint_path)

        if asset_path == DCLAW4_ASSET_PATH:
            self.alias_step = np.pi / 2.0
        elif asset_path == DCLAW3_ASSET_PATH:
            self.alias_step = np.pi / 3.0 * 2.0

        self.obj_motor_engaged = False
        self.use_engineered_rew = use_engineered_rew

    def _reset(self):
        """Resets the environment."""
        obj_joint_id = self.sim_scene.model.joint_name2id('valve_OBJRx')
        self.sim_scene.model.jnt_range[obj_joint_id] = np.array([-6.28, 6.28])

        self._set_target_object_pos(self.target, unbounded=True)
        self._aliased_target_obj_pose = self.to_aliasd_angle(self.target)
        self.obs_list = []

        if self.verification_mode:
            init_slot = np.random.choice([2, 0], 1, p=[0.5, 0.5])
            self._initial_object_pos = init_slot * np.pi / 2.0 + \
                np.random.uniform(-self._success_threshold,
                                  self._success_threshold)
            self.init_locked = init_slot == 2

            self._reset_dclaw_and_object(claw_pos=RESET_POSE,
                                         object_pos=self._initial_object_pos,
                                         object_vel=self._initial_object_vel,
                                         guide_pos=self._target_object_pos)
        else:
            """Turns the object with a fixed initial and fixed target position."""
            # Turn from 0 degrees to 180 degrees.
            init_slot = np.random.choice([1, 0, -1], 1, p=[0, 1, 0])
            self._initial_object_pos = init_slot * np.pi / 2.0 + \
                np.random.uniform(-self._success_threshold,
                                  self._success_threshold)
            # self._initial_object_pos = np.random.uniform(
            #     -self._success_threshold, self._success_threshold)

            self._reset_dclaw_and_object(claw_pos=RESET_POSE,
                                         object_pos=self._initial_object_pos,
                                         object_vel=self._initial_object_vel,
                                         guide_pos=self._target_object_pos)

        # Disengage the motor.
        self.obj_motor_engaged = False
        if self._interactive and self.robot.is_hardware:
            self.robot.set_motors_engaged('guide', False)

    def step(self, action: Any) -> Tuple[Any, float, bool, Dict]:
        """Runs one timestep of the environment with the given action.

        Subclasses must override 4 subcomponents of step:
        - `_step`: Applies an action to the robot
        - `get_obs_dict`: Returns the current observation of the robot.
        - `get_reward_dict`: Calculates the reward for the step.
        - `get_done`: Returns whether the episode should terminate.

        Args:
            action: An action to control the environment.

        Returns:
            observation: The observation of the environment after the timestep.
            reward: The amount of reward obtained during the timestep.
            done: Whether the episode has ended. `env.reset()` should be called
                if this is True.
            info: Auxiliary information about the timestep.
        """
        # Get whether the episode should end.
        done = False
        if self.step_count >= self.max_traj_length:
            done = True

        obs_dict = self.get_obs_dict()

        # Fix object orientation when at target
        if np.abs(obs_dict['target_error']) < self._success_threshold:
            if self.robot.is_hardware:
                if not self.obj_motor_engaged:
                    self.robot.set_motors_engaged('object', True)
                    self.obj_motor_engaged = True
            else:
                obj_joint_id = self.sim_scene.model.joint_name2id(
                    'valve_OBJRx')
                self.sim_scene.model.jnt_range[obj_joint_id] = np.array([
                    self._target_object_pos - self._success_threshold / 2.0,
                    self._target_object_pos + self._success_threshold / 2.0
                ])

        # Perform the step.
        if self.action_mode == "uniform_3_finger":
            action = np.concatenate([action] * 3)
        elif self.action_mode == "only_1_finger":
            action = np.concatenate([action, [0, -1, 2 / 3] * 2])
        elif self.action_mode == "fixed_last_joint":
            action = np.array([
                action[0],
                action[1],
                0,
                action[2],
                action[3],
                0,
                action[4],
                action[5],
                0,
            ])
        action = self._preprocess_action(action)
        self._step(action)
        self.last_action = action

        # Get the observation after the step.
        obs_dict = self.get_obs_dict()
        self.last_obs_dict = obs_dict
        obs = self._get_obs(obs_dict=obs_dict)

        # Get the rewards for the observation.
        batched_action = np.expand_dims(np.atleast_1d(action), axis=0)
        batched_obs_dict = {
            k: np.expand_dims(np.atleast_1d(v), axis=0)
            for k, v in obs_dict.items()
        }
        batched_reward_dict = self.get_reward_dict(batched_action,
                                                   batched_obs_dict)

        # Calculate the total reward.
        reward_dict = {k: v.item() for k, v in batched_reward_dict.items()}
        self.last_reward_dict = reward_dict
        reward = 0
        if self.verification_mode:
            if self.init_locked:
                reward_values = (50.0 * reward_dict[key] for key in [
                    'bonus_aliased_success',
                ])
            else:
                reward_values = (50.0 * (1.0 - reward_dict[key]) for key in [
                    'bonus_aliased_success',
                ])
        else:
            reward_values = (
                reward_dict[key] for key in [
                    'bonus_aliased_success',
                ])
            if self.use_engineered_rew:
                reward_values = (reward_dict[key] for key in [
                    'engineered_rew',
                ])
            if reward_dict['bonus_success'] > 0:
                done = True
        reward = np.sum(np.fromiter(reward_values, dtype=float))

        if not self.verification_mode and self.use_verification_reward:
            if done:
                is_at_true_target = self.verify_by_pi_v(num_steps=20)
                reward += 1e3 * is_at_true_target
            else:
                reward = 0

        # Calculate the score.
        batched_score_dict = self.get_score_dict(batched_obs_dict,
                                                 batched_reward_dict)
        score_dict = {k: v.item() for k, v in batched_score_dict.items()}
        self.last_score_dict = score_dict

        # Combine the dictionaries as the auxiliary information.
        info = collections.OrderedDict()
        info.update(('obs/' + key, val) for key, val in obs_dict.items())
        info.update(('reward/' + key, val) for key, val in reward_dict.items())
        info['reward/total'] = reward
        info.update(('score/' + key, val) for key, val in score_dict.items())

        self.step_count += 1

        return obs, reward, done, info

    def _step(self, action: np.ndarray):
        """Applies an action to the robot."""
        self.robot.step({
            'dclaw': action,
            'guide': np.atleast_1d(self._target_object_pos),
        })

    def _get_obs(self,
                 obs_dict: Optional[Dict[str, np.ndarray]] = None) -> Any:
        """Returns the current observation of the environment.

        This matches the environment's observation space.
        """
        if obs_dict is None:
            obs_dict = self.get_obs_dict()

        if self.use_hist_obs and not self.verification_mode:
            obs_keys = [
                'claw_qpos',
                'aliased_obj_angle_hist',
            ]
        else:
            obs_keys = [
                'claw_qpos',
                'aliased_obj_angle',
            ]

        obs_values = (obs_dict[key] for key in obs_keys)
        obs = np.concatenate([np.ravel(v) for v in obs_values])
        return obs

    def to_aliasd_angle(self, angle):
        # [-np.pi / 4.0, np.pi / 4.0) should be better
        # aliased_angle = np.mod(angle + self.alias_step / 2,
        #                        self.alias_step) - self.alias_step / 2
        aliased_angle = np.mod(angle, self.alias_step)
        return aliased_angle

    def get_obs_dict(self) -> Dict[str, np.ndarray]:
        """Returns the current observation of the environment.

        Returns:
            A dictionary of observation values. This should be an ordered
            dictionary if `observation_keys` isn't set.
        """
        claw_state, object_state, guide_state = self.robot.get_state(
            ['dclaw', 'object', 'guide'])

        # If in interactive mode, use the guide motor position as the goal.
        if self._interactive:
            self._set_target_object_pos(guide_state.qpos)

        # Calculate the signed angle difference to the target in [-pi, pi].
        target_error = self._target_object_pos - object_state.qpos
        # target_error = np.mod(target_error + np.pi, 2 * np.pi) - np.pi

        aliased_obj_angle = self.to_aliasd_angle(object_state.qpos)

        if len(self.obs_list) < self.obs_hist_len:
            self.obs_list = [aliased_obj_angle] * self.obs_hist_len
        else:
            self.obs_list.pop(0)
            self.obs_list.append(aliased_obj_angle)

        obs_dict = collections.OrderedDict((
            ('claw_qpos', claw_state.qpos),
            ('object_qvel', object_state.qvel),
            ('object_angle', object_state.qpos),  # not bounded
            ('aliased_obj_angle', aliased_obj_angle),
            ('aliased_obj_angle_hist', np.concatenate(self.obs_list[::5])),
            ('target_error', target_error),
        ))

        return obs_dict

    def aliased_success_classifier(self,
                                   obs_dict: Dict[str, np.ndarray],
                                   soft_margin: float = 0):
        aliased_dist = np.abs(self._aliased_target_obj_pose -
                              obs_dict['aliased_obj_angle'])
        aliased_dist = min(aliased_dist, self.alias_step - aliased_dist)
        return aliased_dist < self._success_threshold + soft_margin

    def verify_by_pi_v(self, num_steps=20):
        # Reset hand pose first
        object_state, guide_state = self.robot.get_state(['object', 'guide'])
        self._reset_dclaw_and_object(claw_pos=RESET_POSE,
                                     object_pos=object_state.qpos,
                                     object_vel=object_state.qvel,
                                     guide_pos=guide_state.qpos)

        # Reset whether the obj motor is engaged
        if np.abs(self._target_object_pos -
                  object_state.qpos) < self._success_threshold:
            if self.robot.is_hardware:
                self.robot.set_motors_engaged('object', True)
                self.obj_motor_engaged = True
            else:
                obj_joint_id = self.sim_scene.model.joint_name2id(
                    'valve_OBJRx')
                self.sim_scene.model.jnt_range[obj_joint_id] = np.array([
                    self._target_object_pos - self._success_threshold / 2.0,
                    self._target_object_pos + self._success_threshold / 2.0
                ])

        is_at_true_target = True
        for t in range(num_steps):
            # self.render("human")
            obs_dict = self.get_obs_dict()
            obs_keys = [
                'claw_qpos',
                'aliased_obj_angle',
            ]
            obs_values = (obs_dict[key] for key in obs_keys)
            obs = np.concatenate([np.ravel(v) for v in obs_values])

            aliased_success = self.aliased_success_classifier(
                obs_dict=obs_dict, soft_margin=0)
            if not aliased_success:
                is_at_true_target = False
                break

            action, _states = self.pi_v.predict(obs, deterministic=True)

            action = self._preprocess_action(action)
            self._step(action)

        return is_at_true_target

    def get_reward_dict(
        self,
        action: np.ndarray,
        obs_dict: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Returns the reward for the given action and observation."""
        target_dist = np.abs(obs_dict['target_error'])
        is_success = target_dist < self._success_threshold
        # claw_vel = obs_dict['claw_qvel']
        obj_vel = np.squeeze(obs_dict['object_qvel'])
        target_vel = self.target / (np.abs(self.target) + 1e-8) * 1.0
        # engineered_rew = 1 - np.abs(obj_vel - target_vel)
        engineered_rew = self.target / (np.abs(self.target) + 1e-8) * obj_vel

        aliased_success = self.aliased_success_classifier(obs_dict=obs_dict)

        reward_dict = collections.OrderedDict((
            # Penalty for distance away from goal.
            ('target_dist_cost', -5 * target_dist),
            # Reward for close proximity with goal.
            ('bonus_partial', 10 * (target_dist < 0.25)),
            ('bonus_success', 50 * (target_dist < self._success_threshold)),
            ('bonus_aliased_success', aliased_success),
            ('engineered_rew', engineered_rew),
        ))
        return reward_dict

    def get_score_dict(
        self,
        obs_dict: Dict[str, np.ndarray],
        reward_dict: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Returns a standardized measure of success for the environment."""
        target_dist = np.abs(obs_dict['target_error'])

        if self.verification_mode:
            if self.init_locked:
                success = reward_dict['bonus_aliased_success']
            else:
                success = not reward_dict['bonus_aliased_success']
            success = np.array([[success]])
        else:
            success = target_dist < self._success_threshold
        score_dict = collections.OrderedDict((
            ('success', success), ))
        return score_dict

    def _set_target_object_pos(self,
                               target_pos: float,
                               unbounded: bool = False):
        """Sets the goal angle to the given position."""
        # Modulo to [-pi, pi].
        if not unbounded:
            target_pos = np.mod(target_pos + np.pi, 2 * np.pi) - np.pi
        self._target_object_pos = np.asarray(target_pos, dtype=np.float32)

        # Mark the target position in sim.
        # WARNING: euler2quat will mutate a passed numpy array.
        self.model.body_quat[self._target_bid] = euler2quat(
            0, 0, float(target_pos))


if __name__ == "__main__":
    use_real_dclaw = False
    if use_real_dclaw:
        # Create a hardware environment for the D'Claw turn task.
        # `device_path` refers to the device port of the Dynamixel USB device.
        # e.g. '/dev/ttyUSB0' for Linux, '/dev/tty.usbserial-*' for Mac OS.
        env = DClawScrewTask(
            asset_path=DCLAW4_ASSET_PATH,
            verification_mode=False,
            use_verification_reward=False,
            device_path='/dev/ttyUSB0',
            action_mode="fixed_last_joint",
        )
    else:
        # Create a simulation environment for the D'Claw turn task.
        env = DClawScrewTask(
            asset_path=DCLAW4_ASSET_PATH,
            verification_mode=False,
            use_verification_reward=False,
            action_mode="only_1_finger",
        )

    for i in range(1000):
        print("New rollout starts")
        env.reset()
        for t in range(100):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            env.render("human")
