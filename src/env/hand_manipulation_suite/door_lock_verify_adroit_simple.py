import numpy as np
from gym import utils
from gym import spaces
from mjrl.envs import mujoco_env
from mujoco_py import MjViewer
from d4rl import offline_env
import os

from src.env.hand_manipulation_suite.door_lock_adroit_simple import get_aliased_angle

USE_EXTRA_KNOB_ENV = True


class DoorLockVerifyEnv(mujoco_env.MujocoEnv, utils.EzPickle,
                        offline_env.OfflineEnv):
    def __init__(self, **kwargs):
        offline_env.OfflineEnv.__init__(self, **kwargs)
        self.door_hinge_did = 0
        self.door_bid = 0
        self.grasp_sid = 0
        self.handle_sid = 0
        self.knob_sid = 0
        self.extra_knob_sid = 0
        self.is_init_locked = 0
        self.action_dim_simple = 5
        # Override action_space to -1, 1
        self.action_space = spaces.Box(low=-1.0,
                                       high=1.0,
                                       dtype=np.float32,
                                       shape=(self.action_dim_simple, ))

        self.t = 0
        self.max_traj_length = 100

        curr_dir = os.path.dirname(os.path.abspath(__file__))
        self.sim = mujoco_env.get_sim(
            curr_dir + '/assets/door_xknob_adroit_extrageom.xml')
        self.data = self.sim.data
        self.model = self.sim.model

        self.frame_skip = 5
        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }
        self.mujoco_render_frames = False

        self.init_qpos = self.data.qpos.ravel().copy()
        self.init_qvel = self.data.qvel.ravel().copy()
        observation = self.get_obs()
        self.obs_dim = np.sum([
            o.size for o in observation
        ]) if type(observation) is tuple else observation.size

        high = np.inf * np.ones(self.obs_dim)
        low = -high
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        self.seed()

        # change actuator sensitivity
        self.sim.model.actuator_gainprm[self.sim.model.actuator_name2id(
            'A_WRJ1'):self.sim.model.actuator_name2id('A_WRJ0') +
                                        1, :3] = np.array([10, 0, 0])
        self.sim.model.actuator_gainprm[self.sim.model.actuator_name2id(
            'A_FFJ3'):self.sim.model.actuator_name2id('A_THJ0') +
                                        1, :3] = np.array([1, 0, 0])
        self.sim.model.actuator_biasprm[self.sim.model.actuator_name2id(
            'A_WRJ1'):self.sim.model.actuator_name2id('A_WRJ0') +
                                        1, :3] = np.array([0, -10, 0])
        self.sim.model.actuator_biasprm[self.sim.model.actuator_name2id(
            'A_FFJ3'):self.sim.model.actuator_name2id('A_THJ0') +
                                        1, :3] = np.array([0, -1, 0])

        utils.EzPickle.__init__(self)
        ob = self.reset_model()
        self.act_mid = np.mean(self.model.actuator_ctrlrange, axis=1)
        self.act_rng = 0.5 * (self.model.actuator_ctrlrange[:, 1] -
                              self.model.actuator_ctrlrange[:, 0])
        self.door_hinge_did = self.model.jnt_dofadr[self.model.joint_name2id(
            'door_hinge')]
        self.grasp_sid = self.model.site_name2id('S_grasp')
        self.handle_sid = self.model.site_name2id('S_handle')
        self.door_bid = self.model.body_name2id('frame')
        self.knob_sid = self.model.site_name2id('knob_hinge')
        if USE_EXTRA_KNOB_ENV:
            self.extra_knob_sid = self.model.site_name2id('extra_knob')

    def step(self, a):
        a = np.clip(a, -1.0, 1.0)
        whole_action = np.zeros(28)
        whole_action[:4] = np.copy(a[:4])
        whole_action[4:] = np.ones(24) * a[4]

        try:
            whole_action = self.act_mid + whole_action * self.act_rng  # mean center and scale
        except:
            whole_action = whole_action  # only for the initialization phase
        self.do_simulation(whole_action, self.frame_skip)

        done = False
        if self.t >= self.max_traj_length:
            done = True

        ob = self.get_obs()
        door_pose = self.data.qpos[self.door_hinge_did]
        knob_pos = self.data.site_xpos[self.knob_sid].ravel()
        if USE_EXTRA_KNOB_ENV:
            knob_pos = self.data.site_xpos[self.extra_knob_sid].ravel()
        palm_pos = self.data.site_xpos[self.grasp_sid].ravel()

        reward = 0.0
        # dense reward for approaching knob
        reward += -4.0 * np.linalg.norm(palm_pos - knob_pos)
        # Decrease the rew scale to increase exploration
        # Investigate whether SAC subtracts the baseline reward
        if self.is_init_locked:
            # door should be closed after verification
            reward += -0.1 * (door_pose - 0.0)**2

            goal_achieved = True if door_pose <= 0.1 else False
            reward += 1.0 * goal_achieved

            if door_pose > 0.2:
                reward -= 3.0
        else:
            # door should be open after verification
            reward += -0.1 * (door_pose - 1.57)**2

            goal_achieved = True if door_pose >= 0.2 else False
            reward += 10.0 * goal_achieved

            if door_pose < 0.1:
                reward -= 3.0

        self.t += 1

        self.sim.model.site_rgba[self.knob_sid] = [1, 0, 0, 0]
        self.sim.model.site_rgba[self.extra_knob_sid] = [0, 1, 0, 1]

        return ob, reward, done, dict(goal_achieved=goal_achieved)

    def get_obs(self):
        adroit_qpos = self.data.qpos.ravel()[
            1:-2]  # last 2 dimensions are about door hinge and latch
        palm_pos = self.data.site_xpos[self.grasp_sid].ravel()
        door_pose = np.array([self.data.qpos[self.door_hinge_did]])
        knob_pos = self.data.site_xpos[self.knob_sid].ravel()
        if USE_EXTRA_KNOB_ENV:
            knob_pos = self.data.site_xpos[self.extra_knob_sid].ravel()

        latch_pose = self.data.get_joint_qpos("latch")
        latch_visual_pose = get_aliased_angle(latch_pose)

        door_close = 1.0 if door_pose < 0.05 else -1.0
        return np.concatenate([
            adroit_qpos, palm_pos, door_pose, knob_pos, [latch_visual_pose],
            [door_close]
        ])

    def reset_model(self):
        qp = self.init_qpos.copy()
        qv = self.init_qvel.copy()
        self.set_state(qp, qv)

        self.model.body_pos[self.door_bid,
                            0] = self.np_random.uniform(low=-0.3, high=-0.2)
        self.model.body_pos[self.door_bid,
                            1] = self.np_random.uniform(low=0.25, high=0.35)
        self.model.body_pos[self.door_bid,
                            2] = self.np_random.uniform(low=0.252, high=0.35)

        pos_or_neg_example = self.np_random.choice(np.array([0, 1]))
        if pos_or_neg_example < 0.5:
            # init as a positive example: door locked
            latch_init_pose = self.np_random.uniform(low=-0.7, high=1.0)
            self.is_init_locked = True
        else:
            # init as a negative example: example generated by policy
            # TODO (for user): here the latch_init_pose should load from pre-collected
            # policy rollouts
            latch_init_pose = self.np_random.uniform(low=1.0, high=3.1)
            self.is_init_locked = False

        self.sim.data.set_joint_qpos("latch", latch_init_pose)

        self.sim.forward()

        self.t = 0
        return self.get_obs()

    def get_env_state(self):
        """
        Get state of hand as well as objects and targets in the scene
        """
        qp = self.data.qpos.ravel().copy()
        qv = self.data.qvel.ravel().copy()
        door_body_pos = self.model.body_pos[self.door_bid].ravel().copy()
        return dict(qpos=qp, qvel=qv, door_body_pos=door_body_pos)

    def set_env_state(self, state_dict):
        """
        Set the state which includes hand as well as objects and targets in the scene
        """
        qp = state_dict['qpos']
        qv = state_dict['qvel']
        self.set_state(qp, qv)
        self.model.body_pos[self.door_bid] = state_dict['door_body_pos']
        self.sim.forward()

    def mj_viewer_setup(self):
        self.viewer = MjViewer(self.sim)
        self.viewer.cam.azimuth = 90
        self.sim.forward()
        self.viewer.cam.distance = 1.5

    def evaluate_success(self, paths):
        num_success = 0
        num_paths = len(paths)
        # success if door open for 25 steps
        for path in paths:
            if np.sum(path['env_infos']['goal_achieved']) > 25:
                num_success += 1
        success_percentage = num_success * 100.0 / num_paths
        return success_percentage


if __name__ == "__main__":
    env = DoorLockVerifyEnv()
    env.reset()
    env.mj_render()
    for i in range(1000):
        obs = env.reset()
        for t in range(50):
            action = np.zeros(env.action_space.sample().shape)
            obs, reward, done, info = env.step(action)
            env.mj_render()
