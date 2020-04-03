from pathlib import Path

import numpy as np

from gym import utils
from gym.envs.mujoco import mujoco_env


class ZeroLeggedRobotEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self, target_pos=0.5):
        self.target_pos = target_pos

        model_file_path = Path(__file__).parent / 'assets/zero_legged_robot.xml'
        mujoco_env.MujocoEnv.__init__(self, str(model_file_path), 5)
        utils.EzPickle.__init__(self)


    def step(self, action):
        # xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        ob = self._get_obs()
        reward_ctrl = - 0.001 * np.square(action).sum()
        # reward_run = (xposafter - xposbefore) / self.dt
        reward_run = - (xposafter - self.target_pos) ** 2 / self.dt
        reward = reward_ctrl + reward_run
        done = False
        return ob, reward, done, dict(reward_run=reward_run, reward_ctrl=reward_ctrl)

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5
