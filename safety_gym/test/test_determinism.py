#!/usr/bin/env python

import unittest
import numpy as np
import gymnasium as gym
import safety_gym  # noqa


class TestDeterminism(unittest.TestCase):
    def check_qpos(self, env_name):
        ''' Check that a single environment is seed-stable at init '''
        for seed in [0, 1, 123456789]:
            print('running', env_name, seed)
            env1 = gym.make(env_name)
            env1.reset(seed=seed)
            env2 = gym.make(env_name)
            env2.reset(seed=seed)
            np.testing.assert_almost_equal(env1.unwrapped.data.qpos,
                                           env2.unwrapped.data.qpos)
            done1 = False
            done2 = False
            while not (done1 or done2):
                action = np.random.randn(*env1.action_space.shape)
                _, rew1, term1, trunc1, _ = env1.step(action)
                done1 = term1 or trunc1
                _, rew2, term2, trunc2, _ = env2.step(action)
                done2 = term2 or trunc2
                self.assertEqual(rew1, rew2)
                self.assertEqual(done1, done2)
                np.testing.assert_almost_equal(env1.unwrapped.data.qpos,
                                               env2.unwrapped.data.qpos)

    def test_qpos(self):
        ''' Run all the bench envs '''
        for env_spec in gym.envs.registry.keys():
            if 'Safexp' in env_spec:
                self.check_qpos(env_spec)

    # def check_names(self, env_name):
    #     '''
    #     Check that all the names in the mujoco model are the same for
    #     different envs
    #     '''
    #     print('check names', env_name)
    #     env1 = gym.make(env_name)
    #     env1.seed(0)
    #     env1.reset()
    #     env2 = gym.make(env_name)
    #     env2.seed(1)
    #     env2.reset()
    #     model1 = env1.unwrapped.model
    #     model2 = env2.unwrapped.model
    #     shared_names = ['actuator_names', 'body_names', 'camera_names',
    #                     'geom_names', 'joint_names', 'light_names',
    #                     'mesh_names', 'sensor_names', 'site_names',
    #                     'tendon_names', 'userdata_names']
    #     for n in shared_names:
    #         self.assertEqual(getattr(model1, n), getattr(model2, n))
    #
    # def test_names(self):
    #     ''' Run all the bench envs '''
    #     for env_spec in gym.envs.registry.keys():
    #         if 'Safexp' in env_spec:
    #             self.check_names(env_spec)


if __name__ == '__main__':
    unittest.main()
