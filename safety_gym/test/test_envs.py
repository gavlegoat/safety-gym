#!/usr/bin/env python

import unittest
import gymnasium as gym
import safety_gym.envs  # noqa


class TestEnvs(unittest.TestCase):
    def check_env(self, env_name):
        ''' Run a single environment for a single episode '''
        print('running', env_name)
        env = gym.make(env_name)
        env.reset()
        done = False
        while not done:
            _, _, terminated, truncated, _ = \
                env.step(env.action_space.sample())
            done = terminated or truncated

    def test_envs(self):
        ''' Run all the bench envs '''
        for env_spec in gym.envs.registry.keys():
            if 'Safexp' in env_spec:
                self.check_env(env_spec)


if __name__ == '__main__':
    unittest.main()
