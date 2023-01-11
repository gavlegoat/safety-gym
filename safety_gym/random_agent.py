#!/usr/bin/env python

import argparse
import gymnasium as gym
import safety_gym  # noqa
import numpy as np  # noqa


def run_random(env_name):
    env = gym.make(env_name)
    obs, _ = env.reset()
    done = False
    ep_ret = 0
    ep_cost = 0
    while True:
        if done:
            print('Episode Return: %.3f \t Episode Cost: %.3f'
                  % (ep_ret, ep_cost))
            ep_ret, ep_cost = 0, 0
            obs, _ = env.reset()
        assert env.observation_space.contains(obs)
        act = env.action_space.sample()
        assert env.action_space.contains(act)
        obs, reward, terminated, truncated, info = env.step(act)
        done = terminated or truncated
        # print('reward', reward)
        ep_ret += reward
        ep_cost += info.get('cost', 0)
        env.render()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='Safexp-PointGoal1-v0')
    args = parser.parse_args()
    run_random(args.env)
