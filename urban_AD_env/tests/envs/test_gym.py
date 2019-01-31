from __future__ import division, print_function
#import gym

from ..xurban_AD_env.envs.roundabout_env import *


def test_urban_step():
    env = gym.make('urban-v0')

    env.reset()
    for i in range(3):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
    env.close()

    assert env.observation_space.contains(obs)
    assert 0 <= reward <= 1


def test_merge_step():
    env = gym.make('urban-merge-v0')

    env.reset()
    for i in range(3):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
    env.close()

    assert env.observation_space.contains(obs)
    assert 0 <= reward <= 1


def test_roundabout_step(steps=5):
    env =  RoundaboutEnv() #gym.make('urban-roundabout-v0')

    env.reset()
    for i in range(steps):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
    env.close()

    assert env.observation_space.contains(obs)
    assert 0 <= reward <= 1

if __name__ == '__main__':
    test_roundabout_step()