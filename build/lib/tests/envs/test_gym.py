from __future__ import division, print_function
import gym

import urban_AD_env


def test_urban_AD_step():
    env = gym.make('urban_AD-v1')

    env.reset()
    for i in range(3):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
    env.close()

    assert env.observation_space.contains(obs)
    assert 0 <= reward <= 1


def test_merge_step():
    env = gym.make('urban_AD-merge-v1')

    env.reset()
    for i in range(3):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
    env.close()

    assert env.observation_space.contains(obs)
    assert 0 <= reward <= 1


def test_roundabout_step():
    env = gym.make('urban_AD-roundabout-v1')

    env.reset()
    for i in range(3):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
    env.close()

    assert env.observation_space.contains(obs)
    assert 0 <= reward <= 1


def test_parking_step():
    env = gym.make('urban_AD-parking-v1')

    env.reset()
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
    env.close()

    assert action.size == 2
