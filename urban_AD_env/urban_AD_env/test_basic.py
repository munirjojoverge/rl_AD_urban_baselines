from __future__ import division, print_function
#import gym

import sys, os
from os.path import dirname, abspath
file_path = sys.argv[0]
# print('sys.argv[0] =', file_path)             
pathname = os.path.dirname(file_path)        
parent = dirname(dirname(abspath(file_path)))
# print('path =', pathname)
# print('full path =', os.path.abspath(pathname)) 
# print('parent =', parent)

sys.path.insert(0, parent)

import urban_AD_env.envs.roundabout_env as R_env

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


def test_roundabout_step(num_runs = 5, max_steps=50):
    env =  R_env.RoundaboutEnv() #gym.make('urban-roundabout-v0')            
    for run in range(num_runs):
        env.reset()
        done = False
        i = 0
        while not done and i <= max_steps:            
            action = env.action_space.sample()
            # print('steering = {:.3f}'.format(action[0]))
            # print('accel    = {:.3f}'.format(action[1]))
            obs, reward, done, info = env.step(action)
            i += 1
                                    
            env.render()
    env.close()

    # assert env.observation_space.contains(obs)
    # assert 0 <= reward <= 1

if __name__ == '__main__':
    test_roundabout_step()