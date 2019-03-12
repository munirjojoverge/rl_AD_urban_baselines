from __future__ import division, print_function
#import gym

import sys, os
from os.path import dirname, abspath
file_path = sys.argv[0]
# print('sys.argv[0] =', file_path)             
#pathname = os.path.dirname(file_path)        
parent = dirname(dirname(abspath(file_path)))
#print('path =', pathname)
# print('full path =', os.path.abspath(pathname)) 
# print('parent =', parent)

sys.path.insert(0, parent)

from urban_AD_env.envs.parking_env import ParkingEnv
from urban_AD_env.envs.continuous_multi_env import ContinuousMultiEnv
from urban_AD_env.envs.sidepass_env import SidepassEnv
from urban_AD_env.envs.merge_env import MergeEnv

def test_urban_step():
    env = gym.make('urban-v1')

    env.reset()
    for i in range(3):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
    env.close()

    assert env.observation_space.contains(obs)
    assert 0 <= reward <= 1


def test_merge_step(num_runs = 5):
    env = MergeEnv() # gym.make('urban-merge-v1')

    env.reset()
    for i in range(num_runs):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        env.render()

    env.close()

    assert env.observation_space.contains(obs)
    assert 0 <= reward <= 1


def test_roundabout_step(num_runs = 5, max_steps=50):
    env =  ContinuousMultiEnv() #gym.make('urban-roundabout-v1')            
    #env = parking_env.ParkingEnv()
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

def test_urban_sidepass(num_runs = 5, max_steps=50):
    env =  SidepassEnv()

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

    assert env.observation_space.contains(obs)
    assert 0 <= reward <= 1

def test_any_step(env, num_runs = 5, max_steps=50):    
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

if __name__ == '__main__':
    #test_urban_sidepass()
    env =  ParkingEnv()
    test_any_step(env)
