######################################################################
#          Deep Reinforcement Learning for Autonomous Driving
#                  Created/Modified on: February 7, 2019
#                      Author: Munir Jojo-Verge
#######################################################################

from __future__ import division, print_function, absolute_import
import numpy as np


def goal_distance(goal_a, goal_b, weights=None, scale_factor=1, p=0.5):
    assert goal_a.shape == goal_b.shape
    # if weights == None:
    #     try:
    #         if goal_a.shape[1]:
    #             weights = np.ones((goal_a.shape[1],))        
    #     except: 
    #         weights = np.ones(goal_a.shape)
        
    # a = np.abs(goal_a - goal_b)
    # b = scale_factor * a
    # c = np.dot(b, weights)
    # d = -np.power(c,p)
    # return d
    #return - np.power(np.dot(scale_factor * np.abs(goal_a - goal_b), weights), p)

    return np.linalg.norm(goal_a - goal_b, axis=-1)

def rad(deg):
    return deg*np.pi/180

