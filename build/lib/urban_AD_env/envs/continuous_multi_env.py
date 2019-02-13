######################################################################
#          Deep Reinforcement Learning for Autonomous Driving
#                  Created/Modified on: February 7, 2019
#                      Author: Munir Jojo-Verge
#######################################################################

from __future__ import division, print_function, absolute_import
import numpy as np
import random as rd
import pandas
from gym import GoalEnv, spaces

from urban_AD_env.envs.abstract import AbstractEnv
from urban_AD_env.envs.build_populate_scenes import _build_roundabout, _build_merge, _build_multilane, _populate_roundabout, _populate_merge, _populate_multilane
from urban_AD_env.envs.graphics import EnvViewer
from urban_AD_env.envs.utils import goal_distance


import urban_AD_env.vehicle.vehicle_params as vehicle_params


class ContinuousMultiEnv(AbstractEnv, GoalEnv):
    """
        A continuous control environment.

        It implements a reach-type task, where the agent observes their position and velocity and must
        control their acceleration and steering so as to reach a given goal.
        The agent gets trained in 3 environments simulataneously (each episode an env is randomly selected and 
        a goal is placed randomly in the road. The Agent must satisfy all goal elements: x, y, Vx, Vy, cos_heading, sin_heading..
        and others might be used)
        
    """

    STEERING_RANGE = np.pi / 4
    ACCELERATION_RANGE = 5.0

    COLLISION_REWARD     = -1.0    
    REVERSE_REWARD       = -0.8
    OFF_ROAD_REWARD      = -0.8    
    AGAINST_TRAIFFIC_REWARD = -0.9  
    HIGH_VELOCITY_REWARD = 0.6
    
    # Reward Weights on the Obs features - below
    DISTANCE_TO_GOAL_REWARD = 2.0
    REWARD_WEIGHTS = [1/100, 1/100, 1/100, 1/100, 1/10, 1/10]
    #REWARD_WEIGHTS = [x * 20.0 for x in REWARD_WEIGHTS]
    
    SUCCESS_GOAL_REWARD = 0.3

    OBS_SCALE = 100
    REWARD_SCALE = np.absolute(COLLISION_REWARD)
    SUCCESS_GOAL_DISTANCE = 1.5
    HEADING_ERR = np.pi / 4

    SCENES =  ['ROUNDABOUT', 'MERGE', 'MULTILANE']   

    SCENE_CONFIG = {
        SCENES[0]: {
            "other_vehicles_type": "urban_AD_env.vehicle.behavior.IDMVehicle",
            "incoming_vehicle_destination": None,
            "centering_position": [0.5, 0.6],        
            "num_vehicles_inside_roundabout": -1,
            "num_vehicles_incoming": -1,
            "num_vehicles_entering": -1,
            "build_scene": _build_roundabout,
            "populate_scene": _populate_roundabout,
            "screen_width": 600,
            "screen_height": 600
        },

        SCENES[1]: {
            "other_vehicles_type": "urban_AD_env.vehicle.behavior.IDMVehicle",
            "centering_position": [0.3, 0.5],
            "vehicles_count": 0,
            "build_scene": _build_merge,
            "populate_scene": _populate_merge,
            "screen_width": 600,
            "screen_height": 150
        },

        SCENES[2]: {
            "lanes_count": 4,
            "initial_spacing": 3,
            "vehicles_count": 0,
            "duration": 40,
            "other_vehicles_type": "urban_AD_env.vehicle.behavior.IDMVehicle",
            "centering_position": [0.3, 0.5],            
            "build_scene": _build_multilane,
            "populate_scene": _populate_multilane,
            "screen_width": 600,
            "screen_height": 150
        }        
    }

    # OBSERVATION_FEATURES = ['x', 'y', 'vx', 'vy', 'cos_h', 'sin_h', 'time_elapsed']
    OBSERVATION_FEATURES = ['x', 'y', 'vx', 'vy', 'cos_h', 'sin_h']
    OBSERVATION_VEHICLES = 1
    NORMALIZE_OBS = False

    

    def __init__(self):
        super(ContinuousMultiEnv, self).__init__()
        
        # self._max_episode_steps = 50
        obs = self.reset()
        self.prev_achieved_goal = obs['achieved_goal'].copy()

        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-np.inf, np.inf, shape=obs["desired_goal"].shape, dtype=np.float32),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs["achieved_goal"].shape, dtype=np.float32),
            observation=spaces.Box(-np.inf, np.inf, shape=obs["observation"].shape, dtype=np.float32),
        ))
        
        self.action_space = spaces.Box(-1., 1., shape=(2,), dtype=np.float32)        


    def step(self, action):
        # Forward action to the vehicle
        self.vehicle.act({"steering": action[0] * self.STEERING_RANGE,
                          "acceleration": action[1] * self.ACCELERATION_RANGE})
        self._simulate()

        obs = self._observation()

        ##### EXTRA INFO
        longitudinal_s = self.vehicle.lane.local_coordinates(self.vehicle.position)[0]    
        lane_heading = self.vehicle.lane.heading_at(longitudinal_s)
        info = {
            "is_success": self._is_success(obs['achieved_goal'], obs['desired_goal']),
            "is_collision": int(self.vehicle.crashed),
            "is_off_road": int(not self.road.network.is_inside_network(self.vehicle.position)),
            "is_reverse": int(self.vehicle.velocity < 0),
            #"prev_distance": float(goal_distance(self.prev_achieved_goal,obs['desired_goal'])),
            "is_against_traffic": int(np.absolute(lane_heading - self.vehicle.heading) > self.HEADING_ERR),
            "velocity_idx": self.vehicle.velocity/self.vehicle.MAX_VELOCITY

        }        
        self.prev_achieved_goal = obs['achieved_goal'].copy()
        
        reward = self.compute_reward(obs['achieved_goal'], obs['desired_goal'], info)
        terminal = self._is_terminal()

        return obs, reward, terminal, info
 

    def _select_scene(self):        
        self.scene = 0 #rd.randrange(0, len(self.SCENES)-1,1)
        self.config = self.SCENE_CONFIG[self.SCENES[self.scene]].copy()
       
        EnvViewer.SCREEN_HEIGHT = self.config['screen_height']
        EnvViewer.SCREEN_WIDTH = self.config['screen_width']  

    def configure(self, config):
        self.config.update(config)

    def _populate_scene(self):
        populate_scene = self.config['populate_scene']
        populate_scene(self)

    def _build_scene(self):
        build_scene = self.config['build_scene']
        build_scene(self)

    def internal_reset(self):
        # Scene
        self.road = None
        self.vehicle = None
        
        # Running
        self.done = False

        # Rendering
        self.viewer = None
        self.automatic_rendering_callback = None
        self.should_update_rendering = True
        self.rendering_mode = 'human'
        self.enable_auto_render = False

    def reset(self):
        self.internal_reset()
        self._select_scene()
        self._build_scene()
        self._populate_scene()
        return self._observation()

    def _observation(self):
                        
        # Add ego-vehicle
        obs = pandas.DataFrame.from_records([self.vehicle.to_dict()])[self.OBSERVATION_FEATURES]
        ego_obs = np.ravel(obs.copy())

        # Add nearby traffic
        close_vehicles = self.road.closest_vehicles_to(self.vehicle, self.OBSERVATION_VEHICLES - 1)
        if close_vehicles:
            obs = obs.append(pandas.DataFrame.from_records(
                [v.to_dict(self.vehicle)
                 for v in close_vehicles[-self.OBSERVATION_VEHICLES+1:]])[self.OBSERVATION_FEATURES],
                           ignore_index=True)

        # Fill missing rows
        if obs.shape[0] < self.OBSERVATION_VEHICLES:
            rows = -np.ones((self.OBSERVATION_VEHICLES - obs.shape[0], len(self.OBSERVATION_FEATURES)))
            obs = obs.append(pandas.DataFrame(data=rows, columns=self.OBSERVATION_FEATURES), ignore_index=True)

        # Reorder
        obs = obs[self.OBSERVATION_FEATURES]
        
        # Flatten
        obs = np.ravel(obs)
        
        # Goal
        goal = np.ravel(pandas.DataFrame.from_records([self.goal.to_dict()])[self.OBSERVATION_FEATURES])

        # Arrange it as required by Openai GoalEnv
        obs = {
            "observation": obs / self.OBS_SCALE,
            "achieved_goal": ego_obs / self.OBS_SCALE,
            "desired_goal": goal / self.OBS_SCALE
        }
        return obs
    
    def distance_2_goal_reward(self, achieved_goal, desired_goal, p=0.5):
        return - np.power(np.dot(self.OBS_SCALE * np.abs(achieved_goal - desired_goal), self.REWARD_WEIGHTS), p)

    def compute_reward(self, achieved_goal, desired_goal, info, p=0.5):
        """
            Proximity to the goal is rewarded

            We use a weighted p-norm
        :param achieved_goal: the goal that was achieved
        :param desired_goal: the goal that was desired
        :param info: any supplementary information
        :param p: the Lp^p norm used in the reward. Use p<1 to have high kurtosis for rewards in [0, 1]
        :return: the corresponding reward
        """
        # DISTANCE TO GOAL
        distance_to_goal_reward = self.distance_2_goal_reward(achieved_goal, desired_goal, p)
        
        # ON/OFF ROAD REWARD        
        off_road_reward = self.OFF_ROAD_REWARD * np.squeeze(info["is_off_road"])

        # COLLISION REWARD
        collision_reward = self.COLLISION_REWARD * np.squeeze(info["is_collision"])
                
        # HIGH VELOCITY REWARD        
        high_vel_reward = self.HIGH_VELOCITY_REWARD * np.squeeze(info["velocity_idx"]) * distance_to_goal_reward        
        
        # REVERESE DRIVING REWARD
        reverse_reward = self.REVERSE_REWARD * np.squeeze(info["is_reverse"])

        # AGAINST TRAFFIC DRIVING REWARD
        against_traffic_reward = self.AGAINST_TRAIFFIC_REWARD * np.squeeze(info["is_against_traffic"])
        
        reward = (distance_to_goal_reward + \
                  off_road_reward + \
                  reverse_reward + \
                  against_traffic_reward + \
                  high_vel_reward +\
                  collision_reward)

        # self.REWARD_SCALE = np.max(np.absolute([distance_to_goal_reward, off_road_reward, reverse_reward, against_traffic_reward, collision_reward]))
        reward /= self.REWARD_SCALE
        #print(reward)
        return reward 

    def _is_success(self, achieved_goal, desired_goal):
        # DISTANCE TO GOAL
        distance_to_goal_reward = self.distance_2_goal_reward(achieved_goal, desired_goal)
        #print(distance_to_goal_reward)
        return distance_to_goal_reward > -self.SUCCESS_GOAL_REWARD

    def compute_reward_2(self, achieved_goal, desired_goal, info):
        """
            Proximity to the goal is rewarded
        :param achieved_goal: the goal that was achieved
        :param desired_goal: the goal that was desired
        :param info: any supplementary information
        :return: the corresponding reward
        """       
        curr_distance = goal_distance(achieved_goal, desired_goal) * self.OBS_SCALE
        distance_to_goal_reward = (-1.0) * curr_distance * self.DISTANCE_TO_GOAL_REWARD 
        
        # HEADING TOWARDS THE GOAL REWARD: ( previous - current distances to goal)
        # prev_distance = np.squeeze(info["prev_distance"] * self.OBS_SCALE)        
        # heading_towards_goal_reward = (prev_distance - curr_distance) * self.DISTANCE_TO_GOAL_REWARD        
        
        # ON/OFF ROAD REWARD        
        off_road_reward = self.OFF_ROAD_REWARD * np.squeeze(info["is_off_road"])

        # COLLISION REWARD
        collision_reward = self.COLLISION_REWARD * np.squeeze(info["is_collision"])
        
        # # HIGH VELOCITY REWARD
        # achieved_speed = np.linalg.norm([achieved_goal[2], achieved_goal[3]])
        # high_vel_reward = self.HIGH_VELOCITY_REWARD * achieved_speed
        
        # REVERESE DRIVING REWARD
        reverse_reward = self.REVERSE_REWARD * np.squeeze(info["is_reverse"])

        reward = (distance_to_goal_reward + \
                  #heading_towards_goal_reward + \
                  off_road_reward + \
                  reverse_reward + \
                  collision_reward)

        self.REWARD_SCALE = np.max(np.absolute([distance_to_goal_reward,off_road_reward, reverse_reward, collision_reward]))

        return reward / self.REWARD_SCALE
    
    def _reward(self, achieved_goal, desired_goal, info):
       raise NotImplementedError     
        
    def _is_success_2(self, achieved_goal, desired_goal):
        # d = goal_distance(achieved_goal, desired_goal) * self.OBS_SCALE
        # return (d < self.SUCCESS_GOAL_DISTANCE).astype(np.float32)
        return np.linalg.norm(achieved_goal - desired_goal) * self.OBS_SCALE < self.SUCCESS_GOAL_DISTANCE
          

    def _is_terminal(self):
        """
            The episode is over if the ego vehicle crashed or the goal is reached.
        """
        # The episode cannot terminate unless all time steps are done. The reason for this is that HER + DDPG uses constant
        # length episodes. If you plan to use other algorithms, please uncomment this line
        return False # self.vehicle.crashed or self._is_success(obs['achieved_goal'], obs['desired_goal'])
    