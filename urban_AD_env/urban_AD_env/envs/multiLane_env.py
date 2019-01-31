######################################################################
#          Deep Reinforcement Learning for Autonomous Driving
#                  Created/Modified on: January 10, 2019
#                      Author: Munir Jojo-Verge
#######################################################################

from __future__ import division, print_function, absolute_import
import numpy as np
from gym import logger

from urban_AD_env import utils
from urban_AD_env.envs.abstract import AbstractEnv
from urban_AD_env.road.road import Road, RoadNetwork
from urban_AD_env.vehicle.control import EGO_Vehicle
import urban_AD_env.vehicle.vehicle_params as vehicle_params


class MultiLaneEnv(AbstractEnv):
    """
        A urban driving environment.

        The vehicle is driving on a straight urban with several lanes, and is rewarded for reaching a high velocity,
        staying on the rightmost lanes and avoiding collisions.
    """

    COLLISION_REWARD = -1
    """ The reward received when colliding with a vehicle."""
    RIGHT_LANE_REWARD = 0.1
    """ The reward received when driving on the right-most lanes, linearly mapped to zero for other lanes."""
    HIGH_VELOCITY_REWARD = 0.4
    """ The reward received when driving at full speed, linearly mapped to zero for lower speeds."""
    LANE_CHANGE_REWARD = -0
    """ The reward received at each lane change action."""

    DIFFICULTY_LEVELS = {
        "EASY": {
            "lanes_count": 2,
            "initial_spacing": 2,
            "vehicles_count": 5,
            "duration": 20,
            "other_vehicles_type": "urban_AD_env.vehicle.behavior.IDMVehicle",
            "centering_position": [0.3, 0.5]
        },
        "MEDIUM": {
            "lanes_count": 3,
            "initial_spacing": 2,
            "vehicles_count": 10,
            "duration": 30,
            "other_vehicles_type": "urban_AD_env.vehicle.behavior.IDMVehicle",
            "centering_position": [0.3, 0.5]
        },
        "HARD": {
            "lanes_count": 4,
            "initial_spacing": 3,
            "vehicles_count": 50,
            "duration": 40,
            "other_vehicles_type": "urban_AD_env.vehicle.behavior.IDMVehicle",
            "centering_position": [0.3, 0.5]
        },
    }

    def __init__(self):
        super(MultiLaneEnv, self).__init__()
        self.config = self.DIFFICULTY_LEVELS["HARD"].copy()
        self.steps = 0
        self.reset()

    def reset(self):
        self._create_road()
        self._create_vehicles()
        self.steps = 0
        return self._get_obs()

    def step(self, action):
        self.steps += 1
        return super(MultiLaneEnv, self).step(action)

    def set_difficulty_level(self, level):
        if level in self.DIFFICULTY_LEVELS:
            logger.info("Set difficulty level to: {}".format(level))
            self.config.update(self.DIFFICULTY_LEVELS[level])
            self.reset()
        else:
            raise ValueError("Invalid difficulty level, choose among {}".format(str(self.DIFFICULTY_LEVELS.keys())))

    def configure(self, config):
        self.config.update(config)

    def _create_road(self):
        """
            Create a road composed of straight adjacent lanes.
        """
        self.road = Road(network=RoadNetwork.straight_road_network(self.config["lanes_count"]),
                         np_random=self.np_random)

    def _create_vehicles(self):
        """
            Create some new random vehicles of a given type, and add them on the road.
        """
        self.vehicle = EGO_Vehicle.create_random(self.road, 25, spacing=self.config["initial_spacing"])
        self.road.vehicles.append(self.vehicle)

        vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        for _ in range(self.config["vehicles_count"]):
            self.road.vehicles.append(vehicles_type.create_random(self.road))

    def compute_reward(self, action, include_collisions=True):
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :param include_collisions: whether collisions must be penalized in the reward signal
        :return: the corresponding reward

        ACTIONS = {0: 'LANE_LEFT',
               1: 'IDLE',
               2: 'LANE_RIGHT',
               3: 'FASTER',
               4: 'SLOWER'}
        """
        # For simplicity with the working code, let's "translate" the steering and Acc CMDs into "intentions" to calculate the reward
        # To do so, we will make the following assumptions:
        # 1) If you are steering "too much" then we will assume you are making a lane change
        # 2) if you are accelerating "too much" then we will assume that you are performing action 3 = FASTER
        # 3) same if you are braking "too much", 4 = SLOWER
        # Since steering and accel happen simulataneously, we will give priority to the steering
        manouver = 1

        accel = action[1]
        if accel > vehicle_params.MAX_ACCEL * 0.10:            
            manouver = 3
        elif accel < vehicle_params.MAX_DECEL * 0.10:            
            manouver = 4
        
        steering = action[0]
        if abs(steering) > vehicle_params.MAX_STEER_ANGLE * 0.20:
            if steering > 0.0:
                manouver = 0
            else:
                manouver = 2
        

        manouver_reward = {0: self.LANE_CHANGE_REWARD, 1: 0, 2: self.LANE_CHANGE_REWARD, 3: 0, 4: 0}
        
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        state_reward = \
            + self.COLLISION_REWARD * (self.vehicle.crashed and include_collisions) \
            + self.RIGHT_LANE_REWARD * self.vehicle.target_lane_index[2] / (len(neighbours) - 1) \
            + self.HIGH_VELOCITY_REWARD * self.vehicle.velocity_index / (self.vehicle.SPEED_COUNT - 1)
        return utils.remap(manouver_reward[manouver] + state_reward,
                           [self.COLLISION_REWARD * include_collisions, self.HIGH_VELOCITY_REWARD+self.RIGHT_LANE_REWARD],
                           [0, 1])

    def _get_obs(self):
        return super(MultiLaneEnv, self)._get_obs()

    def _is_terminal(self):
        """
            The episode is over if the ego vehicle crashed or the time is out.
        """
        return self.vehicle.crashed or self.steps >= self.config["duration"]

    def _constraint(self, action):
        """
            The constraint signal is the occurrence of collision
        """
        return float(self.vehicle.crashed), self.compute_reward(action, include_collisions=False)
