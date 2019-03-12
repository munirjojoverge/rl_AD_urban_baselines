from __future__ import division, print_function, absolute_import
import numpy as np
from gym import logger

from urban_AD_env import utils
from urban_AD_env.envs.abstract import AbstractEnv
from urban_AD_env.road.road import Road, RoadNetwork
from urban_AD_env.road.lane import LineType, StraightLane
from urban_AD_env.vehicle.control import MDPVehicle
from urban_AD_env.vehicle.dynamics import Obstacle
from urban_AD_env.envs.graphics import EnvViewer


class MultiLaneEnv(AbstractEnv):
    """
        A urban_AD driving environment.

        The vehicle is driving on a straight urban road with several lanes, and is rewarded for reaching a high velocity,
        staying on the rightmost lanes and avoiding collisions.
    """

    COLLISION_REWARD = -1
    """ The reward received when colliding with a vehicle."""
    RIGHT_LANE_REWARD = 0.1
    """ The reward received when driving on the right-most lanes, linearly mapped to zero for other lanes."""
    HIGH_VELOCITY_REWARD = 0.4
    """ The reward received when driving at full speed, linearly mapped to zero for lower speeds."""
    LANE_CHANGE_REWARD = -0.0
    """ The reward received at each lane change action."""

    DIFFICULTY_LEVELS = {
        "EASY": {
            "lanes_count": 2,
            "initial_spacing": 2,
            "vehicles_count": 5,
            "vehicle_radious": 100,         
            "obstacles_count": 20,
            "obstacles_radious": 100,
            "duration": 20,
            "other_vehicles_type": "urban_AD_env.vehicle.behavior.IDMVehicle",
            "centering_position": [0.3, 0.5],
            "collision_reward": COLLISION_REWARD,
            "screen_width": 1200,
            "screen_height": 600
        },
        "MEDIUM": {
            "lanes_count": 3,
            "initial_spacing": 2,
            "vehicles_count": 10,
            "vehicle_radious": 100,         
            "obstacles_count": 20,
            "obstacles_radious": 100,
            "duration": 30,
            "other_vehicles_type": "urban_AD_env.vehicle.behavior.IDMVehicle",
            "centering_position": [0.3, 0.5],
            "collision_reward": COLLISION_REWARD,
            "screen_width": 1200,
            "screen_height": 600
        },
        "HARD": {
            "lanes_count": 4,
            "initial_spacing": 3,
            "vehicles_count": 10,
            "vehicle_radious": 100,         
            "obstacles_count": 20,
            "obstacles_radious": 100,
            "duration": 30,
            "other_vehicles_type": "urban_AD_env.vehicle.behavior.IDMVehicle",
            "centering_position": [0.3, 0.5],
            "collision_reward": COLLISION_REWARD,
            "screen_width": 1200,
            "screen_height": 600
        },
    }

    def __init__(self):
        super(MultiLaneEnv, self).__init__()
        self.config = self.DIFFICULTY_LEVELS["HARD"].copy()
        self.steps = 0
        self.reset()
        EnvViewer.SCREEN_HEIGHT = self.config['screen_height']
        EnvViewer.SCREEN_WIDTH = self.config['screen_width']  


    def reset(self):
        self._create_road()
        self._create_vehicles()
        self.steps = 0
        return self._observation()

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
        ### Ego ###
        ego_vehicle = MDPVehicle.create_random(self.road, 25, spacing=self.config["initial_spacing"])
        ego_vehicle.lane
        self.vehicle = ego_vehicle
        self.road.vehicles.append(self.vehicle)   

        ### Obstacle ###
        obstacle = Obstacle(self.road, ego_vehicle.position+[60,0])
        self.road.vehicles.append(obstacle)
        for _ in range(self.config["obstacles_count"]):
            self.road.vehicles.append(Obstacle.create_random(self.road))

        # if ego_vehicle.lane == self.config["lanes_count"]-1:
        #     obstacle = Obstacle(self.road, ego_vehicle.position+[120, -StraightLane.DEFAULT_WIDTH])
        # else:
        #     obstacle = Obstacle(self.road, ego_vehicle.position+[120, -StraightLane.DEFAULT_WIDTH])
        # self.road.vehicles.append(obstacle)

        ### Other Vehicles
        vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])        
        for _ in range(self.config["vehicles_count"]):
            self.road.vehicles.append(vehicles_type.create_random(self.road))

    def _reward(self, action):
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        action_reward = {0: self.LANE_CHANGE_REWARD, 1: 0, 2: self.LANE_CHANGE_REWARD, 3: 0, 4: 0}
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        state_reward = \
            + self.config["collision_reward"] * self.vehicle.crashed \
            + self.RIGHT_LANE_REWARD * self.vehicle.target_lane_index[2] / (len(neighbours) - 1) \
            + self.HIGH_VELOCITY_REWARD * self.vehicle.velocity_index / (self.vehicle.SPEED_COUNT - 1)
        return utils.remap(action_reward[action] + state_reward,
                           [self.config["collision_reward"], self.HIGH_VELOCITY_REWARD+self.RIGHT_LANE_REWARD],
                           [0, 1])

    def _observation(self):
        return super(MultiLaneEnv, self)._observation()

    def _is_terminal(self):
        """
            The episode is over if the ego vehicle crashed or the time is out.
        """
        return self.vehicle.crashed or self.steps >= self.config["duration"]

    def _constraint(self, action):
        """
            The constraint signal is the occurrence of collision
        """
        return float(self.vehicle.crashed)
