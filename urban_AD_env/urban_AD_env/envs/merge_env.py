######################################################################
#          Deep Reinforcement Learning for Autonomous Driving
#                  Created/Modified on: January 10, 2019
#                      Author: Munir Jojo-Verge
#######################################################################

from __future__ import division, print_function, absolute_import
import numpy as np

from urban_AD_env import utils
from urban_AD_env.envs.abstract import AD_UrbanEnv
from urban_AD_env.road.lane import LineType, StraightLane, SineLane
from urban_AD_env.road.road import Road, RoadNetwork
from urban_AD_env.vehicle.control import ControlledVehicle, EGO_Vehicle
from urban_AD_env.vehicle.dynamics import Obstacle
import urban_AD_env.vehicle.vehicle_params as vehicle_params

class MergeEnv(AD_UrbanEnv):
    """
        A urban merge negotiation environment.

        The ego-vehicle is driving on a urban and approached a merge, with some vehicles incoming on the access ramp.
        It is rewarded for maintaining a high velocity and avoiding collisions, but also making room for merging
        vehicles.
    """

    COLLISION_REWARD = -1
    RIGHT_LANE_REWARD = 0.1
    HIGH_VELOCITY_REWARD = 0.2
    MERGING_VELOCITY_REWARD = -0.5
    LANE_CHANGE_REWARD = -0.05

    DEFAULT_CONFIG = {"other_vehicles_type": "urban_AD_env.vehicle.behavior.IDMVehicle",
                      "centering_position": [0.3, 0.5]}

    def __init__(self):
        super(MergeEnv, self).__init__()
        self.config = self.DEFAULT_CONFIG.copy()
        self.reset()

    def configure(self, config):
        self.config.update(config)

    def _get_obs(self):
        return super(MergeEnv, self)._get_obs()

    def compute_reward(self, action):
        """
            The vehicle is rewarded for driving with high velocity on lanes to the right and avoiding collisions, but
            an additional altruistic penalty is also suffered if any vehicle on the merging lane has a low velocity.
        :param action: the action performed
        :return: the reward of the state-action transition
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

        manouver_reward = {0: self.LANE_CHANGE_REWARD,
                         1: 0,
                         2: self.LANE_CHANGE_REWARD,
                         3: 0,
                         4: 0}
        reward = self.COLLISION_REWARD * self.vehicle.crashed \
                 + self.RIGHT_LANE_REWARD * self.vehicle.lane_index[2] / 1 \
                 + self.HIGH_VELOCITY_REWARD * self.vehicle.velocity_index / (self.vehicle.SPEED_COUNT - 1)

        # Altruistic penalty
        for vehicle in self.road.vehicles:
            if vehicle.lane_index == ("b", "c", 2) and isinstance(vehicle, ControlledVehicle):
                reward += self.MERGING_VELOCITY_REWARD * \
                          (vehicle.target_velocity - vehicle.velocity) / vehicle.target_velocity

        return utils.remap(manouver_reward[manouver] + reward,
                           [self.COLLISION_REWARD, self.HIGH_VELOCITY_REWARD + self.RIGHT_LANE_REWARD],
                           [0, 1])

    def _is_terminal(self):
        """
            The episode is over when a collision occurs or when the access ramp has been passed.
        """
        return self.vehicle.crashed or self.vehicle.curr_position[0] > 370

    def reset(self):
        self._make_road()
        self._make_vehicles()
        return self._get_obs()

    def _make_road(self):
        """
            Make a road composed of a straight urban and a merging lane.
        :return: the road
        """
        net = RoadNetwork()

        # urban lanes
        ends = [150, 80, 80, 150]  # Before, converging, merge, after
        c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE
        y = [0, StraightLane.DEFAULT_WIDTH]
        line_type = [[c, s], [n, c]]
        line_type_merge = [[c, s], [n, s]]
        for i in range(2):
            net.add_lane("a", "b", StraightLane([0, y[i]], [sum(ends[:2]), y[i]], line_types=line_type[i]))
            net.add_lane("b", "c", StraightLane([sum(ends[:2]), y[i]], [sum(ends[:3]), y[i]], line_types=line_type_merge[i]))
            net.add_lane("c", "d", StraightLane([sum(ends[:3]), y[i]], [sum(ends), y[i]], line_types=line_type[i]))

        # Merging lane
        amplitude = 3.25
        ljk = StraightLane([0, 6.5 + 4 + 4], [ends[0], 6.5 + 4 + 4], line_types=[c, c], forbidden=True)
        lkb = SineLane(ljk.curr_position(ends[0], -amplitude), ljk.curr_position(sum(ends[:2]), -amplitude),
                       amplitude, 2 * np.pi / (2*ends[1]), np.pi / 2, line_types=[c, c], forbidden=True)
        lbc = StraightLane(lkb.curr_position(ends[1], 0), lkb.curr_position(ends[1], 0) + [ends[2], 0],
                           line_types=[n, c], forbidden=True)
        net.add_lane("j", "k", ljk)
        net.add_lane("k", "b", lkb)
        net.add_lane("b", "c", lbc)
        road = Road(network=net, np_random=self.np_random)
        road.vehicles.append(Obstacle(road, lbc.curr_position(ends[2], 0)))
        self.road = road

    def _make_vehicles(self):
        """
            Populate a road with several vehicles on the urban and on the merging lane, as well as an ego-vehicle.
        :return: the ego-vehicle
        """
        road = self.road
        ego_vehicle = EGO_Vehicle(road, road.network.get_lane(("a", "b", 1)).curr_position(30, 0), velocity=30)
        road.vehicles.append(ego_vehicle)

        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        road.vehicles.append(other_vehicles_type(road, road.network.get_lane(("a", "b", 0)).curr_position(90, 0), velocity=29))
        road.vehicles.append(other_vehicles_type(road, road.network.get_lane(("a", "b", 1)).curr_position(70, 0), velocity=31))
        road.vehicles.append(other_vehicles_type(road, road.network.get_lane(("a", "b", 0)).curr_position(5, 0), velocity=31.5))

        merging_v = other_vehicles_type(road, road.network.get_lane(("j", "k", 0)).curr_position(110, 0), velocity=20)
        merging_v.target_velocity = 30
        road.vehicles.append(merging_v)
        self.vehicle = ego_vehicle
