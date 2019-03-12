######################################################################
#          Deep Reinforcement Learning for Autonomous Driving
#                  Created/Modified on: February 7, 2019
#                      Author: Munir Jojo-Verge
#######################################################################

from __future__ import division, print_function, absolute_import
import numpy as np
import random as rd
from gym import logger

from urban_AD_env import utils
from urban_AD_env.envs.abstract import AbstractEnv
from urban_AD_env.road.road import Road, RoadNetwork
from urban_AD_env.road.lane import LineType, StraightLane
from urban_AD_env.vehicle.control import MDPVehicle
from urban_AD_env.envs.graphics import EnvViewer
from urban_AD_env.vehicle.dynamics import Obstacle

class SidepassEnv(AbstractEnv):
    """
        A urban_AD driving environment.

        The vehicle is driving on a straight urban with only 2 lanes (in opposite direction), and is rewarded for 
        reaching a goal (position, speed, heading) that is placed at a certain distance infront of an obstacle located in its lane.
        The vehicle must sidepass the obstacle and avoid collisions with vehicles coming in the opposite direction on the only adjacent lane.
    """

    COLLISION_REWARD = -1
    """ The reward received when colliding with a vehicle."""
    RIGHT_LANE_REWARD = 0.1
    """ The reward received when driving on the right-most lanes, linearly mapped to zero for other lanes."""
    HIGH_VELOCITY_REWARD = 0.4
    """ The reward received when driving at full speed, linearly mapped to zero for lower speeds."""
    LANE_CHANGE_REWARD = -0
    """ The reward received at each lane change action."""

    SCENES =  ['1 lane in each direction',
               '2 lanes on Egos direction and 1 on the opposite'] 
    SCENE_CONFIG = {
        SCENES[0]: {
            "lanes_count_Ego": 3,
            "lanes_count_opposite": 0,
            "road_length": 250,
            "initial_spacing": 2,
            "vehicles_count": 5,
            "default_velocities": [10, 20],
            "duration": 10,
            "other_vehicles_type": "urban_AD_env.vehicle.behavior.IDMVehicle",
            "centering_position": [0.3, 0.5],
            "collision_reward": COLLISION_REWARD,
            "screen_width": 1200,
            "screen_height": 400
        },
        SCENES[1]: {
            "lanes_count_Ego": 2,
            "lanes_count_opposite": 1,
            "road_length": 250,
            "default_velocities": [10, 20],
            "initial_spacing": 2,
            "vehicles_count": 10,
            "duration": 30,
            "other_vehicles_type": "urban_AD_env.vehicle.behavior.IDMVehicle",
            "centering_position": [0.3, 0.5],
            "collision_reward": COLLISION_REWARD,
            "screen_width": 1200,
            "screen_height": 400
        },        
    }

    def __init__(self):
        super(SidepassEnv, self).__init__()        
        self.steps = 0
        self.reset()
    
    def _select_scene(self, scene=None):        
        if scene==None:            
            self.scene = rd.randrange(0, len(self.SCENES)-1,1)
        elif scene in range(0,len(self.SCENES)):
            self.scene = scene
        else:
            raise ValueError("Invalid scene id. Choose among {}".format(str(range(0,len(self.SCENES)))))

        self.config = self.SCENE_CONFIG[self.SCENES[self.scene]].copy()
       
        EnvViewer.SCREEN_HEIGHT = self.config['screen_height']
        EnvViewer.SCREEN_WIDTH = self.config['screen_width']  

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
        self._select_scene(0)
        self._build_scene()
        self._populate_scene()
        self.steps = 0
        return self._observation()
    
    def step(self, action):
        self.steps += 1
        return super(SidepassEnv, self).step(action)
    
    def _build_scene(self):
        """
            Create a road composed of straight adjacent lanes.
        """        
        self.road = Road(network=RoadNetwork.straight_road_network(self.config["lanes_count_Ego"]),
                         np_random=self.np_random)

        # length = self.config["road_length"]        
        # c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE
        # net = RoadNetwork()

        # for lane in range(self.config["lanes_count_Ego"]):
        #     origin = [0, lane * StraightLane.DEFAULT_WIDTH]
        #     end = [length, lane* StraightLane.DEFAULT_WIDTH]
        #     line_types = [s, c if lane == self.config["lanes_count_Ego"] - 1 else n]
        #     net.add_lane('a', 'b', StraightLane(origin, end, line_types=line_types))


        # for lane in range(1, self.config["lanes_count_opposite"]+1):
        #     origin = [length, -lane * StraightLane.DEFAULT_WIDTH]
        #     end = [0, -lane * StraightLane.DEFAULT_WIDTH]
        #     line_types = [n if lane == 1 else s, c if lane == self.config["lanes_count_opposite"] else n]
        #     net.add_lane('a', 'b', StraightLane(origin, end, line_types=line_types))

        # self.road = Road(network=net, np_random=self.np_random)
        
        

    def _populate_scene(self):
        """
            Create some new random vehicles of a given type, and add them on the road.
        """        
        ### Ego ###
        ego_vehicle = MDPVehicle.create_random(self.road, 25, spacing=self.config["initial_spacing"])
        ego_vehicle.lane
        self.vehicle = ego_vehicle
        self.road.vehicles.append(self.vehicle)        
        # right_most_lane = self.config["lanes_count_Ego"]-1
        # lane = self.road.network.get_lane(('a', 'b', right_most_lane))
        # ego_vehicle = MDPVehicle(self.road, lane.position(30, 0), heading=lane.heading, velocity=10)
        # self.vehicle = ego_vehicle
        # self.road.vehicles.append(ego_vehicle)

        ### Obstacle ###
        obstacle = Obstacle(self.road, ego_vehicle.position+[60,0])
        self.road.vehicles.append(obstacle)
        if ego_vehicle.lane == self.config["lanes_count_Ego"]-1:
            obstacle = Obstacle(self.road, ego_vehicle.position+[120, -StraightLane.DEFAULT_WIDTH])
        else:
            obstacle = Obstacle(self.road, ego_vehicle.position+[120, -StraightLane.DEFAULT_WIDTH])
        self.road.vehicles.append(obstacle)

         ### Other Vehicles
        # vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        # for _ in range(self.config["vehicles_count"]):
        #     self.road.vehicles.append(vehicles_type.create_random(self.road))

        ### Other Vehicles
        vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        for _ in range(self.config["vehicles_count"]):
            self.road.vehicles.append(vehicles_type.create_random(self.road))




        # other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        # _from = 'a'
        # _to = 'b'        
        # spacing = 60
        # DEFAULT_VELOCITIES = self.config["default_velocities"]

        # for _ in range(self.config["vehicles_count"]):            
        #     _id = self.road.np_random.choice(len(self.road.network.graph[_from][_to]))
        #     lane = self.road.network.get_lane((_from, _to, _id))
        #     velocity = self.road.np_random.uniform(DEFAULT_VELOCITIES[0], DEFAULT_VELOCITIES[1])
        #     default_spacing = 1.5*velocity
        #     offset = spacing + default_spacing * np.exp(-5 / 30 * len(self.road.network.graph[_from][_to]))
        #     x0 = np.max([v.position[0] for v in self.road.vehicles]) if len(self.road.vehicles) else 3*offset
        #     x0 += offset * self.road.np_random.uniform(0.9, 1.1)        
        #     self.road.vehicles.append(other_vehicles_type(self.road, position=lane.position(x0, 0), heading=lane.heading, velocity=velocity))
        
        #self.road.vehicles.append(vehicles_type(road, road.network.get_lane(('b', 'a', 1)).position(30, 0), velocity=10))  

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
            + self.HIGH_VELOCITY_REWARD * self.vehicle.velocity_index / (self.vehicle.SPEED_COUNT - 1) \
            + self.RIGHT_LANE_REWARD * self.vehicle.target_lane_index[2] / (len(neighbours) - 1)
        return utils.remap(action_reward[action] + state_reward,
                           [self.config["collision_reward"], self.HIGH_VELOCITY_REWARD+self.RIGHT_LANE_REWARD],
                           [0, 1])

    def _observation(self):
        return super(SidepassEnv, self)._observation()

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
