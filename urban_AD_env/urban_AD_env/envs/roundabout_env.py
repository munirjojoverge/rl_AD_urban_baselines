######################################################################
#          Deep Reinforcement Learning for Autonomous Driving
#                  Created/Modified on: January 10, 2019
#                      Author: Munir Jojo-Verge
#######################################################################

from __future__ import division, print_function, absolute_import
import numpy as np

from urban_AD_env import utils
#import utils
from urban_AD_env.envs.abstract import AD_UrbanEnv
from urban_AD_env.envs.graphics import EnvViewer
from urban_AD_env.road.lane import LineType, StraightLane, CircularLane, SineLane
from urban_AD_env.road.road import Road, RoadNetwork
from urban_AD_env.vehicle.control import EGO_Vehicle
import urban_AD_env.vehicle.vehicle_params as vehicle_params

class RoundaboutEnv(AD_UrbanEnv):

    COLLISION_REWARD     = -100.0
    HIGH_VELOCITY_REWARD = 0.2 # Only if it's inside the rode
    RIGHT_LANE_REWARD    = 0
    LANE_CHANGE_REWARD   = -0.05
    REVERSE_REWARD       = -5.0
    OFF_ROAD_REWARD      = -10.0
    DISTANCE_TO_GOAL_REWARD = 1.0  
    GOAL_REACHED_REWARD  = 100.0

    DURATION = 90

    DEFAULT_CONFIG = {"other_vehicles_type": "urban_AD_env.vehicle.behavior.IDMVehicle",
                      "incoming_vehicle_destination": None,
                      "centering_position": [0.5, 0.6]}

    def __init__(self):        
        self.num_vehicles = 4
        super(RoundaboutEnv, self).__init__()
        self.config = self.DEFAULT_CONFIG.copy()        
        self.steps = 0
        self.reset()
        
        EnvViewer.SCREEN_HEIGHT = 600
        

    def configure(self, config):
        self.config.update(config)

    def _get_obs(self):
        return super(RoundaboutEnv, self)._get_obs()

    def render(self,mode='human'):
        super(RoundaboutEnv, self).render(mode)
        
    
    def compute_reward(self, achieved_goal, desired_goal, info={}):
        """            
        """  
        # Compute distance between goal and the achieved goal.        
        try:
            if achieved_goal.shape[1]:
                reward =  (-1.0)* self.goal_distance(achieved_goal[:,:2], desired_goal[:,:2])      
        except:
            reward = (-1.0)* self.goal_distance(achieved_goal[:2], desired_goal[:2])
            
        # #################################################################################3
        # #### EGO Distance to Goal (position Error)
        # ego_curr_pos = np.array([achieved_goal[0], achieved_goal[1]]) # [x, y]                       
        # ego_goal_pos = np.array([desired_goal[0], desired_goal[1]]) # [x, y]                               

        # self.vehicle.distance_to_goal = self.goal_distance(ego_goal_pos, ego_curr_pos)
        # prev_distance_to_goal = self.goal_distance(ego_goal_pos, self.vehicle.prev_position)
        

        # # HEADING TOWARDS THE GOAL REWARD: (current - previous distances to goal)
        # self.heading_towards_goal_reward = (prev_distance_to_goal - self.vehicle.distance_to_goal) * self.DISTANCE_TO_GOAL_REWARD * 2

        # # DECREASING OR INCREADING THE DISTANCE WRT INITIAL POSITION
        # if self.ini_dist_to_goal < 0.1:
        #    reward_factor = 1
        # else:
        #     reward_factor = ( (-1/self.ini_dist_to_goal) * self.vehicle.distance_to_goal + 1)
            
        # self.reward_wrt_initial_dist_to_goal = reward_factor * self.DISTANCE_TO_GOAL_REWARD

        # self.vehicle.distance_goal_reward = self.heading_towards_goal_reward # + self.reward_wrt_initial_dist_to_goal

        
        # # ON/OFF ROAD REWARD
        # self.vehicle.is_on_the_road = self.road.network.is_inside_network(self.vehicle.curr_position)
        # self.vehicle.off_road_reward = self.OFF_ROAD_REWARD * int(not self.vehicle.is_on_the_road)

        # # COLLISION REWARD
        # self.vehicle.collision_reward = self.COLLISION_REWARD * self.vehicle.crashed * abs(self.vehicle.velocity)
        
        # # HIGH VELOCITY REWARD
        # self.vehicle.high_vel_reward = self.HIGH_VELOCITY_REWARD * self.vehicle.velocity_index / max(self.vehicle.SPEED_COUNT - 1, 1)
        
        # # REVERESE DRIVING REWARD
        # self.vehicle.reverse_reward = self.REVERSE_REWARD * int(self.vehicle.velocity < 0)

        # # LANE CHANGE REWARD
        # #self.vehicle.lane_change_reward = self.LANE_CHANGE_REWARD * (manouver in [0, 2])
        
        
        # # return utils.remap(reward, [self.COLLISION_REWARD+self.LANE_CHANGE_REWARD, self.HIGH_VELOCITY_REWARD], [0, 1])
        # #reward = self.vehicle.collision_reward + self.vehicle.high_vel_reward + self.vehicle.lane_change_reward + self.vehicle.off_road_reward + self.vehicle.distance_goal_reward

        # reward = self.vehicle.distance_goal_reward + \
        #          self.vehicle.reverse_reward
        #         #  self.vehicle.off_road_reward + \
        #         #  self.vehicle.collision_reward

        # # GOAL REACHED REWARD
        # if np.any(reward) < self.distance_threshold:
        #     reward += self.GOAL_REACHED_REWARD


        self.vehicle.reward = reward

        return reward

    def _is_terminal(self, achieved_goal, desired_goal, info={}):
        """
            The episode is over when:
            1) a collision occurs             
            2) Ego has reached the Goal
            3) Too many steps
        """
        d = self.goal_distance(achieved_goal[:2], desired_goal[:2])

        return self.vehicle.crashed or \
               d < self.distance_threshold or \
               self.steps >= self.DURATION

    def _is_success(self, achieved_goal, desired_goal):
        """
            The episode is succesful:            
            1) Ego has reached the Goal            
        """        
        return (self.goal_distance(achieved_goal[:2], desired_goal[:2]) < self.distance_threshold).astype(np.float32)

    
    def reset(self):
        self._make_road()
        self._make_vehicles()
        self.steps = 0
        return self._get_obs()

    def step(self, action):
        self.steps += 1
        return super(RoundaboutEnv, self).step(action)

    def _make_road(self):
        # Circle lanes: (s)outh/(e)ast/(n)orth/(w)est (e)ntry/e(x)it.
        center = [0, 0]  # [m]
        radius = 30  # [m]
        alpha = 20  # [deg]

        net = RoadNetwork()
        radii = [radius, radius+4]
        n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED
        line = [[c, s], [n, c]]
        for lane in [0, 1]:
            net.add_lane("se", "ex", CircularLane(center, radii[lane], rad(90-alpha), rad(alpha), line_types=line[lane]))
            net.add_lane("ex", "ee", CircularLane(center, radii[lane], rad(alpha), rad(-alpha), line_types=line[lane]))
            net.add_lane("ee", "nx", CircularLane(center, radii[lane], rad(-alpha), rad(-90+alpha), line_types=line[lane]))
            net.add_lane("nx", "ne", CircularLane(center, radii[lane], rad(-90+alpha), rad(-90-alpha), line_types=line[lane]))
            net.add_lane("ne", "wx", CircularLane(center, radii[lane], rad(-90-alpha), rad(-180+alpha), line_types=line[lane]))
            net.add_lane("wx", "we", CircularLane(center, radii[lane], rad(-180+alpha), rad(-180-alpha), line_types=line[lane]))
            net.add_lane("we", "sx", CircularLane(center, radii[lane], rad(180-alpha), rad(90+alpha), line_types=line[lane]))
            net.add_lane("sx", "se", CircularLane(center, radii[lane], rad(90+alpha), rad(90-alpha), line_types=line[lane]))

        # Access lanes: (r)oad/(s)ine
        access = 200  # [m]
        dev = 120  # [m]
        a = 5  # [m]
        delta_st = 0.20*dev  # [m]

        delta_en = dev-delta_st
        w = 2*np.pi/dev
        net.add_lane("ser", "ses", StraightLane([2, access], [2, dev/2], line_types=[s, c]))
        net.add_lane("ses", "se", SineLane([2+a, dev/2], [2+a, dev/2-delta_st], a, w, -np.pi/2, line_types=[c, c]))
        net.add_lane("sx", "sxs", SineLane([-2-a, -dev/2+delta_en], [-2-a, dev/2], a, w, -np.pi/2+w*delta_en, line_types=[c, c]))
        net.add_lane("sxs", "sxr", StraightLane([-2, dev / 2], [-2, access], line_types=[n, c]))

        net.add_lane("eer", "ees", StraightLane([access, -2], [dev / 2, -2], line_types=[s, c]))
        net.add_lane("ees", "ee", SineLane([dev / 2, -2-a], [dev / 2 - delta_st, -2-a], a, w, -np.pi / 2, line_types=[c, c]))
        net.add_lane("ex", "exs", SineLane([-dev / 2 + delta_en, 2+a], [dev / 2, 2+a], a, w, -np.pi / 2 + w * delta_en, line_types=[c, c]))
        net.add_lane("exs", "exr", StraightLane([dev / 2, 2], [access, 2], line_types=[n, c]))

        net.add_lane("ner", "nes", StraightLane([-2, -access], [-2, -dev / 2], line_types=[s, c]))
        net.add_lane("nes", "ne", SineLane([-2 - a, -dev / 2], [-2 - a, -dev / 2 + delta_st], a, w, -np.pi / 2, line_types=[c, c]))
        net.add_lane("nx", "nxs", SineLane([2 + a, dev / 2 - delta_en], [2 + a, -dev / 2], a, w, -np.pi / 2 + w * delta_en, line_types=[c, c]))
        net.add_lane("nxs", "nxr", StraightLane([2, -dev / 2], [2, -access], line_types=[n, c]))
        
        road = Road(network=net, np_random=self.np_random)
        self.road = road

    def _make_vehicles(self):
        """
            Populate a road with several vehicles on the urban and on the merging lane, as well as an ego-vehicle.
        :return: the ego-vehicle
        """        
        ################### ADDING EGO VEHICLE ###################
        # Ego-vehicle
        ego_lane = self.road.network.get_lane(("ser", "ses", 0))
        
        # Ego Initial State
        ego_ini_pos  = ego_lane.position(140, 0)
        ego_ini_heading = 0.0 # ego_lane.heading_at(140)
        ego_ini_vel = 5 # m/s
        ego_ini_state = np.array([ego_ini_pos[0], ego_ini_pos[1], ego_ini_heading, ego_ini_vel]) # [x, y, heading, velocity]         

        # Ego Goal State
        ego_goal_pos  = np.array([0, 0]) #np.array([ego_ini_pos[0], -ego_ini_pos[1]])
        ego_goal_heading = ego_ini_heading
        ego_goal_vel = 10 # m/s
        ego_goal_state = np.array([ego_goal_pos[0], ego_goal_pos[1], ego_goal_heading, ego_goal_vel]) # [x, y, heading, velocity]         
              
        ego_vehicle = EGO_Vehicle(self.road,
                                 ego_ini_state,
                                 ego_goal_state).plan_route_to("nxs")        
        EGO_Vehicle.SPEED_MIN = 0
        EGO_Vehicle.SPEED_MAX = 15
        EGO_Vehicle.SPEED_COUNT = 4
        self.road.vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle
        
        # To be able to run HER, the environment needs a goal state (Look at abstract.py)
        self.ego_goal_state = ego_goal_state
        # For now I'm just leaving this as it is, where the Distance includes heading and velocity as a 3rd and 4th dimention
        self.ini_dist_to_goal = self.goal_distance(self.ego_goal_state,ego_ini_state)
        
        # ################### ADDING OTHER VEHICLES ###################
        # # Incoming vehicle
        # destinations = ["exr", "sxr", "nxr"]
        # other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        # vehicle = other_vehicles_type.make_on_lane(self.road,
        #                                            ("we", "sx", 1),
        #                                            longitudinal=5 + self.np_random.randn()*position_deviation,
        #                                            velocity=16 + self.np_random.randn()*velocity_deviation)

        # if self.config["incoming_vehicle_destination"] is not None:
        #     destination = destinations[self.config["incoming_vehicle_destination"]]
        # else:
        #     destination = self.np_random.choice(destinations)
        # vehicle.plan_route_to(destination)
        # vehicle.randomize_behavior()
        # self.road.vehicles.append(vehicle)

        # # Other vehicles
        # for i in list(range(1, self.num_vehicles)) + list(range(-1, 0)):
        #     vehicle = other_vehicles_type.make_on_lane(self.road,
        #                                                ("we", "sx", 0),
        #                                                longitudinal=20*i + self.np_random.randn()*position_deviation,
        #                                                velocity=16 + self.np_random.randn()*velocity_deviation)
        #     vehicle.plan_route_to(self.np_random.choice(destinations))
        #     vehicle.randomize_behavior()
        #     self.road.vehicles.append(vehicle)

        # # Entering vehicle
        # vehicle = other_vehicles_type.make_on_lane(self.road,
        #                                            ("eer", "ees", 0),
        #                                            longitudinal=50 + self.np_random.randn() * position_deviation,
        #                                            velocity=16 + self.np_random.randn() * velocity_deviation)
        # vehicle.plan_route_to(self.np_random.choice(destinations))
        # vehicle.randomize_behavior()
        # self.road.vehicles.append(vehicle)


def rad(deg):
    return deg*np.pi/180