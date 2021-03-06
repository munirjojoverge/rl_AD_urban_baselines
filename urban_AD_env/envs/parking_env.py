from __future__ import division, print_function, absolute_import
import numpy as np
import pandas
from gym import GoalEnv, spaces

from urban_AD_env.envs.abstract import AbstractEnv
from urban_AD_env.envs.graphics import EnvViewer
from urban_AD_env.road.lane import StraightLane, LineType
from urban_AD_env.road.road import Road, RoadNetwork
from urban_AD_env.vehicle.dynamics import Vehicle, Obstacle
from urban_AD_env.vehicle.control import MDPVehicle



class ParkingEnv(AbstractEnv, GoalEnv):
    """
        A continuous control environment.

        It implements a reach-type task, where the agent observes their position and velocity and must
        control their acceleration and steering so as to reach a given goal.

        Credits to Munir Jojo-Verge for the idea and initial implementation.
    """

    STEERING_RANGE = np.pi / 4
    ACCELERATION_RANGE = 5.0

    COLLISION_REWARD     = -1.0        
    MOVING_REWARD        = +0.2
    OVER_OTHER_PARKING_SPOT_REWARD = -0.2
    REACHING_GOAL_REWARD = +1.0

    PARKING_MAX_VELOCITY = 7.0 # m/s

    OBS_SCALE = 100
    REWARD_SCALE = np.absolute(COLLISION_REWARD)

    REWARD_WEIGHTS = [5/100, 5/100, 1/100, 1/100, 5/10, 5/10]
    SUCCESS_THRESHOLD = 0.27

    DEFAULT_CONFIG = {        
        "other_vehicles_type": "urban_AD_env.vehicle.behavior.IDMVehicle",
        "centering_position": [0.5, 0.5],
        "parking_spots": 15, #'random', # Parking Spots Per side
        "vehicles_count": 0, #'random', # Total number of cars  
        "screen_width": 600 * 2,
        "screen_height": 600 * 2        
    }    

    OBSERVATION_FEATURES = ['x', 'y', 'vx', 'vy', 'cos_h', 'sin_h']

    OBSERVATION_NEAR_EGO = 0 # How many vehicles "near" EGO you want to include in the "observation"/"state space"
    OBSERVATION_NEAR_GOAL = 0 # How many vehicles "near" GOAL you want to include in the "observation"/"state space" (Testing a better awarnes around the goal)
    NORMALIZE_OBS = False

    def __init__(self):
        super(ParkingEnv, self).__init__()
        self.config = self.DEFAULT_CONFIG.copy()
        if self.config["parking_spots"] == 'random':
            self.parking_spots = self.np_random.randint(1,21)
        else:
            self.parking_spots = self.config["parking_spots"]

        if self.config["vehicles_count"] == 'random':
            self.vehicles_count = self.np_random.randint(self.parking_spots) * 2
        else:
            self.vehicles_count = self.config["vehicles_count"]
        assert (self.vehicles_count < self.parking_spots*2)

        obs = self.reset()
        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-np.inf, np.inf, shape=obs["desired_goal"].shape, dtype=np.float32),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs["achieved_goal"].shape, dtype=np.float32),
            observation=spaces.Box(-np.inf, np.inf, shape=obs["observation"].shape, dtype=np.float32),
        ))
        self.action_space = spaces.Box(-1., 1., shape=(2,), dtype=np.float32)
        self.REWARD_WEIGHTS = np.array(self.REWARD_WEIGHTS)
        EnvViewer.SCREEN_HEIGHT = EnvViewer.SCREEN_WIDTH // 2        

        

    def step(self, action):
        # Forward action to the vehicle
        # self.vehicle.act({"steering": action[0] * self.STEERING_RANGE,
        #                   "acceleration": action[1] * self.ACCELERATION_RANGE})

        self.vehicle.act({
            "acceleration": action[0] * self.ACCELERATION_RANGE,
            "steering": action[1] * self.STEERING_RANGE
        })
        self._simulate()

        obs = self._observation()
        info = {
            "is_success": self._is_success(obs['achieved_goal'], obs['desired_goal']),
            "is_collision": int(self.vehicle.crashed),
            "is_over_others_parking_spot": int(self.is_over_others_parking_spot(self.vehicle.position)),
            "velocity_idx": self.vehicle.velocity/self.PARKING_MAX_VELOCITY
        }

        reward = self.compute_reward(obs['achieved_goal'], obs['desired_goal'], info)
        terminal = self._is_terminal()
        return obs, reward, terminal, info

    def is_over_others_parking_spot(self, position):        
        for _from, to_dict in self.road.network.graph.items():
            for _to, lanes in to_dict.items():
                for _id, lane in enumerate(lanes):
                    if lane != self.goal.lane:
                        over_others_parking_spots = lane.on_lane(position)
                    if (over_others_parking_spots):
                        return True
                    
        return False


    def reset(self):
        self._build_parking()
        self._populate_parking()
        return self._observation()

    def configure(self, config):
        self.config.update(config)

    def rot(self, point, angle):
        assert len(point)==2
        x = point[0]
        y = point[1]
        cos_ang = np.cos(angle)
        sin_ang = np.sin(angle)
        x_rot = x*cos_ang - y*sin_ang
        y_rot = x*sin_ang + y*cos_ang
        return x_rot, y_rot

    def _build_parking(self):
        """
            Create a road composed of straight adjacent lanes.
            We will have 4 parking configurations based on (parking angle):
            https://www.webpages.uidaho.edu/niatt_labmanual/chapters/parkinglotdesign/theoryandconcepts/parkingstalllayoutconsiderations.htm
            parking angle = 90, 75, 60, 45
        """
        net = RoadNetwork()
        
        lt = (LineType.CONTINUOUS, LineType.CONTINUOUS)
        
        spots_offset = 0.0
        parking_angles = np.deg2rad([90, 75, 60, 45, 0])
        aisle_width = 10.0
        length = 8.0
        width = 4.0
        
        # Let's start by randomly choosing the parking angle        
        #angle = parking_angles[self.np_random.randint(len(parking_angles))]        
        angle = 0 #np.pi/3
        # Let's now build the parking lot
        for k in range(self.parking_spots):
            x1 = (k - self.parking_spots // 2) * (width + spots_offset) - width / 2
            y1 = aisle_width/2            
            x2 = x1
            y2 = y1 + length

            x3 = x1
            y3 = -y1
            x4 = x3
            y4 = -y2

            x1, y1 = self.rot((x1,y1), angle)
            x2, y2 = self.rot((x2,y2), angle)

            x3, y3 = self.rot((x3,y3), angle)
            x4, y4 = self.rot((x4,y4), angle)
            
            net.add_lane("a", "b", StraightLane([x1, y1], [x2, y2], width=width, line_types=lt))
            net.add_lane("b", "c", StraightLane([x3, y3], [x4, y4], width=width, line_types=lt))

        self.road = Road(network=net,
                         np_random=self.np_random)

    def _populate_parking(self):
        """
            Create some new random vehicles of a given type, and add them on the road.
        """
        ##### ADDING EGO #####
        self.vehicle = MDPVehicle(self.road, [0, 0], 2*np.pi*self.np_random.rand(), velocity=0)
        self.vehicle.MAX_VELOCITY = self.PARKING_MAX_VELOCITY
        self.road.vehicles.append(self.vehicle)
        
        ##### ADDING GOAL #####
        parking_spots_used =[]
        lane = self.np_random.choice(self.road.network.lanes_list())
        parking_spots_used.append(lane)
        goal_heading = lane.heading #+ self.np_random.randint(2) * np.pi
        self.goal = Obstacle(self.road, lane.position(lane.length/2, 0), heading=goal_heading)
        self.goal.COLLISIONS_ENABLED = False
        self.road.vehicles.insert(0, self.goal)

        ##### ADDING OTHER VEHICLES #####
        # vehicles_type = utils.class_from_path(scene.config["other_vehicles_type"])             
        for _ in range(self.vehicles_count):            
            while lane in parking_spots_used: # this loop should never be infinite since we assert that there should be more parking spots/lanes than vehicles
                lane = self.np_random.choice(self.road.network.lanes_list()) # to-do: chceck for empty spots
            parking_spots_used.append(lane)
            
            vehicle_heading = lane.heading #+ self.np_random.randint(2) * np.pi
            self.road.vehicles.append(Vehicle(self.road, lane.position(lane.length/2, 0), heading=vehicle_heading, velocity=0))

    def _observation(self):

        ##### ADDING EGO #####
        obs = pandas.DataFrame.from_records([self.vehicle.to_dict()])[self.OBSERVATION_FEATURES]
        ego_obs = np.ravel(obs.copy())

        ##### ADDING NEARBY (TO EGO) TRAFFIC #####
        close_vehicles = self.road.closest_vehicles_to(self.vehicle, self.OBSERVATION_NEAR_EGO)
        if close_vehicles:
            obs = obs.append(pandas.DataFrame.from_records(
                [v.to_dict(self.vehicle)
                 for v in close_vehicles])[self.OBSERVATION_FEATURES],
                           ignore_index=True)

        # Fill missing rows
        needed = self.OBSERVATION_NEAR_EGO + 1
        missing = needed - obs.shape[0]
        if obs.shape[0] < (needed):
            rows = -np.ones((missing, len(self.OBSERVATION_FEATURES)))
            obs = obs.append(pandas.DataFrame(data=rows, columns=self.OBSERVATION_FEATURES), ignore_index=True)

        ##### ADDING NEARBY (TO GOAL) TRAFFIC #####
        close_vehicles = self.road.closest_vehicles_to(self.goal, self.OBSERVATION_NEAR_GOAL)
        if close_vehicles:
            obs = obs.append(pandas.DataFrame.from_records(
                [v.to_dict(self.vehicle)
                 for v in close_vehicles])[self.OBSERVATION_FEATURES],
                           ignore_index=True)

        # Fill missing rows
        needed = self.OBSERVATION_NEAR_EGO + self.OBSERVATION_NEAR_GOAL + 1
        missing = needed - obs.shape[0]
        if obs.shape[0] < (needed):
            rows = -np.ones((missing, len(self.OBSERVATION_FEATURES)))
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

        # obs = np.ravel(pandas.DataFrame.from_records([self.vehicle.to_dict()])[self.OBSERVATION_FEATURES])
        # goal = np.ravel(pandas.DataFrame.from_records([self.goal.to_dict()])[self.OBSERVATION_FEATURES])
        # obs = {
        #     "observation": obs / self.OBS_SCALE,
        #     "achieved_goal": obs / self.OBS_SCALE,
        #     "desired_goal": goal / self.OBS_SCALE
        # }
        # return obs

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
        
        # return - np.power(np.dot(self.OBS_SCALE * np.abs(achieved_goal - desired_goal), self.REWARD_WEIGHTS), p)

        # DISTANCE TO GOAL
        distance_to_goal_reward = self.distance_2_goal_reward(achieved_goal, desired_goal, p)
        
        # OVER OTHER PARKING SPOTS REWARD        
        over_other_parking_spots_reward = self.OVER_OTHER_PARKING_SPOT_REWARD * np.squeeze(info["is_over_others_parking_spot"])

        # COLLISION REWARD
        collision_reward = self.COLLISION_REWARD * np.squeeze(info["is_collision"])

        # MOVING REWARD        
        # moving_reward = self.MOVING_REWARD * np.squeeze(info["velocity_idx"])

        # REACHING THE GOAL REWARD
        # reaching_goal_reward = self.REACHING_GOAL_REWARD *  np.squeeze(info["is_success"])

        reward = (distance_to_goal_reward + \
                  over_other_parking_spots_reward + \
                 # reverse_reward + \
                 # against_traffic_reward + \
                 # moving_reward +\
                #  reaching_goal_reward + \
                  collision_reward)

        reward /= self.REWARD_SCALE
        #print(reward)
        return reward 
                

    def _reward(self, action):
        raise NotImplementedError

    def _is_success(self, achieved_goal, desired_goal):
        # DISTANCE TO GOAL
        distance_to_goal_reward = self.distance_2_goal_reward(achieved_goal, desired_goal)
        #print(distance_to_goal_reward)
        self.vehicle.is_success = (distance_to_goal_reward > -self.SUCCESS_THRESHOLD)
        return self.vehicle.is_success

        # Let's try something new: Dicouple everything        
        # Let me start defining the thresholds in SI units ( m, m/s, degrees)
        # x_error_thr = 0.1
        # y_error_thr = 0.1
        # vx_error_thr = 0.1 #
        # vy_error_thr = 0.1 #0.27
        # heading_error_thr = np.deg2rad(5)
        # cos_h_error_thr = np.cos(heading_error_thr)
        # sin_h_error_thr = np.sin(heading_error_thr)

        # thresholds = [x_error_thr, y_error_thr, vx_error_thr, vy_error_thr, cos_h_error_thr, sin_h_error_thr]

        # errors = self.OBS_SCALE * np.abs(desired_goal - achieved_goal)
        
        # success = np.less_equal(errors,thresholds)

        # self.vehicle.is_success = np.all(success)
        # return self.vehicle.is_success

        

    def _is_terminal(self):
        """
            The episode is over if the ego vehicle crashed or the goal is reached.
        """
        # The episode cannot terminate unless all time steps are done. The reason for this is that HER + DDPG uses constant
        # length episodes. If you plan to use other algorithms, please uncomment this line
        #if info["is_collision"] or info["is_success"]:

        if self.vehicle.crashed: # or self.vehicle.is_success:
            self.reset()
        return False # self.vehicle.crashed or self._is_success(obs['achieved_goal'], obs['desired_goal'])
    
