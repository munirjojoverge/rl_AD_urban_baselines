from __future__ import division, print_function
import numpy as np
import pandas as pd

from urban_AD_env import utils
from urban_AD_env.logger import Loggable


class Vehicle(Loggable):
    """
        A moving vehicle on a road, and its dynamics.

        The vehicle is represented by a dynamical system: a modified bicycle model.
        It's state is propagated depending on its steering and acceleration actions.
    """
    COLLISIONS_ENABLED = True
    """ Enable collision detection between vehicles """

    LENGTH = 5.0
    """ Vehicle length [m] """
    WIDTH = 2.0
    """ Vehicle width [m] """
    DEFAULT_VELOCITIES = [23, 24]
    """ Range for random initial velocities [m/s] """

    def __init__(self, road, state):
        self.road = road
        self.curr_position = np.array([state[0],state[1]]).astype('float')
        self.prev_position = self.curr_position
        self.heading = state[2]
        self.velocity = state[3]
        self.lane_index = self.road.network.get_closest_lane_index(self.curr_position) if self.road else np.nan
        self.lane = self.road.network.get_lane(self.lane_index) if self.road else None
        self.action = {'steering': 0, 'acceleration': 0}
        self.crashed = False
        self.log = []

    @classmethod
    def make_on_lane(cls, road, lane_index, longitudinal, velocity=0):
        """
            Create a vehicle on a given lane at a longitudinal position.

        :param road: the road where the vehicle is driving
        :param lane_index: index of the lane where the vehicle is located
        :param longitudinal: longitudinal position along the lane
        :param velocity: initial velocity in [m/s]
        :return: A vehicle with at the specified position
        """
        lane = road.network.get_lane(lane_index)
        position = lane.position(longitudinal, 0)
        heading = lane.heading_at(longitudinal)
        state = np.array([position[0], position[1], heading, velocity]).astype('float')
        return cls(road, state)

    @classmethod
    def create_random(cls, road, velocity=None, spacing=1):
        """
            Create a random vehicle on the road.

            The lane and /or velocity are chosen randomly, while longitudinal position is chosen behind the last
            vehicle in the road with density based on the number of lanes.

        :param road: the road where the vehicle is driving
        :param velocity: initial velocity in [m/s]. If None, will be chosen randomly
        :param spacing: ratio of spacing to the front vehicle, 1 being the default
        :return: A vehicle with random position and/or velocity
        """
        velocity = velocity or road.np_random.randint(Vehicle.DEFAULT_VELOCITIES[0], Vehicle.DEFAULT_VELOCITIES[1])
        default_spacing = 1.5*velocity
        _from = road.np_random.choice(list(road.network.graph.keys()))
        _to = road.np_random.choice(list(road.network.graph[_from].keys()))
        _id = road.np_random.choice(len(road.network.graph[_from][_to]))
        offset = spacing * default_spacing * np.exp(-5 / 30 * len(road.network.graph[_from][_to]))
        x0 = np.max([v.curr_position[0] for v in road.vehicles]) if len(road.vehicles) else 3*offset
        x0 += offset

        position = road.network.get_lane((_from, _to, _id)).curr_position(x0, 0)
        heading = 0
        state = np.array([position[0], position[1], heading, velocity]).astype('float')

        v = cls(road, state)
        return v

    @classmethod
    def create_from(cls, vehicle):
        """
            Create a new vehicle from an existing one.
            Only the vehicle dynamics are copied, other properties are default.

        :param vehicle: a vehicle
        :return: a new vehicle at the same dynamical state
        """

        state = np.array([vehicle.curr_position[0], vehicle.curr_position[1], vehicle.heading, vehicle.velocity]).astype('float')
        v = cls(vehicle.road, state)
        return v

    def act(self, action=None):
        """
            Store an action to be repeated.

        :param action: the input action
        """
        if action:
            self.action = action

    def step(self, dt):
        """
            Propagate the vehicle state given its actions.

            Integrate a modified bicycle model with a 1st-order response on the steering wheel dynamics.
            If the vehicle is crashed, the actions are overridden with erratic steering and braking until complete stop.
            The vehicle's current lane is updated.

        :param dt: timestep of integration of the model [s]
        """
        # REVERSE NOT ALLOWED (for any vehicle)
        if self.velocity <= 0:
            self.velocity = 0

        if self.crashed:
            self.action['steering'] = 0
            self.action['acceleration'] = -self.velocity

        v = self.velocity * np.array([np.cos(self.heading), np.sin(self.heading)])
        self.prev_position = self.curr_position.copy()
        self.curr_position += v * dt
        self.heading += self.velocity * np.tan(self.action['steering']) / self.LENGTH * dt
        self.velocity += self.action['acceleration'] * dt

        if self.road:
            self.lane_index = self.road.network.get_closest_lane_index(self.curr_position)
            self.lane = self.road.network.get_lane(self.lane_index)

    def lane_distance_to(self, vehicle):
        """
            Compute the signed distance to another vehicle along current lane.

        :param vehicle: the other vehicle
        :return: the distance to the other vehicle [m]
        """
        if not vehicle:
            return np.nan
        return self.lane.local_coordinates(vehicle.curr_position)[0] - self.lane.local_coordinates(self.curr_position)[0]

    def check_collision(self, other):
        """
            Check for collision with another vehicle.

        :param other: the other vehicle
        """
        if not self.COLLISIONS_ENABLED or self.crashed or other is self:
            return

        # Fast spherical pre-check
        if np.linalg.norm(other.curr_position - self.curr_position) > self.LENGTH:
            return

        # Accurate rectangular check
        if utils.rotated_rectangles_intersect((self.curr_position, 0.9*self.LENGTH, 0.9*self.WIDTH, self.heading),
                                              (other.curr_position, 0.9*other.LENGTH, 0.9*other.WIDTH, other.heading)):
            self.velocity = other.velocity = min(self.velocity, other.velocity)
            self.crashed = other.crashed = True

    def to_dict(self, origin_vehicle=None):
        d = {
            'presence': 1,
            'x': self.curr_position[0],
            'y': self.curr_position[1],
            'vx': self.velocity * np.cos(self.heading),
            'vy': self.velocity * np.sin(self.heading)
        }
        if origin_vehicle:
            origin_dict = origin_vehicle.to_dict()
            for key in ['x', 'y', 'vx', 'vy']:
                d[key] -= origin_dict[key]
        return d

    def dump(self):
        """
            Update the internal log of the vehicle, containing:
                - its kinematics;
                - some metrics relative to its neighbour vehicles.
        """
        data = {
            'x': self.curr_position[0],
            'y': self.curr_position[1],
            'psi': self.heading,
            'vx': self.velocity * np.cos(self.heading),
            'vy': self.velocity * np.sin(self.heading),
            'v': self.velocity,
            'acceleration': self.action['acceleration'],
            'steering': self.action['steering']}

        if self.road:
            for lane_index in self.road.network.side_lanes(self.lane_index):
                lane_coords = self.road.network.get_lane(lane_index).local_coordinates(self.curr_position)
                data.update({
                    'dy_lane_{}'.format(lane_index): lane_coords[1],
                    'psi_lane_{}'.format(lane_index): self.road.network.get_lane(lane_index).heading_at(lane_coords[0])
                })
            front_vehicle, rear_vehicle = self.road.neighbour_vehicles(self)
            if front_vehicle:
                data.update({
                    'front_v': front_vehicle.velocity,
                    'front_distance': self.lane_distance_to(front_vehicle)
                })
            if rear_vehicle:
                data.update({
                    'rear_v': rear_vehicle.velocity,
                    'rear_distance': rear_vehicle.lane_distance_to(self)
                })

        self.log.append(data)

    def get_log(self):
        """
            Cast the internal log as a DataFrame.

        :return: the DataFrame of the Vehicle's log.
        """
        return pd.DataFrame(self.log)

    def __str__(self):
        return "{} #{}: {}".format(self.__class__.__name__, id(self) % 1000, self.curr_position)

    def __repr__(self):
        return self.__str__()


class Obstacle(Vehicle):
    """
        A motionless obstacle at a given position.
    """

    def __init__(self, road, position):
        super(Obstacle, self).__init__(road, position, velocity=0)
        self.target_velocity = 0
        self.LENGTH = self.WIDTH