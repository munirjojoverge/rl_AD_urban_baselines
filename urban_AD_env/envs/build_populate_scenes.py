
######################################################################
#          Deep Reinforcement Learning for Autonomous Driving
#                  Created/Modified on: February 7, 2019
#                      Author: Munir Jojo-Verge
#######################################################################
from __future__ import division, print_function, absolute_import

import numpy as np
import random as rd
from datetime import timedelta

from urban_AD_env.road.lane import LineType, StraightLane, CircularLane, SineLane
from urban_AD_env.road.road import Road, RoadNetwork
from urban_AD_env.vehicle.dynamics import Vehicle, Obstacle
from urban_AD_env.vehicle.control import ControlledVehicle
from urban_AD_env import utils
from urban_AD_env.envs.utils import goal_distance, rad

ROUNDABOUT = [  ["se", "ex"],
                ["ex", "ee"],
                ["ee", "nx"],               
                ["nx", "ne"],
                ["ne", "wx"],
                ["wx", "we"],
                ["we", "sx"],
                ["sx", "se"] ]

def _build_roundabout(scene):
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

    road = Road(network=net, np_random=scene.np_random)
    scene.road = road

def _populate_roundabout(scene):
    """
        Populate a road with several vehicles on the urban_AD and on the merging lane, as well as an ego-vehicle.
    :return: the ego-vehicle
    """        
    
    ################### ADDING EGO VEHICLE ###################
    # Ego-lane
    ego_lane = scene.road.network.get_lane(("ser", "ses", 0))
    
    # Ego Initial State
    ego_ini_pos  = ego_lane.position(140, 0)
    ego_ini_heading = ego_lane.heading_at(140)
    ego_ini_vel = 5 # m/s                
    #scene.vehicle = Vehicle(scene.road, [200, scene.np_random.randint(0, 12)], 2*np.pi*scene.np_random.rand(), 0)

    # Ego Vehicle
    ego_vehicle = Vehicle(scene.road,
                                ego_ini_pos,
                                velocity=ego_ini_vel,
                                heading=ego_ini_heading)

    scene.road.vehicles.append(ego_vehicle)
    scene.vehicle = ego_vehicle

    ################### ADDING ALL OTHER VEHICLES ###################
    position_deviation = 2
    velocity_deviation = 2

    # Incoming vehicles                
    destinations = ["exr", "sxr", "nxr"]
    other_vehicles_type = utils.class_from_path(scene.config["other_vehicles_type"])

    for i in list(range(1, scene.config["num_vehicles_incoming"])) + list(range(-1, 0)):
        vehicle = other_vehicles_type.make_on_lane(scene.road,
                                                ("we", "sx", 1),
                                                longitudinal=5 + scene.np_random.randn()*position_deviation,
                                                velocity=16 + scene.np_random.randn()*velocity_deviation)

        if scene.config["incoming_vehicle_destination"] is not None:
            destination = destinations[scene.config["incoming_vehicle_destination"]]
        else:
            destination = scene.np_random.choice(destinations)
        vehicle.plan_route_to(destination)
        vehicle.randomize_behavior()
        scene.road.vehicles.append(vehicle)

    # Vehicles inside the round-about
    for i in list(range(1, scene.config["num_vehicles_inside_roundabout"])) + list(range(-1, 0)):
        vehicle = other_vehicles_type.make_on_lane(scene.road,
                                                    ("we", "sx", 0),
                                                    longitudinal=20*i + scene.np_random.randn()*position_deviation,
                                                    velocity=16 + scene.np_random.randn()*velocity_deviation)
        vehicle.plan_route_to(scene.np_random.choice(destinations))
        vehicle.randomize_behavior()
        scene.road.vehicles.append(vehicle)


    # Entering vehicle
    for i in list(range(1, scene.config["num_vehicles_entering"])) + list(range(-1, 0)):
        vehicle = other_vehicles_type.make_on_lane(scene.road,
                                                ("eer", "ees", 0),
                                                longitudinal=50 + scene.np_random.randn() * position_deviation,
                                                velocity=16 + scene.np_random.randn() * velocity_deviation)
        vehicle.plan_route_to(scene.np_random.choice(destinations))
        vehicle.randomize_behavior()
        scene.road.vehicles.append(vehicle)


    # The goal will be for now an obstacle that we want to reach. Once we reach it = collision and it will terminate the episode.    
    roundabout_section = int(scene.np_random.rand() * 7) + 1
    lane_num = int(scene.np_random.rand() * 1) + 1
    longitudinal = 5 + scene.np_random.randn()*position_deviation
    scene.goal = Obstacle.make_on_lane(scene.road,
                                      (ROUNDABOUT[roundabout_section][0], ROUNDABOUT[roundabout_section][1], lane_num),
                                       longitudinal=longitudinal,
                                       velocity=0)
     
    # lane_coords = scene.goal.lane.local_coordinates(scene.goal.position)
    # lane_next_coords = lane_coords[0]
    scene.goal.heading = scene.goal.lane.heading_at(longitudinal)
    
    #scene.goal = Obstacle(scene.road, np.array([ego_ini_pos[0], -ego_ini_pos[1]]))        
    # Let's calculate a reasonable manouver duration depending on where the Ego car is located and where the goal is located
    # For these quick estimate we will use the average EGO speed based on:
    # Vehicle.SPEED_MIN & Vehicle.SPEED_MAX
    # and the distance between the EGO and the goal
    # ego_ini_pos = scene.vehicle.position
    #goal_pos = scene.goal.position
    # d = goal_distance(ego_ini_pos, goal_pos)
    # ego_avg_speed = (scene.vehicle.MAX_VELOCITY - 0.0)/2
    # manouver_duration = d/ego_avg_speed
    # scene.goal.manouver_duration = manouver_duration    
    scene.goal.COLLISIONS_ENABLED = False
    scene.road.vehicles.insert(0, scene.goal)
    
    #generate_random_goal(scene)


def _build_merge(scene):
    """
        Make a road composed of a straight urban_AD and a merging lane.
    :return: the road
    """
    net = RoadNetwork()

    # urban_AD lanes
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
    lkb = SineLane(ljk.position(ends[0], -amplitude), ljk.position(sum(ends[:2]), -amplitude),
                    amplitude, 2 * np.pi / (2*ends[1]), np.pi / 2, line_types=[c, c], forbidden=True)
    lbc = StraightLane(lkb.position(ends[1], 0), lkb.position(ends[1], 0) + [ends[2], 0],
                        line_types=[n, c], forbidden=True)
    net.add_lane("j", "k", ljk)
    net.add_lane("k", "b", lkb)
    net.add_lane("b", "c", lbc)
    road = Road(network=net, np_random=scene.np_random)
    road.vehicles.append(Obstacle(road, lbc.position(ends[2], 0)))
    scene.road = road

def _populate_merge(scene):
    """
        Populate a road with several vehicles on the urban_AD and on the merging lane, as well as an ego-vehicle.
    :return: the ego-vehicle
    """
    road = scene.road
    ego_vehicle = Vehicle(road, road.network.get_lane(("a", "b", 1)).position(30, 0), velocity=30)
    road.vehicles.append(ego_vehicle)

    # other_vehicles_type = utils.class_from_path(scene.config["other_vehicles_type"])    
    # road.vehicles.append(other_vehicles_type(road, road.network.get_lane(("a", "b", 0)).position(90, 0), velocity=29))
    # road.vehicles.append(other_vehicles_type(road, road.network.get_lane(("a", "b", 1)).position(70, 0), velocity=31))
    # road.vehicles.append(other_vehicles_type(road, road.network.get_lane(("a", "b", 0)).position(5, 0), velocity=31.5))

    # merging_v = other_vehicles_type(road, road.network.get_lane(("j", "k", 0)).position(110, 0), velocity=20)
    # merging_v.target_velocity = 30
    # road.vehicles.append(merging_v)
    scene.vehicle = ego_vehicle

    generate_random_goal(scene)


def _build_multilane(scene):
    """
        Create a road composed of straight adjacent lanes.
    """
    scene.road = Road(network=RoadNetwork.straight_road_network(scene.config["lanes_count"]),
                        np_random=scene.np_random)

def _populate_multilane(scene):
    """
        Create some new random vehicles of a given type, and add them on the road.
    """
    scene.vehicle = Vehicle.create_random(scene.road, 25, spacing=scene.config["initial_spacing"])
    scene.road.vehicles.append(scene.vehicle)

    vehicles_type = utils.class_from_path(scene.config["other_vehicles_type"])
    for _ in range(scene.config["vehicles_count"]):
        scene.road.vehicles.append(vehicles_type.create_random(scene.road))
        
    generate_random_goal(scene)

def generate_random_goal(scene, manouver_duration_goal=None):
    scene.goal = Obstacle.create_random(scene.road)
    scene.goal.COLLISIONS_ENABLED = False
    
    if manouver_duration_goal == None:
        # Let's calculate a reasonable manouver duration depending on where the Ego car is located and where the goal is located
        # For these quick estimate we will use the average EGO speed based on:
        # Vehicle.SPEED_MIN & Vehicle.SPEED_MAX
        # and the distance between the EGO and the goal
        ego_ini_pos = scene.vehicle.position
        goal_pos = scene.goal.position
        d = goal_distance(ego_ini_pos, goal_pos)
        ego_avg_speed = (scene.vehicle.MAX_VELOCITY - 0.0)/2
        manouver_duration_goal = d/ego_avg_speed
       

    scene.goal.manouver_duration = manouver_duration_goal    
    scene.road.vehicles.insert(0, scene.goal)




