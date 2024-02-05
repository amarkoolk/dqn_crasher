import gymnasium as gym
from gymnasium import RewardWrapper, Wrapper

from highway_env.road.road import Road, RoadNetwork
from highway_env import utils

import math
import matplotlib.pyplot as plt

class CrashResetWrapper(Wrapper):
    
    def __init__(self, env):
        super().__init__(env)

    def _reset(self) -> None:

        # Create Road Environment
        print(f'Using Custom Reset Wrapper!')
        lane_count = self.unwrapped.config['lanes_count']
        show_traj  = self.unwrapped.config['show_trajectories']

        self.road = Road(network=RoadNetwork.straight_road_network(lane_count, speed_limit=30),
                        np_random=self.get_wrapper_attr('np_random'), record_history=show_traj)
        
        # Create Road Vehicles

        other_vehicles_type = utils.class_from_path(self.unwrapped.config["other_vehicles_type"])

        spawn_configs = self.unwrapped.config['spawn_configs']
        spawn_config = self.get_wrapper_attr('np_random').choice(spawn_configs)
        spawn_distance = self.get_wrapper_attr('np_random').normal(self.unwrapped.config["mean_distance"], self.unwrapped.config["mean_distance"]/10)
        starting_vel = self.unwrapped.config["initial_speed"]+self.get_wrapper_attr('np_random').normal(self.unwrapped.config["mean_delta_v"], 5)
        self.controlled_vehicles = []
        for _ in range(self.unwrapped.config["ego_vehicles"]):
            # Behind Left
            if spawn_config == 'behind_left':
                lane1 = self.road.network.graph['0']['1'][0]
                vehicle = self.action_type.vehicle_class(road = self.road, \
                                                        position = lane1.position(0, 0), \
                                                        heading = lane1.heading_at(0), \
                                                        speed = starting_vel)
                
                self.controlled_vehicles.append(vehicle)
                self.road.vehicles.append(vehicle)

                lane2 = self.road.network.graph['0']['1'][1]
                vehicle = other_vehicles_type(road = self.road, \
                                            position = lane2.position(spawn_distance, 0), \
                                            heading = lane2.heading_at(spawn_distance), \
                                            speed = starting_vel)

                self.road.vehicles.append(vehicle)
            elif spawn_config == 'behind_right':
                lane1 = self.road.network.graph['0']['1'][1]
                vehicle = self.action_type.vehicle_class(road = self.road, \
                                                        position = lane1.position(0, 0), \
                                                        heading = lane1.heading_at(0), \
                                                        speed = starting_vel)
                
                self.controlled_vehicles.append(vehicle)
                self.road.vehicles.append(vehicle)

                lane2 = self.road.network.graph['0']['1'][0]
                vehicle = other_vehicles_type(road = self.road, \
                                            position = lane2.position(spawn_distance, 0), \
                                            heading = lane2.heading_at(spawn_distance), \
                                            speed = starting_vel)
                self.road.vehicles.append(vehicle)
            elif spawn_config == 'behind_center':
                lane1 = self.get_wrapper_attr('np_random').choice(self.road.network.graph['0']['1'])
                vehicle = self.action_type.vehicle_class(road = self.road, \
                                                        position = lane1.position(0, 0), \
                                                        heading = lane1.heading_at(0), \
                                                        speed = starting_vel)
                
                self.controlled_vehicles.append(vehicle)
                self.road.vehicles.append(vehicle)

                lane2 = lane1
                vehicle = other_vehicles_type(road = self.road, \
                                            position = lane2.position(spawn_distance, 0), \
                                            heading = lane2.heading_at(spawn_distance), \
                                            speed = starting_vel)
                self.road.vehicles.append(vehicle)
            elif spawn_config == 'adjacent_left':
                lane1 = self.road.network.graph['0']['1'][0]
                vehicle = self.action_type.vehicle_class(road = self.road, \
                                                        position = lane1.position(0, 0), \
                                                        heading = lane1.heading_at(0), \
                                                        speed = starting_vel)
                
                self.controlled_vehicles.append(vehicle)
                self.road.vehicles.append(vehicle)

                lane2 = self.road.network.graph['0']['1'][1]
                vehicle = other_vehicles_type(road = self.road, \
                                            position = lane2.position(0, 0), \
                                            heading = lane2.heading_at(0), \
                                            speed = starting_vel)
                self.road.vehicles.append(vehicle)
            elif spawn_config == 'adjacent_right':
                lane1 = self.road.network.graph['0']['1'][1]
                vehicle = self.action_type.vehicle_class(road = self.road, \
                                                        position = lane1.position(0, 0), \
                                                        heading = lane1.heading_at(0), \
                                                        speed = starting_vel)
                
                self.controlled_vehicles.append(vehicle)
                self.road.vehicles.append(vehicle)

                lane2 = self.road.network.graph['0']['1'][0]
                vehicle = other_vehicles_type(road = self.road, \
                                            position = lane2.position(0, 0), \
                                            heading = lane2.heading_at(0), \
                                            speed = starting_vel)
                self.road.vehicles.append(vehicle)
            elif spawn_config == 'forward_left':
                lane1 = self.road.network.graph['0']['1'][0]
                vehicle = self.action_type.vehicle_class(road = self.road, \
                                                        position = lane1.position(spawn_distance, 0), \
                                                        heading = lane1.heading_at(spawn_distance), \
                                                        speed = starting_vel)
                
                self.controlled_vehicles.append(vehicle)
                self.road.vehicles.append(vehicle)

                lane2 = self.road.network.graph['0']['1'][1]
                vehicle = other_vehicles_type(road = self.road, \
                                            position = lane2.position(0, 0), \
                                            heading = lane2.heading_at(0), \
                                            speed = starting_vel)
                self.road.vehicles.append(vehicle)
            elif spawn_config == 'forward_right':
                lane1 = self.road.network.graph['0']['1'][1]
                vehicle = self.action_type.vehicle_class(road = self.road, \
                                                        position = lane1.position(spawn_distance, 0), \
                                                        heading = lane1.heading_at(spawn_distance), \
                                                        speed = starting_vel)
                
                self.controlled_vehicles.append(vehicle)
                self.road.vehicles.append(vehicle)

                lane2 = self.road.network.graph['0']['1'][0]
                vehicle = other_vehicles_type(road = self.road, \
                                            position = lane2.position(0, 0), \
                                            heading = lane2.heading_at(0), \
                                            speed = starting_vel)
                self.road.vehicles.append(vehicle)
            elif spawn_config == 'forward_center':
                lane1 = self.get_wrapper_attr('np_random').choice(self.road.network.graph['0']['1'])
                vehicle = self.action_type.vehicle_class(road = self.road, \
                                                        position = lane1.position(spawn_distance, 0), \
                                                        heading = lane1.heading_at(spawn_distance), \
                                                        speed = starting_vel)
                
                self.controlled_vehicles.append(vehicle)
                self.road.vehicles.append(vehicle)

                lane2 = lane1
                vehicle = other_vehicles_type(road = self.road, \
                                            position = lane2.position(0, 0), \
                                            heading = lane2.heading_at(0), \
                                            speed = starting_vel)
                self.road.vehicles.append(vehicle)
            

class CrashRewardWrapper(RewardWrapper):

    def __init__(self, env):
        super().__init__(env)


    def reward(self, reward):

        road = self.get_wrapper_attr('road')
        ego_vehicle = road.vehicles[0]
        other_vehicle = road.vehicles[1]

        dx = other_vehicle.position[0] - ego_vehicle.position[0]
        dy = other_vehicle.position[1] - ego_vehicle.position[1]
        vx0 = (math.cos(ego_vehicle.heading))*ego_vehicle.speed
        vx1 = (math.cos(other_vehicle.heading))*other_vehicle.speed
        vy0 = (math.sin(ego_vehicle.heading))*ego_vehicle.speed
        vy1 = (math.sin(other_vehicle.heading))*other_vehicle.speed

        dvx = vx1 - vx0
        dvy = vy1 - vy0

        ttc_x = dx/dvx if abs(dvx) > 1e-6 else dx/1e-6
        ttc_y = dy/dvy if abs(dvy) > 1e-6 else dy/1e-6

        # Calculate Rewards
        if abs(dvx) < self.unwrapped.config["tolerance"]:
            if abs(dx) < self.unwrapped.config["tolerance"]:
                r_x = self.unwrapped.config['ttc_x_reward']
            else:
                r_x = 0
        else:
            try:
                r_x = 1.0/(1.0 + math.exp(-4-0.1*ttc_x)) if ttc_x <= 0 else -1.0/(1.0 + math.exp(4-0.1*ttc_x))
            except OverflowError:
                r_x = 0.0
        
        if abs(dvy) < self.unwrapped.config["tolerance"]:
            if abs(dy) < self.unwrapped.config["tolerance"]:
                r_y = self.unwrapped.config['ttc_y_reward']
            else:
                r_y = 0
        else:
            try:
                r_y = 1.0/(1.0 + math.exp(-4-0.1*ttc_y)) if ttc_y <= 0 else -1.0/(1.0 + math.exp(4-0.1*ttc_y))
            except OverflowError:
                r_y = 0.0

        crashed = self.get_wrapper_attr('vehicle').crashed

        return r_x + r_y + float(crashed) * self.unwrapped.config['crash_reward']