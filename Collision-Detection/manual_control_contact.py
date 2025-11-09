#!/usr/bin/env python3
"""
Simple manual control with contact tracking collision detection.
Clean implementation - finds closest EGO vertex to NPC edge and classifies.
"""

import os
import sys
from enum import Enum

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import gymnasium as gym
import highway_env
import numpy as np
import pygame
from utils.config import load_config

from contact_tracking import find_primary_contact, classify_collision, format_contact


class Action(Enum):
    LANE_LEFT = 0
    IDLE = 1
    LANE_RIGHT = 2
    FASTER = 3
    SLOWER = 4


class ContactController:
    """Manual controller with contact tracking collision detection."""
    
    def __init__(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, "configs/env/single_agent.yaml")
        self.config = load_config(config_path)
        
        self.env = gym.make("highway-v0", config=self.config, render_mode="human")
        self.running = True
        
        # Track previous position to determine movement direction
        self.prev_ego_position = None
        self.last_action = None
        self.recent_actions = []  # Track last few actions to determine movement intent
        
        # CRITICAL: Store the controlled vehicle ID at initialization
        # This ensures we always know which vehicle WE are controlling
        self.controlled_vehicle_id = None
        
        print("\n" + "="*60)
        print("Contact Tracking Collision Detection")
        print("="*60)
        print("Controls: Arrow keys to drive, R to reset, Q to quit\n")
    
    def get_keyboard_action(self):
        action = Action.IDLE
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    self.running = False
                elif event.key == pygame.K_r:
                    return "RESET"
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            action = Action.LANE_LEFT
        elif keys[pygame.K_RIGHT]:
            action = Action.LANE_RIGHT
        elif keys[pygame.K_UP]:
            action = Action.FASTER
        elif keys[pygame.K_DOWN] or keys[pygame.K_SPACE]:
            action = Action.SLOWER
        
        return action
    
    def check_collision(self, info):
        """Check collision using contact tracking."""
        if not info.get("crashed", False):
            return
        
        env = self.env.unwrapped
        road = env.road
        
        if road is None:
            return
        
        # CRITICAL: Identify the controlled vehicle
        # Use stored ID if available, otherwise use env.vehicle and store it
        ego = None
        if self.controlled_vehicle_id is not None:
            # Find vehicle by stored ID
            for vehicle in road.vehicles:
                if id(vehicle) == self.controlled_vehicle_id:
                    ego = vehicle
                    break
        
        # Fallback: use env.vehicle and store its ID
        if ego is None:
            ego = env.vehicle
            if ego is not None:
                self.controlled_vehicle_id = id(ego)
        
        if ego is None:
            return
        
        # Determine movement direction from ACTUAL ACTIONS taken (more reliable than position)
        movement_direction = None
        if self.recent_actions:
            # Check last few actions - if LEFT was pressed, user moved LEFT
            # Filter out IDLE actions
            non_idle_actions = [a for a in self.recent_actions if a != Action.IDLE]
            if non_idle_actions:
                last_non_idle = non_idle_actions[-1]
                if last_non_idle == Action.LANE_LEFT:
                    movement_direction = "LEFT"
                elif last_non_idle == Action.LANE_RIGHT:
                    movement_direction = "RIGHT"
        
        # Also check position change as backup
        position_based_direction = None
        if self.prev_ego_position is not None:
            position_change = ego.position - self.prev_ego_position
            if abs(position_change[1]) > 0.01:  # Significant Y movement
                position_based_direction = "LEFT" if position_change[1] > 0 else "RIGHT"
        
        self.prev_ego_position = ego.position.copy()
        
        # Find colliding NPC
        for npc in road.vehicles:
            if npc is ego or id(npc) == self.controlled_vehicle_id:
                continue
            
            # Find primary contact using SAT to determine collision side
            dt = 1.0 / env.config["simulation_frequency"]
            contact = find_primary_contact(
                ego.polygon(), 
                npc.polygon(),
                ego_velocity=ego.velocity * dt,
                npc_velocity=npc.velocity * dt,
                threshold=10.0  # Increased threshold to catch all contacts when vehicles are overlapping
            )
            
            if contact:
                print("\n" + "="*60)
                print("COLLISION DETECTED")
                print("="*60)
                
                # DEBUG: Verify which vehicle is which
                print(f"\n[DEBUG] Vehicle Identity:")
                print(f"  Controlled vehicle (stored ID: {self.controlled_vehicle_id}):")
                print(f"    Position: {ego.position}")
                print(f"    Current ID: {id(ego)}")
                print(f"    Matches stored ID: {id(ego) == self.controlled_vehicle_id}")
                print(f"    Receives actions: YES (this is EGO)")
                print(f"  Other vehicle (NPC):")
                print(f"    Position: {npc.position}")
                print(f"    ID: {id(npc)}")
                print(f"    Receives actions: NO (this is NPC)")
                print(f"  env.vehicle ID: {id(env.vehicle) if env.vehicle else 'None'}")
                print(f"  env.vehicle matches EGO: {env.vehicle is ego if env.vehicle else False}")
                
                # List all vehicles
                print(f"\n[DEBUG] All vehicles in road.vehicles:")
                for i, vehicle in enumerate(road.vehicles):
                    is_ego = vehicle is ego or id(vehicle) == self.controlled_vehicle_id
                    is_env_vehicle = vehicle is env.vehicle
                    print(f"    Vehicle {i}: ID={id(vehicle)}, Position={vehicle.position}, "
                          f"EGO={is_ego}, env.vehicle={is_env_vehicle}")
                
                if movement_direction:
                    print(f"\n[DEBUG] Movement (from actions): EGO moved {movement_direction} (to {'higher' if movement_direction == 'LEFT' else 'lower'} Y)")
                if position_based_direction:
                    print(f"[DEBUG] Movement (from position): EGO moved {position_based_direction} (to {'higher' if position_based_direction == 'LEFT' else 'lower'} Y)")
                if movement_direction and position_based_direction and movement_direction != position_based_direction:
                    print(f"⚠️  WARNING: Action-based movement ({movement_direction}) differs from position-based ({position_based_direction})")
                
                print(f"\nEGO Y={ego.position[1]:.2f}, NPC Y={npc.position[1]:.2f}")
                print(f"NPC is to EGO's {'LEFT' if npc.position[1] > ego.position[1] else 'RIGHT'} (higher Y = left)")
                print(f"\nContact: {format_contact(contact)}")
                print(f"  Contact type: {contact.contact_type}")
                if contact.contact_type == "edge-edge":
                    print(f"  EGO edge ID: {contact.ego_edge_id}")
                else:
                    print(f"  EGO vertex ID: {contact.ego_vertex_id}")
                print(f"  NPC edge ID: {contact.npc_edge_id}")
                
                print(f"\nClassification: {classify_collision(contact)}")
                print("="*60 + "\n")
                
                input("Press Enter to continue...")
                self.running = False
                return
    
    def run(self):
        self.env.reset()
        self.prev_ego_position = None  # Reset position tracking
        self.recent_actions = []  # Reset action tracking
        
        # Store controlled vehicle ID after reset
        env = self.env.unwrapped
        if env.vehicle is not None:
            self.controlled_vehicle_id = id(env.vehicle)
            print(f"[DEBUG] Initial controlled vehicle ID stored: {self.controlled_vehicle_id}")
        
        while self.running:
            action = self.get_keyboard_action()
            
            if action == "RESET":
                self.env.reset()
                self.prev_ego_position = None  # Reset position tracking
                self.recent_actions = []  # Reset action tracking
                # Re-store controlled vehicle ID after reset
                env = self.env.unwrapped
                if env.vehicle is not None:
                    self.controlled_vehicle_id = id(env.vehicle)
                    print(f"[DEBUG] After reset - controlled vehicle ID stored: {self.controlled_vehicle_id}")
                continue
            
            # Track action for movement detection
            if action != "RESET":
                self.recent_actions.append(action)
                # Keep only last 10 actions
                if len(self.recent_actions) > 10:
                    self.recent_actions.pop(0)
            
            _, _, _, _, info = self.env.step(action.value if action != "RESET" else Action.IDLE.value)
            self.env.render()
            self.check_collision(info)
        
        self.env.close()


if __name__ == "__main__":
    try:
        controller = ContactController()
        controller.run()
    except KeyboardInterrupt:
        print("\nQuitting...")

