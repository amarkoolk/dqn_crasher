#!/usr/bin/env python3
"""
Simple contact tracking - shows which part of EGO hit which part of NPC.

Output format:
  - EGO's [corner] hit NPC's [edge]
  - EGO's [edge] hit NPC's [corner]
  - EGO's [corner] hit NPC's [corner]
  - EGO's [edge] hit NPC's [edge]
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

from contact_tracking import ContactTrackingDetector


class Action(Enum):
    LANE_LEFT = 0
    IDLE = 1
    LANE_RIGHT = 2
    FASTER = 3
    SLOWER = 4


class SimpleContactController:
    """
    Simple controller that shows which part of EGO hit which part of NPC.
    """
    
    CORNER_NAMES = {
        0: "rear-left corner",
        1: "front-left corner",
        2: "front-right corner",
        3: "rear-right corner"
    }
    
    EDGE_NAMES = {
        0: "rear edge",
        1: "left edge",
        2: "front edge",
        3: "right edge"
    }
    
    def __init__(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, "configs/env/single_agent.yaml")
        self.config = load_config(config_path)
        
        self.env = gym.make("highway-v0", config=self.config, render_mode="human")
        self.detector = ContactTrackingDetector(early_exit=False, penetration_threshold=0.05)
        self.running = True
        
        print("\n" + "="*60)
        print("Simple Contact Tracking - EGO vs NPC Collision")
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
        """Check collision and print which parts made contact."""
        if not info.get("crashed", False):
            return
        
        env = self.env.unwrapped
        ego = env.vehicle
        road = env.road
        if ego is None or road is None:
            return
        
        dt = 1.0 / env.config["simulation_frequency"]
        
        # Find colliding NPC
        for npc in road.vehicles:
            if npc is ego:
                continue
            
            # Test EGO vertices against NPC edges
            ego_hits_npc = self.detector.detect(
                ego.polygon(),
                npc.polygon(),
                ego.velocity * dt,
                npc.velocity * dt
            )
            
            # Test NPC vertices against EGO edges
            npc_hits_ego = self.detector.detect(
                npc.polygon(),
                ego.polygon(),
                npc.velocity * dt,
                ego.velocity * dt
            )
            
            if ego_hits_npc.intersecting or npc_hits_ego.intersecting:
                print("\n" + "="*60)
                print("COLLISION DETECTED")
                print("="*60)
                
                # EGO's corners hitting NPC's edges
                if ego_hits_npc.contact_points:
                    for contact in ego_hits_npc.contact_points:
                        if contact.penetration > 0.001:  # Real penetration
                            ego_corner = self.CORNER_NAMES[contact.vertex_id]
                            npc_edge = self.EDGE_NAMES[contact.edge_id]
                            print(f"  • EGO's {ego_corner} hit NPC's {npc_edge}")
                
                # NPC's corners hitting EGO's edges
                if npc_hits_ego.contact_points:
                    for contact in npc_hits_ego.contact_points:
                        if contact.penetration > 0.001:  # Real penetration
                            npc_corner = self.CORNER_NAMES[contact.vertex_id]
                            ego_edge = self.EDGE_NAMES[contact.edge_id]
                            print(f"  • NPC's {npc_corner} hit EGO's {ego_edge}")
                
                print("="*60 + "\n")
                
                input("Press Enter to continue...")
                self.running = False
                return
    
    def run(self):
        self.env.reset()
        while self.running:
            action = self.get_keyboard_action()
            
            if action == "RESET":
                self.env.reset()
                continue
            
            _, _, _, _, info = self.env.step(action.value)
            self.env.render()
            self.check_collision(info)
        
        self.env.close()


if __name__ == "__main__":
    try:
        controller = SimpleContactController()
        controller.run()
    except KeyboardInterrupt:
        print("\nQuitting...")


