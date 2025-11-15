#!/usr/bin/env python3
"""
Manual control harness that exercises the new MTV-based collision classifier.

As soon as highway-env reports a crash we:
- Run the SAT solver to obtain the MTV
- Project both vehicles onto the MTV axis
- Extract extremal vertices/edges involved in the contact
- Classify the collision using the dead-simple geometry rules
"""

import os
import sys
from enum import Enum

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gymnasium as gym
import highway_env
import numpy as np
import pygame
from utils.config import load_config

from new_implementation import (
    detect_and_classify_collision,
    print_classification,
)

# Monkey-patch to add speed labels to vehicles
original_display = None

def display_vehicle_with_speed(vehicle, surface, transparent=False, offscreen=False, label=False, draw_roof=False):
    """Custom vehicle display that shows speed labels."""
    from highway_env.vehicle.graphics import VehicleGraphics

    # Call original display method
    global original_display
    if original_display is None:
        original_display = VehicleGraphics.display

    original_display(vehicle, surface, transparent, offscreen, label, draw_roof)

    # Add speed label next to vehicle
    if not offscreen and surface.is_visible(vehicle.position):
        speed = np.linalg.norm(vehicle.velocity)
        font = pygame.font.Font(None, 20)
        text = f"{speed:.1f}"
        text_surface = font.render(text, True, (10, 10, 10), (255, 255, 255))
        position = surface.pos2pix(vehicle.position[0], vehicle.position[1])
        # Position label slightly above and to the right of vehicle
        surface.blit(text_surface, (position[0] + 20, position[1] - 25))


class Action(Enum):
    LANE_LEFT = 0
    IDLE = 1
    LANE_RIGHT = 2
    FASTER = 3
    SLOWER = 4


class SATContactController:
    """Manual controller that surfaces MTV-based SAT collision classification."""

    def __init__(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.join(script_dir, '..')
        config_path = os.path.join(parent_dir, "configs/env/single_agent.yaml")
        self.config = load_config(config_path)

        # Enable speed labels by monkey-patching vehicle display
        from highway_env.vehicle.graphics import VehicleGraphics
        global original_display
        if original_display is None:
            original_display = VehicleGraphics.display
        VehicleGraphics.display = display_vehicle_with_speed

        self.env = gym.make("highway-v0", config=self.config, render_mode="human")
        self.running = True

        # Store the controlled vehicle ID at initialization
        self.controlled_vehicle_id = None

        print("\n" + "="*60)
        print("MTV-Based SAT Collision Detection")
        print("="*60)
        print("Dead-simple MTV classification using:")
        print("  - SAT minimum translation vector (MTV)")
        print("  - Projections onto the MTV axis only")
        print("  - Extremal vertex/edge lookup for contact typing")
        print("  - No alignment, heading, or lateral heuristics")
        print("\nControls: Arrow keys to drive, R to reset, Q to quit\n")
    
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

    def print_speeds(self, info):
        """Print speed information to console (matching src implementation)."""
        if "ego_speed" in info and "npc_speed" in info:
            print(f"Ego Speed: {info['ego_speed']:.2f} m/s | NPC Speed: {info['npc_speed']:.2f} m/s", end='\r')

    def check_collision(self, info):
        """Check collision using the MTV-based SAT classification pipeline."""
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
        
        # Find colliding NPC
        for npc in road.vehicles:
            if npc is ego or id(npc) == self.controlled_vehicle_id:
                continue
            
            # Detect collision using MTV-based SAT
            dt = 1.0 / env.config["simulation_frequency"]
            mtv_result = detect_and_classify_collision(
                ego_polygon=ego.polygon(),
                npc_polygon=npc.polygon(),
                ego_velocity=ego.velocity * dt,
                npc_velocity=npc.velocity * dt,
                debug=True,  # Include all projection data for debugging
            )
            
            if mtv_result is not None:
                print("\n" + "="*60)
                print("COLLISION DETECTED (MTV-Based SAT)")
                print("="*60)
                print()
                print_classification(mtv_result.classification)
                print("="*60 + "\n")
                
                # Auto-quit after collision
                self.running = False
                return
    
    def run(self):
        self.env.reset()
        
        # Store controlled vehicle ID after reset
        env = self.env.unwrapped
        if env.vehicle is not None:
            self.controlled_vehicle_id = id(env.vehicle)
        
        while self.running:
            action = self.get_keyboard_action()
            
            if action == "RESET":
                self.env.reset()
                # Re-store controlled vehicle ID after reset
                env = self.env.unwrapped
                if env.vehicle is not None:
                    self.controlled_vehicle_id = id(env.vehicle)
                continue
            
            _, _, _, _, info = self.env.step(action.value if action != "RESET" else Action.IDLE.value)

            # Calculate and add speeds to info (matching src/dqn_crasher/training/runner.py:316-321)
            env = self.env.unwrapped
            if env.vehicle is not None:
                ego = env.vehicle
                info["ego_speed"] = (ego.velocity[0] ** 2 + ego.velocity[1] ** 2) ** 0.5

                # Find closest NPC
                for npc in env.road.vehicles:
                    if npc is not ego:
                        info["npc_speed"] = (npc.velocity[0] ** 2 + npc.velocity[1] ** 2) ** 0.5
                        break

            self.env.render()
            self.print_speeds(info)
            self.check_collision(info)
        
        self.env.close()


if __name__ == "__main__":
    try:
        controller = SATContactController()
        controller.run()
    except KeyboardInterrupt:
        print("\nQuitting...")
