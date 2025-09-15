#!/usr/bin/env python3
"""
Manual Control for Collision Detection Research

This script allows manual control of vehicles using keyboard input to reproduce
and analyze collision scenarios. Based on highway-env's human rendering capabilities.

Controls:
- Arrow Keys: Vehicle control (left/right for lane changes, up/down for speed)
- Space: Emergency brake
- R: Reset environment
- Q: Quit

Usage:
    python examples/manual_control.py
"""

import os
import sys
import time
from enum import Enum

# Add current directory to path for imports  
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import gymnasium as gym
import highway_env
import numpy as np
import pygame
from utils.config import load_config

# Initialize pygame for keyboard input
pygame.init()


class Action(Enum):
    LANE_LEFT = 0
    IDLE = 1
    LANE_RIGHT = 2
    FASTER = 3
    SLOWER = 4


class ManualController:
    def __init__(self, env_config_path="configs/env/single_agent.yaml"): 
        """Initialize manual controller with environment configuration."""
        
        # Resolve config path relative to this script's directory
        if not os.path.isabs(env_config_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            env_config_path = os.path.join(script_dir, env_config_path)
            env_config_path = os.path.abspath(env_config_path)
        
        # Load environment configuration
        self.config = load_config(env_config_path)
        
        # Override config for manual control
        self.config["duration"] = 200  # Longer episodes for manual control
        self.config["policy_frequency"] = 1  # Responsive control (1 = every frame)
        self.config["simulation_frequency"] = 15  # Standard simulation frequency
        
        # Create environment with human rendering
        self.env = gym.make(
            "highway-v0",  # Use standard highway environment
            config=self.config,
            render_mode="human"
        )
        
        # Control state
        self.current_action = Action.IDLE
        self.running = True
        self.quit_on_crash = True  # Quit after crash for analysis
        
        print("Manual Control Initialized")
        print("Controls:")
        print("  Arrow Keys: Vehicle control")
        print("  Space: Emergency brake")
        print("  R: Reset environment")
        print("  Q: Quit")
        
    def get_keyboard_action(self):
        """Get action from keyboard input using pygame events and continuous key checking."""
        action = Action.IDLE
        
        # Process pygame events from the highway-env window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    self.running = False
                elif event.key == pygame.K_r:
                    return "RESET"
        
        # Check for continuous key presses (more responsive)
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            action = Action.LANE_LEFT
        elif keys[pygame.K_RIGHT]:
            action = Action.LANE_RIGHT
        elif keys[pygame.K_UP]:
            action = Action.FASTER
        elif keys[pygame.K_DOWN]:
            action = Action.SLOWER
        elif keys[pygame.K_SPACE]:
            action = Action.SLOWER
        
        return action
    
    def detect_collision_state(self, obs, info):
        """
        Detect if vehicles are in collision or near-collision state.
        
        Args:
            obs: Current observation
            info: Environment info dict
            
        Returns:
            dict: Collision analysis
        """
        # Extract vehicle states from observation
        # Assuming Kinematics observation with [presence, x, y, vx, vy] features
        if len(obs.shape) == 2 and obs.shape[1] >= 5:
            ego_state = obs[0]  # First vehicle (ego)
            
            collision_info = {
                'collision_detected': info.get('crashed', False),
                'ego_position': [ego_state[1], ego_state[2]],
                'ego_velocity': [ego_state[3], ego_state[4]],
                'nearby_vehicles': []
            }
            
            # Analyze nearby vehicles
            for i in range(1, min(obs.shape[0], 5)):  # Check up to 4 other vehicles
                if obs[i, 0] > 0:  # Vehicle present
                    vehicle_info = {
                        'position': [obs[i, 1], obs[i, 2]],
                        'velocity': [obs[i, 3], obs[i, 4]],
                        'distance': np.sqrt((obs[i, 1] - ego_state[1])**2 + 
                                          (obs[i, 2] - ego_state[2])**2)
                    }
                    collision_info['nearby_vehicles'].append(vehicle_info)
            
            return collision_info
        
        return {'collision_detected': False}
    
    def analyze_collision_directionality(self, collision_info, info):
        """
        Analyze collision directionality when collision is detected.
        
        Args:
            collision_info: Collision information from detect_collision_state
            info: Environment info dict (will be modified to add collision details)
            
        Returns:
            dict: Directionality analysis
        """
        if not collision_info['collision_detected']:
            return None
            
        ego_pos = np.array(collision_info['ego_position'])
        ego_vel = np.array(collision_info['ego_velocity'])
        
        analysis = {
            'collision_type': 'unknown',
            'impact_angle': None,
            'relative_velocity': None,
            'closest_vehicle': None,
            'ego_type': 'ego',  # First vehicle is always manually controlled
            'other_type': 'npc'  # All other vehicles are AI-controlled
        }
        
        if collision_info['nearby_vehicles']:
            # Find closest vehicle (likely collision partner)
            closest_vehicle = min(collision_info['nearby_vehicles'], 
                                key=lambda v: v['distance'])
            
            npc_pos = np.array(closest_vehicle['position'])
            npc_vel = np.array(closest_vehicle['velocity'])
            
            # Calculate relative motion vectors
            rel_pos = npc_pos - ego_pos  # Vector from ego to NPC
            rel_vel = npc_vel - ego_vel  # Relative velocity vector
            
            # Calculate impact angle (angle of collision vector)
            impact_angle = np.arctan2(rel_pos[1], rel_pos[0])
            
            # Classify collision type based on relative position and velocity
            if abs(rel_pos[0]) > abs(rel_pos[1]):  # Longitudinal collision
                if rel_pos[0] > 0:  # NPC is ahead of ego
                    if rel_vel[0] < 0:  # NPC moving slower than ego (ego catching up)
                        collision_type = "rear-end (ego rear-ended NPC)"
                    else:  # NPC moving faster than ego (NPC pulling away, shouldn't collide)
                        collision_type = "rear-end (unusual - NPC ahead but faster)"
                else:  # NPC is behind ego
                    if rel_vel[0] > 0:  # NPC moving faster than ego (NPC catching up)
                        collision_type = "rear-end (NPC rear-ended ego)"
                    else:  # Both moving same direction, ego faster
                        collision_type = "head-on (vehicles approaching each other)"
            else:  # Lateral collision
                if rel_pos[1] > 0:
                    collision_type = "side-swipe (from left)"
                else:
                    collision_type = "side-swipe (from right)"
            
            # Calculate relative speed magnitude
            relative_speed = np.linalg.norm(rel_vel)
            
            analysis.update({
                'collision_type': collision_type,
                'impact_angle': np.degrees(impact_angle),
                'relative_velocity': relative_speed,
                'closest_vehicle': closest_vehicle,
                'relative_position': rel_pos.tolist(),
                'approach_velocity': rel_vel.tolist()
            })
            
            # Add collision info to the environment info dict
            info['collision'] = {
                'kind': collision_type,
                'angle_deg': float(np.degrees(impact_angle)),
                'relative_speed': float(relative_speed),
                'ego_type': 'ego',
                'other_type': 'npc'
            }
        
        return analysis
    
    def run(self):
        """Run the manual control loop."""
        obs, info = self.env.reset()
        episode_count = 0
        step_count = 0
        
        print(f"\nStarting Episode {episode_count + 1}")
        
        while self.running:
            # Get keyboard input
            action = self.get_keyboard_action()
            
            # Handle reset
            if action == "RESET":
                obs, info = self.env.reset()
                episode_count += 1
                step_count = 0
                print(f"\nStarting Episode {episode_count + 1}")
                continue
            
            # Take environment step
            obs, reward, terminated, truncated, info = self.env.step(action.value)
            step_count += 1
            
            # Render environment
            self.env.render()
            
            # Analyze collision state
            collision_info = self.detect_collision_state(obs, info)
            
            if collision_info['collision_detected']:
                print(f"\n*** COLLISION DETECTED at step {step_count} ***")
                
                # Analyze collision directionality
                directionality = self.analyze_collision_directionality(collision_info, info)
                if directionality:
                    print(f"Collision Type: {directionality['collision_type']}")
                    print(f"Impact Angle: {directionality['impact_angle']:.1f} degrees")
                    print(f"Relative Speed: {directionality['relative_velocity']:.2f} m/s")
                    print(f"Vehicle Types: {directionality['ego_type']} vs {directionality['other_type']}")
                    print(f"Ego Position: {collision_info['ego_position']}")
                    print(f"Ego Velocity: {collision_info['ego_velocity']}")
                    print(f"Added to info['collision']: {info.get('collision', 'None')}")
                
                if self.quit_on_crash:
                    print("Quitting after crash for analysis...")
                    self.running = False
                    break
                else:
                    # Wait for user input before continuing
                    input("Press Enter to continue or 'r' + Enter to reset...")
            
            # Check for episode end
            if terminated or truncated:
                print(f"\nEpisode {episode_count + 1} ended after {step_count} steps")
                print(f"Reward: {reward:.2f}")
                
                # Auto-reset after episode end
                time.sleep(1)
                obs, info = self.env.reset()
                episode_count += 1
                step_count = 0
                print(f"\nStarting Episode {episode_count + 1}")
            
            # Small delay for control (reduced for better responsiveness)
            time.sleep(0.01)
        
        print(f"\nSession ended. Completed {episode_count} episodes.")
        self.env.close()


def main():
    """Main function to run manual control."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Manual Control for Collision Detection')
    parser.add_argument('--continue-after-crash', action='store_true', 
                       help='Continue playing after crashes instead of quitting')
    parser.add_argument('--config', default='configs/env/single_agent.yaml',
                       help='Environment configuration file (relative to script directory or absolute path)')
    
    args = parser.parse_args()
    
    try:
        controller = ManualController(args.config)
        controller.quit_on_crash = not args.continue_after_crash
        controller.run()
    except KeyboardInterrupt:
        print("\nManual control interrupted by user.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
