#!/usr/bin/env python3

import os
import sys
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

# Add current directory to path for imports  
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import gymnasium as gym
import highway_env
import numpy as np
import pygame
from utils.config import load_config

from sat_collision import separating_axis_theorem

# Initialize pygame for keyboard input
pygame.init()


class Action(Enum):
    LANE_LEFT = 0
    IDLE = 1
    LANE_RIGHT = 2
    FASTER = 3
    SLOWER = 4


@dataclass(frozen=True)
class CollisionReport:
    classification: str
    impact_angle_deg: float
    longitudinal_component: float
    lateral_component: float
    relative_speed: float
    heading_difference_deg: float
    ego_heading_deg: float
    other_heading_deg: float


class UniversalManualController:
    """
    Manual controller with HEADING-AWARE collision classification.
    Works correctly in ALL highway-env environments:
    - highway-v0 (straight roads)
    - roundabout-v0 (circular roads)
    - intersection-v0 (crossroads)
    - parking-v0 (parking lot)
    - merge-v0 (merging lanes)
    - racetrack-v0 (oval track)
    """
    
    def __init__(self, env_config_path="configs/env/single_agent.yaml"): 
        """Initialize manual controller with environment configuration."""
        
        if not os.path.isabs(env_config_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            env_config_path = os.path.join(script_dir, env_config_path)
            env_config_path = os.path.abspath(env_config_path)
        
        self.config = load_config(env_config_path)
        
        self.env = gym.make(
            "highway-v0",
            config=self.config,
            render_mode="human"
        )
        
        self.running = True
        self.quit_on_crash = True
        
        print("Universal Manual Control Initialized")
        print("Controls:")
        print("  Left/Right Arrows: Lane changes")
        print("  Up/Down Arrows: Speed control (faster/slower)")
        print("  Space: Emergency brake (slower)")
        print("  R: Reset environment")
        print("  Q: Quit")
        print("\n⭐ HEADING-AWARE collision classification enabled!")
        print("   Works in: highway, roundabout, intersection, parking, etc.")
        
    def get_keyboard_action(self):
        """Get action from keyboard input using pygame events and continuous key checking."""
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
        elif keys[pygame.K_DOWN]:
            action = Action.SLOWER
        elif keys[pygame.K_SPACE]:
            action = Action.SLOWER
        
        return action
    
    def compute_collision_report(self, info: dict) -> Optional[CollisionReport]:
        """
        Use SAT to derive the MTV and classify the crash using ego-relative axes.
        """
        if not info.get("crashed", False):
            return None

        env = self.env.unwrapped
        ego = env.vehicle
        road = env.road
        if ego is None or road is None:
            return None

        dt = 1.0 / env.config["simulation_frequency"]

        collisions = []
        for vehicle in road.vehicles:
            if vehicle is ego:
                continue
            result = separating_axis_theorem(
                ego.polygon(),
                vehicle.polygon(),
                ego.velocity * dt,
                vehicle.velocity * dt,
            )
            if result.intersecting and result.minimum_translation_vector is not None:
                collisions.append((vehicle, result.minimum_translation_vector))

        if not collisions:
            return None

        other, mtv = min(collisions, key=lambda pair: np.linalg.norm(pair[1]))
        classification, impact_angle, long_component, lat_component = self.classify_mtv(
            mtv, ego.heading
        )
        if not classification:
            return None

        relative_velocity = other.velocity - ego.velocity
        heading_diff = (other.heading - ego.heading + np.pi) % (2 * np.pi) - np.pi

        report = CollisionReport(
            classification=classification,
            impact_angle_deg=impact_angle,
            longitudinal_component=long_component,
            lateral_component=lat_component,
            relative_speed=float(np.linalg.norm(relative_velocity)),
            heading_difference_deg=float(np.degrees(heading_diff)),
            ego_heading_deg=float(np.degrees(ego.heading)),
            other_heading_deg=float(np.degrees(other.heading)),
        )

        info["collision"] = {
            "kind": report.classification,
            "angle_deg": report.impact_angle_deg,
            "relative_speed": report.relative_speed,
            "longitudinal_mtv": report.longitudinal_component,
            "lateral_mtv": report.lateral_component,
            "heading_diff_deg": report.heading_difference_deg,
        }
        return report

    @staticmethod
    def classify_mtv(
        mtv: np.ndarray, heading: float
    ) -> tuple[Optional[str], float, float, float]:
        """
        Project the MTV onto the ego frame and map it to a human label.
        """
        if np.linalg.norm(mtv) < 1e-6:
            return None, float("nan"), 0.0, 0.0

        ego_forward = np.array([np.cos(heading), np.sin(heading)])
        ego_left = np.array([-np.sin(heading), np.cos(heading)])

        longitudinal = float(np.dot(mtv, ego_forward))
        lateral = float(np.dot(mtv, ego_left))

        impact_angle_deg = float(
            np.degrees(np.arctan2(lateral, longitudinal))
        )  # ego frame angle

        if abs(longitudinal) >= abs(lateral):
            label = (
                "rear impact (other hit ego from behind)"
                if longitudinal > 0
                else "front impact (ego hit vehicle ahead)"
            )
        else:
            label = (
                "right-side impact (other on ego's right)"
                if lateral > 0
                else "left-side impact (other on ego's left)"
            )

        return label, impact_angle_deg, longitudinal, lateral
    
    def run(self):
        """Run the manual control loop."""
        _, info = self.env.reset()
        episode_count = 0
        step_count = 0
        
        print(f"\nStarting Episode {episode_count + 1}")
        
        while self.running:
            action = self.get_keyboard_action()
            
            if action == "RESET":
                _, info = self.env.reset()
                episode_count += 1
                step_count = 0
                print(f"\nStarting Episode {episode_count + 1}")
                continue
            
            _, reward, terminated, truncated, info = self.env.step(action.value)
            step_count += 1
            self.env.render()
            
            report = self.compute_collision_report(info)
            
            if report:
                print(f"\n*** COLLISION DETECTED at step {step_count} ***")
                print(f"Classification: {report.classification}")
                print(f"Impact Angle: {report.impact_angle_deg:.1f}° (ego frame)")
                print(f"MTV (longitudinal, lateral): {report.longitudinal_component:.2f} m, "
                      f"{report.lateral_component:.2f} m")
                print(f"Relative Speed: {report.relative_speed:.2f} m/s")
                print(f"Heading Difference: {report.heading_difference_deg:.1f}°")
                print(f"Ego Heading: {report.ego_heading_deg:.1f}°")
                print(f"Other Heading: {report.other_heading_deg:.1f}°")
                print(f"Added to info['collision']: {info.get('collision', 'None')}")

                if self.quit_on_crash:
                    print("Quitting after crash for analysis...")
                    self.running = False
                    break
                else:
                    input("Press Enter to continue or 'r' + Enter to reset...")
            
            if terminated or truncated:
                print(f"\nEpisode {episode_count + 1} ended after {step_count} steps")
                print(f"Reward: {reward:.2f}")
                
                time.sleep(1)
                _, info = self.env.reset()
                episode_count += 1
                step_count = 0
                print(f"\nStarting Episode {episode_count + 1}")
            
            time.sleep(0.01)
        
        print(f"\nSession ended. Completed {episode_count} episodes.")
        self.env.close()


def main():
    """Main function to run manual control."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Universal Manual Control for Collision Detection')
    parser.add_argument('--continue-after-crash', action='store_true', 
                       help='Continue playing after crashes instead of quitting')
    parser.add_argument('--config', default='configs/env/single_agent.yaml',
                       help='Environment configuration file (relative to script directory or absolute path)')
    
    args = parser.parse_args()
    
    try:
        controller = UniversalManualController(args.config)
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
