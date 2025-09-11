# Collision Detection Research

This folder contains a clean setup for collision detection research using manual control to analyze collision directionality in highway driving scenarios.

## Overview

This research focuses on:

1. **Collision Detection**: Identifying when collisions occur between vehicles
2. **Directionality Analysis**: Determining the direction and type of collision (rear-end, side-swipe, head-on)
3. **Crash Reproduction**: Manual control for crash analysis and research

## Key Components

### Manual Control

- `manual_control.py` - Keyboard-controlled vehicle for collision research

### Environment Configuration

- `configs/env/` - Environment configurations with Kinematics observation
  - `single_agent.yaml` - Single-agent configuration for manual control
  - `multi_agent.yaml` - Multi-agent configuration

### Utilities

- `utils/config.py` - Configuration loading utility

### Documentation & Visualization

- `docs/` - Documentation and observation references
- `viz/` - Trajectory visualization tools (for future analysis)

## Observation Configuration

Uses **KinematicObservation** with:

- **Features**: `{presence, x, y, vx, vy}` - absolute coordinates, non-normalized
- **Vehicles**: 5 vehicles observed
- **Absolute positioning**: True (world coordinates)
- **See behind**: True (observes vehicles behind ego)

### Traffic Configuration

Both configurations include **continuous vehicle spawning** to ensure the road stays populated:

- **vehicles_count**: Initial number of vehicles spawned at episode start
- **spawn_probability**: Probability (0.0-1.0) of spawning new vehicles each simulation step
  - `single_agent.yaml`: 0.5 probability (moderate traffic)
  - `multi_agent.yaml`: 0.6 probability (heavier traffic)

Reference: [Highway-Env Observations](https://highway-env.farama.org/observations/)

## Usage

### Manual Control

Run the manual control script to start keyboard-controlled driving:

```bash
# Basic usage with default single-agent config
python manual_control.py

# Use multi-agent configuration
python manual_control.py --config configs/env/multi_agent.yaml

# Continue playing after crashes (instead of quitting)
python manual_control.py --continue-after-crash
```

### Controls

- **Arrow Keys**: Vehicle control (left/right for lane changes, up/down for speed)
- **Space**: Emergency brake
- **R**: Reset environment
- **Q**: Quit

### Environment Setup

The environment configurations are optimized for collision research:

```python
import gymnasium as gym
from utils.config import load_config

config = load_config("configs/env/single_agent.yaml")
env = gym.make("highway-v0", config=config, render_mode="human")
```

## Research Goals

1. **Collision Geometry**: Analyze vehicle positions and velocities at collision using manual control
2. **Directionality Classification**: Classify collision types (rear-end, side-swipe, head-on) based on relative motion
3. **Crash Reproduction**: Use keyboard control to reproduce and study specific crash scenarios
4. **Real-time Analysis**: Detect collision states and directionality during manual driving

## Getting Started

1. Run `python manual_control.py` to start manual driving
2. Use keyboard controls to explore different collision scenarios
3. Observe collision detection and analysis output in real-time
4. Experiment with different environment configurations
