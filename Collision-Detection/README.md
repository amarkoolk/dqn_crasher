# Collision Detection Research

This folder contains the research infrastructure for analyzing collision directionality in highway driving scenarios.

## Overview

This research focuses on:

1. **Collision Detection**: Identifying when collisions occur between vehicles
2. **Directionality Analysis**: Determining the direction and type of collision (rear-end, side-swipe, head-on)
3. **Crash Reproduction**: Manual control and scenario replay for crash analysis

## Key Components

### Core Training Infrastructure

- `main.py` - Main training entry point
- `runner.py` - Multi-agent training runner
- `agents/dqn_agent.py` - DQN agent implementation
- `utils/` - Helper utilities and observation processing

### Environment Configuration

- `configs/env/` - Environment configurations with Kinematics observation
- `configs/model/` - Model training configurations

### Scenario Management

- `scenarios/` - Scenario definitions and policy implementations
- `examples/crash_wrappers.py` - Crash-focused environment wrappers

### Visualization & Analysis

- `viz/viz.py` - Trajectory visualization with oriented bounding boxes
- `examples/crash_analysis.py` - Crash analysis tools
- `examples/trajectory_parsing.py` - Trajectory parsing utilities

### Manual Control (Keyboard)

- `examples/manual_control.py` - Keyboard control implementation (to be created)

## Observation Configuration

Uses **KinematicObservation** with:

- **Features**: `{presence, x, y, vx, vy}` - absolute coordinates, non-normalized
- **Vehicles**: 5 vehicles observed
- **Absolute positioning**: True (world coordinates)
- **See behind**: True (observes vehicles behind ego)

Reference: [Highway-Env Observations](https://highway-env.farama.org/observations/)

## Usage

### Basic Environment Setup

```python
import gymnasium as gym
from configs.env.multi_agent import load_config

config = load_config("multi_agent.yaml")
env = gym.make("crash-v0", config=config, render_mode="rgb_array")
```

### Manual Control

```python
# Run manual control for crash reproduction
python examples/manual_control.py
```

### Trajectory Analysis

```python
# Analyze stored trajectories
python examples/crash_analysis.py
```

### Visualization

```python
# Visualize episodes with bounding boxes
from viz.viz import TrajectoryVisualizer
vis.plot_episode_with_bboxes(transitions)
```

## Research Goals

1. **Collision Geometry**: Analyze vehicle positions and velocities at collision
2. **Directionality Classification**: Classify collision types based on relative motion
3. **Crash Reproduction**: Use manual control to reproduce specific crash scenarios
4. **Validation**: Verify directionality detection against known crash scenarios

## Next Steps

1. Implement keyboard control for manual crash reproduction
2. Develop collision directionality detection algorithm
3. Create collision classification system
4. Validate against recorded crash scenarios
