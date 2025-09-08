# Highway-Env Observations Reference

This document provides a reference for Highway-Env observation types, specifically focusing on the **KinematicObservation** used in this collision detection research.

## Overview

Highway-Env supports several observation types that can be configured through the environment configuration. Each observation type provides different representations of the environment state.

**Source**: [Highway-Env Observations Documentation](https://highway-env.farama.org/observations/)

## Observation Types

### 1. KinematicObservation (Used in this research)

The KinematicObservation provides a **V × F** array describing **V** nearby vehicles with **F** features each.

#### Configuration

```yaml
observation:
  type: Kinematics
  normalize: False # Use raw values, not normalized
  see_behind: True # Include vehicles behind ego
  absolute: True # Use absolute world coordinates
  vehicles_count: 5 # Observe up to 5 vehicles
  features: ["presence", "x", "y", "vx", "vy"] # Feature set
```

#### Features Used

| Feature    | Description                           | Range  |
| ---------- | ------------------------------------- | ------ |
| `presence` | Vehicle exists (1) or placeholder (0) | {0, 1} |
| `x`        | World X coordinate (absolute)         | meters |
| `y`        | World Y coordinate (absolute)         | meters |
| `vx`       | Velocity in X direction               | m/s    |
| `vy`       | Velocity in Y direction               | m/s    |

#### Example Observation Matrix

```
Vehicle     | presence | x    | y   | vx   | vy  |
------------|----------|------|-----|------|-----|
ego-vehicle | 1.0      | 5.0  | 4.0 | 15.0 | 0.0 |
vehicle 1   | 1.0      | -10.0| 4.0 | 12.0 | 0.0 |
vehicle 2   | 1.0      | 13.0 | 8.0 | 13.5 | 0.0 |
vehicle 3   | 0.0      | 0.0  | 0.0 | 0.0  | 0.0 |
vehicle 4   | 0.0      | 0.0  | 0.0 | 0.0  | 0.0 |
```

**Note**: The ego-vehicle is always in the first row. Placeholder vehicles (presence=0) fill remaining slots.

#### Key Properties for Collision Detection

1. **Absolute Coordinates**: Positions are in world coordinates, making collision geometry calculations straightforward
2. **Non-normalized**: Raw meter and m/s values for direct physical interpretation
3. **Velocity Components**: `vx` and `vy` provide full velocity vectors for collision analysis
4. **Presence Flag**: Distinguishes real vehicles from padding

### 2. Other Observation Types (Not used in this research)

#### Grayscale Image

- Provides W × H grayscale image of the scene
- Useful for CNN-based approaches
- Not suitable for precise collision geometry analysis

#### Occupancy Grid

- W × H × F grid around ego vehicle
- Each cell contains feature information
- Good for spatial reasoning but less precise than Kinematics

#### Time to Collision (TTC)

- Predicts collision times for nearby vehicles
- Encoded for discrete time bins and lanes
- Complementary to Kinematics for collision prediction

#### Lidar

- Distances and velocities in angular sectors
- Sensor-like representation
- Less detailed than full Kinematics

## Collision Detection Applications

### Position-Based Detection

```python
# Extract positions from observation
ego_pos = obs[0, 1:3]  # [x, y]
other_pos = obs[i, 1:3]  # [x, y] for vehicle i

# Calculate distance
distance = np.linalg.norm(other_pos - ego_pos)
collision = distance < collision_threshold
```

### Velocity-Based Analysis

```python
# Extract velocities
ego_vel = obs[0, 3:5]  # [vx, vy]
other_vel = obs[i, 3:5]  # [vx, vy]

# Calculate relative velocity
rel_vel = other_vel - ego_vel
approach_rate = np.dot(rel_vel, (other_pos - ego_pos)) / distance
```

### Directionality Classification

```python
# Calculate collision angle
rel_pos = other_pos - ego_pos
collision_angle = np.arctan2(rel_pos[1], rel_pos[0])

# Classify collision type
if abs(rel_pos[0]) > abs(rel_pos[1]):
    collision_type = "longitudinal"  # rear-end or head-on
else:
    collision_type = "lateral"  # side-swipe
```

## Configuration Examples

### Multi-Agent Configuration

```yaml
observation:
  observation_config:
    type: Kinematics
    normalize: False
    see_behind: True
    absolute: True
    vehicles_count: 5
  type: MultiAgentObservation
```

### Single Agent Configuration

```yaml
observation:
  normalize: false
  type: Kinematics
```

## Research Advantages

The KinematicObservation with absolute, non-normalized coordinates provides several advantages for collision detection research:

1. **Direct Physical Interpretation**: Values are in real-world units (meters, m/s)
2. **Precise Geometry**: Absolute coordinates enable accurate collision geometry calculations
3. **Complete Velocity Information**: Full velocity vectors support directionality analysis
4. **Consistent Reference Frame**: All vehicles in same coordinate system
5. **Collision Detection Ready**: No coordinate transformations needed

## Frame Stacking

When using frame stacking (temporal history), the observation becomes:

- Shape: `(vehicles_count, features * frame_stack)`
- Most recent frame data is at the end of the feature dimension
- Useful for analyzing collision approach patterns over time

## References

- [Highway-Env Observations Documentation](https://highway-env.farama.org/observations/)
- [Highway-Env GitHub Repository](https://github.com/Farama-Foundation/HighwayEnv)
- [Gymnasium Environment Documentation](https://gymnasium.farama.org/)
