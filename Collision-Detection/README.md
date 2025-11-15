# Collision Detection - Contact Tracking Method

This folder contains the implementation of a contact tracking collision detection method for autonomous vehicle simulation.

## Overview

This implementation provides:

1. **Contact Tracking Detection**: Track which vertices hit which edges with rich collision data
2. **Simple Output**: Clear statements of "EGO's corner X hit NPC's edge Y"
3. **Active State Tracking**: Maintains edges and vertices of both EGO and NPC vehicles

## Core Method: Contact Tracking

**File**: `contact_tracking.py`

**Algorithm**:

- Test each vertex of polygon A against each edge of polygon B
- Test each vertex of polygon B against each edge of polygon A
- Record all contact points where vertices penetrate edges
- Compute contact normals directly from edge geometry

**Advantages**:

- **Richer data**: Exact contact geometry (which corner hit which edge)
- **Detailed tracking**: Records vertex ID, edge ID, position, normal, and penetration depth
- **Simpler geometry**: Just point-to-edge distance tests

**Output Data**:

- Vertex ID and Edge ID for each contact
- Contact position (world coordinates)
- Contact normal vector
- Penetration depth

## Key Components

### Collision Detection

- `contact_tracking.py` - **YOUR METHOD:** Clean implementation including:
  - `Contact` dataclass (vertex/edge contact data)
  - `find_primary_contact()` function (main detection algorithm)
  - `classify_collision()` function (collision classification)
  - `format_contact()` function (human-readable formatting)
  - Uses SAT to detect collision and MTV to identify collision side
- `sat_collision.py` - Original SAT method (for reference/baseline)

### Manual Control

- `manual_control_contact.py` - **Test controller** with contact tracking collision detection
- `manual_control_universal.py` - Original SAT-based controller (baseline)

### Environment Configuration

- `configs/env/` - Environment configurations with Kinematics observation
  - `single_agent.yaml` - Single-agent configuration for manual control
  - `multi_agent.yaml` - Multi-agent configuration

### Utilities

- `utils/config.py` - Configuration loading utility

### Documentation

- `docs/` - Documentation and observation references

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

### Test Contact Tracking Method

Run the contact tracking controller:

```bash
cd Collision-Detection
python manual_control_contact.py
```

**Output Example**:

```
COLLISION DETECTED!
Classification: Rear-End Collision

Contact Details:
  • EGO's Front-Right Corner hit NPC's Rear edge
  • EGO's Front-Left Corner hit NPC's Rear edge
```

### Original SAT-based Control (Baseline)

Run the original controller for comparison:

```bash
python manual_control_universal.py

# Or with options:
python manual_control_universal.py --continue-after-crash --config configs/env/multi_agent.yaml
```

### Controls

- **Arrow Keys**: Vehicle control (left/right for lane changes, up/down for speed)
- **Space**: Emergency brake
- **R**: Reset environment
- **Q**: Quit

## Contact Tracking Implementation Details

The contact tracking method provides detailed information about each contact point:

```python
from contact_tracking import find_primary_contact, classify_collision, format_contact

# Detect collision and find primary contact
dt = 1.0 / env.config["simulation_frequency"]
contact = find_primary_contact(
    ego_polygon=ego.polygon(),
    npc_polygon=npc.polygon(),
    ego_velocity=ego.velocity * dt,
    npc_velocity=npc.velocity * dt,
    threshold=0.5  # Distance threshold in meters
)

# Access results
if contact:
    print(f"Contact: {format_contact(contact)}")
    print(f"EGO vertex {contact.ego_vertex_id} hit NPC edge {contact.npc_edge_id}")
    print(f"Position: {contact.position}")
    print(f"Distance: {contact.distance}m")

    # Classify collision
    classification = classify_collision(contact)
    print(f"Classification: {classification}")
```

**Data Structure**:

```python
@dataclass(frozen=True)
class Contact:
    ego_vertex_id: int       # ID of EGO vertex (0-3, see ordering below)
    npc_edge_id: int         # ID of NPC edge (0-3, see ordering below)
    distance: float          # Distance from vertex to edge (meters)
    position: np.ndarray     # Contact position (world coordinates)
    signed_distance: float   # Signed distance (for internal use)
```

**Highway-env Vertex Ordering** (verified from polygon data):

- 0: Rear-Right Corner
- 1: Rear-Left Corner
- 2: Front-Left Corner
- 3: Front-Right Corner

**Highway-env Edge Ordering**:

- 0: Rear edge (connects vertex 0 → 1)
- 1: Left edge (connects vertex 1 → 2)
- 2: Front edge (connects vertex 2 → 3)
- 3: Right edge (connects vertex 3 → 0)

## Research Goals

1. **Collision Geometry**: Analyze exact contact points during collisions
2. **Active State Tracking**: Maintain and track all vertices and edges
3. **Richer Collision Data**: Identify which specific corner hit which specific edge
4. **Directionality Classification**: Classify collision types based on contact geometry
5. **Future DQN Integration**: Use rich collision data for reward shaping

## Environment Setup

The environment configurations are optimized for collision research:

```python
import gymnasium as gym
from utils.config import load_config

config = load_config("configs/env/single_agent.yaml")
env = gym.make("highway-v0", config=config, render_mode="human")
```

## Getting Started

1. Run `python manual_control_contact.py` to test contact tracking
2. Use keyboard controls to explore different collision scenarios
3. Observe detailed contact information in real-time
4. Compare with original SAT method using `manual_control_universal.py`
