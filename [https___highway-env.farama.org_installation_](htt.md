<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# [https://highway-env.farama.org/installation/](https://highway-env.farama.org/installation/)

[https://highway-env.farama.org/quickstart/](https://highway-env.farama.org/quickstart/)
[https://highway-env.farama.org/environments/highway/](https://highway-env.farama.org/environments/highway/)
[https://highway-env.farama.org/observations/](https://highway-env.farama.org/observations/)
[https://highway-env.farama.org/actions/](https://highway-env.farama.org/actions/)
[https://highway-env.farama.org/dynamics/](https://highway-env.farama.org/dynamics/)
[https://highway-env.farama.org/dynamics/road/lane/](https://highway-env.farama.org/dynamics/road/lane/)
[https://highway-env.farama.org/dynamics/road/road/](https://highway-env.farama.org/dynamics/road/road/)
[https://highway-env.farama.org/dynamics/road/regulation/](https://highway-env.farama.org/dynamics/road/regulation/)
[https://highway-env.farama.org/multi_agent/](https://highway-env.farama.org/multi_agent/)
[https://highway-env.farama.org/make_your_own/](https://highway-env.farama.org/make_your_own/)
[https://highway-env.farama.org/faq/](https://highway-env.farama.org/faq/)
scrape this entire website and put it all in one place from each tab

The **highway-env** documentation covers installation, environment setup, configuration, available observation types, and core environment features for simulation-based reinforcement learning tasks in traffic scenarios.[^1][^2][^3]

## Installation

- **Python version**: Requires Python ≥3.5.[^2]
- **Ubuntu graphics dependencies**:
    - Run:
`sudo apt-get update -y`
`sudo apt-get install -y python-dev libsdl-image1.2-dev libsdl-mixer1.2-dev libsdl-ttf2.0-dev libsdl1.2-dev libsmpeg-dev python-numpy subversion libportmidi-dev ffmpeg libswscale-dev libavformat-dev libavcodec-dev libfreetype6-dev gcc`.[^2]
- **Windows 10 recommendation**: Use the preferred setup method (details are not specified in the documentation).[^2]
- **Stable release**:
`pip install highway-env`.[^2]
- **Development version**:
`pip install --user git+https://github.com/eleurent/highway-env`.[^2]


## Getting Started

- **Environment creation**:

```python
import gymnasium
import highway_env
env = gymnasium.make('highway-v0', render_mode='rgb_array')
env.reset()
```

- **Configuration**: The environment uses a `config` dictionary, which can be accessed and modified after creation (`env.unwrapped.config`), or passed during creation as `config={...}`.[^1]
- **Customizing environments**: Change features such as number of lanes at runtime or during creation.[^1]


## Core Environments

- **Default scenario ("highway-v0")**: Multi-lane highway, reward for speed, right lane usage, and collision avoidance.[^3]
- **Key configuration settings**:
    - `observation`: `"type": "Kinematics"`
    - `action`: `"type": "DiscreteMetaAction"`
    - `lanes_count`: 4
    - `vehicles_count`: 50
    - `duration`: 40 (seconds)
    - `collision_reward`: -1
    - `reward_speed_range`:  (m/s)
    - `simulation_frequency`: 15 (Hz)
    - `policy_frequency`: 1 (Hz)
    - `other_vehicles_type`: `"highway_env.vehicle.behavior.IDMVehicle"`
    - `screen_width`: 600 (px), `screen_height`: 150 (px)[^3]
- **Faster simulation**: Use `highway-fast-v0` for ~15x speedup.[^3]


## Observations

- **Observation types**:
    - *Kinematics*: $V \times F$ array for nearby vehicles with features like position, velocity, and presence. Can use absolute or relative coordinates, and is optionally normalized.
    - *Grayscale Image*: $W \times H$ array of the scene; customizable shape, RGB-to-grayscale conversion weights, and stack size for frame stacking.
    - *Occupancy Grid*: $W \times H \times F$ grid around the ego vehicle, with feature channels for vehicle detection and kinematic properties.
    - *Time to Collision*: Array predicting collision times for nearby vehicles, encoded for discrete time bins and lanes.
    - *Lidar*: Array describing distances and velocities in angular sectors around the ego vehicle.
- **Custom configuration**:
    - Observation types set via the `config` dictionary:

```python
config = {
  "observation": {
    "type": "OccupancyGrid",
    "vehicles_count": 15,
    "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
    "features_range": {
      "x": [-100, 100],
      "y": [-100, 100],
      "vx": [-20, 20],
      "vy": [-20, 20]
    },
    "grid_size": [[-27.5, 27.5], [-27.5, 27.5]],
    "grid_step": [5, 5],
    "absolute": False
  }
}
env = gymnasium.make('highway-v0', config=config)
```


## Agent Training

- **Reinforcement learning integration**: Compatible with libraries like Stable Baselines3 (SB3) using DQN, PPO, etc.
- **Training example**:

```python
from stable_baselines3 import DQN
env = gymnasium.make("highway-fast-v0")
model = DQN('MlpPolicy', env, ...)
model.learn(int(2e4))
model.save("highway_dqn/model")
```

- **Custom models and improved performance**: Various observation types, reward shaping, and architectures (e.g., CNN or goal-conditioned policies) can be explored for better results.[^1]


## Summary Table

| Topic | Key Features/Details | Source |
| :-- | :-- | :-- |
| Installation | Ubuntu: apt-get dependencies <br> pip install highway-env | [^2] |
| Environment | `highway-v0`, customizable config via `config` dict | [^1][^3] |
| Observations | Kinematics, Grayscale, Occupancy Grid, Time to Collision, Lidar |  |
| Training | SB3 (DQN), PPO, custom policies | [^1] |

The above sections summarize the core content from the **highway-env** documentation for fast reference. For more detailed, tab-specific information, additional sections from the site can be accessed similarly.[^3][^1][^2]

<div style="text-align: center">⁂</div>

[^1]: https://highway-env.farama.org/installation/

[^2]: https://highway-env.farama.org/quickstart/

[^3]: https://highway-env.farama.org/environments/highway/

