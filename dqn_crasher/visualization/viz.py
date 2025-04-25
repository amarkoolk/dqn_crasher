import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from trajectory_store import TrajectoryStore
from matplotlib.animation import FuncAnimation



STATE_SPEC = [
    ("car1", ["presence", "x", "y", "vx", "vy"]),
    ("car2", ["presence", "x", "y", "vx", "vy"]),
]
RELATIVE_MAP = {
    "car2": "car1",  # car2's x,y are relative to car1's x,y
}

def get_state_slices(state_spec):
    """
    Build a dict mapping each entity to its variable name -> index in the flat state vector.
    """
    slices = {}
    idx = 0
    for entity, vars in state_spec:
        slices[entity] = {var: idx + i for i, var in enumerate(vars)}
        idx += len(vars)
    return slices

STATE_SLICES = get_state_slices(STATE_SPEC)

class TrajectoryVisualizer:
    def __init__(self, state_slices, relative_map=None, ego_size=(5,2), npc_size=(5,2),
                 lane_markings=[(-6, 'solid'), (-2, 'dashed'), (2, 'solid')]):
        self.slices = state_slices
        self.relative_map = relative_map or {}
        self.ego_size      = ego_size
        self.npc_size      = npc_size
        self.lane_markings = lane_markings

    def extract(self, transitions):
        """
        Given a list of transition dicts, each with 'state',
        return a nested dict:
          { entity: { var: np.array([...]) } }
        Automatically flattens nested lists if needed.
        """
        raw_states = []
        for t in transitions:
            state = np.array(t["state"])
            # Flatten if nested: e.g., shape (1,10) -> (10,)
            if state.ndim > 1 and state.shape[0] == 1:
                state = state.squeeze(0)
            raw_states.append(state)
        states = np.stack(raw_states, axis=0)

        data = {}
        for entity, idxs in self.slices.items():
            data[entity] = {}
            for var, idx in idxs.items():
                data[entity][var] = states[:, idx]
        return data

    def plot_episode(self, transitions, title=None, global_frame=True):
        """
        Plot the XY trajectories of each entity.
        If global_frame=True, entities listed in RELATIVE_MAP
        will be offset by their parent entity's position.
        """
        trajs = self.extract(transitions)
        plt.figure()
        for entity, d in trajs.items():
            mask = d["presence"] > 0.5
            xs = d["x"].copy()
            ys = -d["y"].copy()
            if global_frame and entity in self.relative_map:
                parent = self.relative_map[entity]
                xs = xs + trajs[parent]["x"]
                ys = ys + trajs[parent]["y"]
            plt.plot(xs[mask], ys[mask], label=entity)
        plt.xlabel("X position")
        plt.ylabel("Y position")
        plt.title(title or "Episode Trajectories")
        plt.legend()
        plt.axis('equal')
        plt.show()

    

    def plot_multiple(self, episodes, episode_indices=None, global_frame=True):
        """
        Plot multiple episodes in sequence.
        episodes: dict of episode_idx -> list of transitions
        episode_indices: list of episode indices to plot
        """
        if episode_indices is None:
            episode_indices = sorted(episodes.keys())
        for ep in episode_indices:
            print(f"--- Episode {ep} ---")
            self.plot_episode(episodes[ep], title=f"Episode {ep}", global_frame=global_frame)

    def _make_bbox(self, x, y, vx, vy, size, color, alpha):
        """
        Create a Rectangle patch for a vehicle.
        x,y: center position
        vx,vy: velocity components for heading
        size: (length, width)
        color: edge color
        alpha: transparency
        """
        length, width = size
        heading = np.degrees(np.arctan2(vy, vx))
        anchor_x = x - length / 2
        anchor_y = y - width / 2
        return patches.Rectangle(
            (anchor_x, anchor_y), 
            length, width,
            angle=heading,
            linewidth=1, edgecolor=color, facecolor='none', alpha=alpha
        )

    def plot_episode_with_bboxes(self, transitions, title=None, global_frame=True):
        """
        Static plot: bounding‐box history + lane lines + velocity subplot.
        """
        trajs = self.extract(transitions)
        # Prepare figure
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14,6))

        # Compute limits
        all_x, all_y = [], []
        for ent in trajs:
            xs = trajs[ent]["x"]
            ys = trajs[ent]["y"]
            if global_frame and ent in self.relative_map:
                parent = self.relative_map[ent]
                xs = xs + trajs[parent]["x"]
                ys = ys + trajs[parent]["y"]
            all_x.append(xs)
            all_y.append(ys)
        all_x = np.concatenate(all_x)
        all_y = np.concatenate(all_y)
        x_min, x_max = np.min(all_x)-self.ego_size[0], np.max(all_x)+self.ego_size[0]
        y_min, y_max = np.min(all_y)-self.ego_size[1], np.max(all_y)+self.ego_size[1]

        # 1) Draw lane markings
        for y, style in self.lane_markings:
            ax0.hlines(y, x_min, x_max, colors='k', linestyles=style)

        # 2) Draw bounding boxes over time
        N = len(transitions)
        for i in range(N):
            alpha = max(0.2, float(i)/N)
            # Ego
            ex = trajs["car1"]["x"][i]
            ey = -trajs["car1"]["y"][i]
            evx= trajs["car1"]["vx"][i]
            evy= trajs["car1"]["vy"][i]
            bbox_e = self._make_bbox(ex, ey, evx, evy, self.ego_size, 'blue', alpha)
            ax0.add_patch(bbox_e)

            # NPC (global)
            nx = trajs["car2"]["x"][i]
            ny = -trajs["car2"]["y"][i]
            if global_frame and "car2" in self.relative_map:
                parent = self.relative_map["car2"]
                nx += trajs[parent]["x"][i]
                ny += trajs[parent]["y"][i]
            nvx = trajs["car2"]["vx"][i] + evx
            nvy = trajs["car2"]["vy"][i] + evy
            bbox_n = self._make_bbox(nx, ny, nvx, nvy, self.npc_size, 'red', alpha)
            ax0.add_patch(bbox_n)

        ax0.axis('equal')
        ax0.set_xlabel("X position")
        ax0.set_ylabel("Y position")
        ax0.set_title(title or "Bounding Box Trajectories")

        # 3) Velocity vs time
        t = np.arange(N)
        v_ego = trajs["car1"]["vx"]
        v_npc = trajs["car2"]["vx"] + trajs["car1"]["vx"]
        ax1.plot(t, v_ego, label="Scenario NPC")
        ax1.plot(t, v_npc, label="MOBIL")
        ax1.set_xlabel("Time step")
        ax1.set_ylabel("Velocity")
        ax1.set_title("Velocity vs Time")
        ax1.grid(True)
        ax1.legend()

        plt.tight_layout()
        plt.show()

    def animate_episode(self,
                        transitions,
                        title=None,
                        interval=100,
                        repeat=False,
                        save_path=None):
        """
        Animate one episode:
        - Left panel: bounding boxes + lane lines over time
        - Right panel: velocity curves growing over time

        Args:
            transitions: list of dicts with 'state'
            title: optional plot title
            interval: ms between frames
            repeat: whether to loop the animation
            save_path: filepath (e.g., 'episode0.mp4') to save; if None, just return the FuncAnimation
        """
        trajs = self.extract(transitions)
        N = len(transitions)

        # Precompute positions and velocities
        ego_x = trajs["car1"]["x"]
        ego_y = -trajs["car1"]["y"]
        ego_vx = trajs["car1"]["vx"]
        ego_vy = trajs["car1"]["vy"]

        npc_x = trajs["car2"]["x"].copy()
        npc_y = -trajs["car2"]["y"].copy()
        npc_vx = trajs["car2"]["vx"]
        npc_vy = trajs["car2"]["vy"]

        # Setup figure & axes
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 6))
        # Axis 0 limits & lanes
        x_all = np.concatenate([ego_x, npc_x])
        y_all = np.concatenate([ego_y, npc_y])
        x_min, x_max = x_all.min() - self.ego_size[0], x_all.max() + self.ego_size[0]
        y_min, y_max = y_all.min() - self.ego_size[1], y_all.max() + self.ego_size[1]
        ax0.set_xlim(x_min, x_max)
        ax0.set_ylim(y_min, y_max)
        for y, style in self.lane_markings:
            ax0.hlines(y, x_min-30, x_max+30, colors='k', linestyles=style)
        ax0.set_xlabel("X position")
        ax0.set_ylabel("Y position")
        ax0.set_title(title or "Episode Animation")

        # Axis 1 setup
        t = np.arange(N)
        ax1.set_xlim(0, N)
        v_max = max(ego_vx.max(), npc_vx.max())
        ax1.set_ylim(0, v_max * 1.1)
        line_ego, = ax1.plot([], [], label="Scenario NPC", color='blue')
        line_npc, = ax1.plot([], [], label="MOBIL", color='red')
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Velocity")
        ax1.legend()
        ax1.grid(True)

        # Animation update function
        def update(i):
            # Clear previous boxes
            for patch in ax0.patches[:]:
                patch.remove()

            # Draw bounding boxes up to current frame i
            # for j in range(i + 1):
            alpha = 1.0
            # Ego bbox
            bbox_e = self._make_bbox(
                ego_x[i], ego_y[i], ego_vx[i], ego_vy[i],
                self.ego_size, 'blue', alpha
            )
            ax0.add_patch(bbox_e)
            # NPC bbox
            bbox_n = self._make_bbox(
                npc_x[i], npc_y[i], npc_vx[i], npc_vy[i],
                self.npc_size, 'red', alpha
            )
            ax0.add_patch(bbox_n)

            # Update Axes Limits
            xmin = ego_x[i] - 30
            xmax = ego_x[i] + 30
            ymin = -30
            ymax = 30
            ax0.set_xlim(xmin, xmax)
            ax0.set_ylim(ymin, ymax)

            # Update velocity lines
            line_ego.set_data(t[:i + 1], ego_vx[:i + 1])
            line_npc.set_data(t[:i + 1], npc_vx[:i + 1])
            return ax0.patches + [line_ego, line_npc]

        anim = FuncAnimation(fig, update, frames=N,
                             interval=interval, blit=False, repeat=repeat)

        if save_path:
            anim.save(save_path, dpi=100)
            plt.close(fig)
        else:
            return anim

# Example usage:
store = TrajectoryStore("trajectories/my_agent.jsonl")
episodes = store.load('../mobil_vs_cutin/all_episodes.jsonl')
vis = TrajectoryVisualizer(STATE_SLICES, None)
vis.plot_episode_with_bboxes(episodes[2], title='Episode 0')
anim = vis.animate_episode(episodes[2], title = 'Cutin vs MOBIL',interval=1000)
plt.show()