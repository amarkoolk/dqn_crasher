import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from trajectory_store import TrajectoryStore
from matplotlib.animation import FuncAnimation





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


class TrajectoryVisualizer:
    def __init__(self, state_slices, relative_map=None, ego_size=(5,2), npc_size=(5,2),
                 lane_markings=[(-6, 'solid'), (-2, 'dashed'), (2, 'solid')], entity_names=None, frame_stack=1,
                 color_1 = 'green', color_2 = 'blue'):
        self.slices = state_slices
        self.relative_map = relative_map or {}
        self.ego_size      = ego_size
        self.npc_size      = npc_size
        self.lane_markings = lane_markings
        self.frame_stack   = frame_stack
        # Derive entities dynamically
        self.entities = list(self.slices.keys())# Display names per entity
        if entity_names is not None:
            # Expect dict {entity_key: display_name}
            self.display_names = entity_names
        else:
            self.display_names = {ent: ent for ent in self.entities}
        # Map colors and sizes per entity (customize as needed)
        self.color_map = {ent: (color_1 if i == 0 else color_2)
                            for i, ent in enumerate(self.entities)}
        self.size_map  = {ent: (ego_size if i == 0 else npc_size)
                            for i, ent in enumerate(self.entities)}
        self.cache = {}

    def load(self, transitions):
        data = self.extract(transitions)
        N = data['action'].shape[0]
        cache = {}
        cache['N'] = N
        for ent in self.entities:
            arr = data[ent]
            x, y = arr['x'], arr['y']
            cache[f'{ent}_x'] = x
            cache[f'{ent}_y'] = -y
            cache[f'{ent}_vx'] = arr['vx']
            cache[f'{ent}_vy'] = -arr['vy']
        cache['actions'] = data['action']
        cache['rewards'] = data['reward']
        cache['ttc_x'] = data['ttc_x']
        cache['ttc_y'] = data['ttc_y']
        return cache

    def extract(self, transitions):
        raw_states = []
        for t in transitions:
            # flatten *all* the nested lists into one 1D array
            flat = np.array(t['state']).ravel()

            if self.frame_stack > 1:
                # how many numbers per frame?
                dim = flat.size // self.frame_stack
                # pick the most recent frame
                s = flat[-dim:]
            else:
                s = flat

            raw_states.append(s)

        # now stack into (N, dim) as before
        states = np.stack(raw_states, axis=0)

        data = {}
        for ent, idxs in self.slices.items():
            data[ent] = {var: states[:, idx] for var, idx in idxs.items()}

        data['action'] = np.array([t['action'] for t in transitions])
        data['reward'] = np.array([t['reward'] for t in transitions])
        data['ttc_x'] = np.array([t['ttc_x'] for t in transitions])
        data['ttc_y'] = np.array([t['ttc_y'] for t in transitions])
        return data

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
        cache = self.load(transitions)
        N = cache['N']
        ents = self.entities

        fig, ax = plt.subplots(3, 2, figsize=(18, 6))

        # Bounding boxes
        all_x = np.concatenate([cache[f'{e}_x'] for e in ents])
        all_y = np.concatenate([cache[f'{e}_y'] for e in ents])
        x_min, x_max = all_x.min() - self.ego_size[0], all_x.max() + self.ego_size[0]
        y_min, y_max = all_y.min() - self.ego_size[1], all_y.max() + self.ego_size[1]
        ax[0,0].set_xlim(x_min, x_max)
        ax[0,0].set_ylim(y_min, y_max)
        for y_line, style in self.lane_markings:
            ax[0,0].hlines(y_line, x_min, x_max, colors='k', linestyles=style)
        for i in range(N):
            alpha = max(0.2, i / N)
            for e in ents:
                x = cache[f'{e}_x'][i]
                y = cache[f'{e}_y'][i]
                vx = cache[f'{e}_vx'][i]
                vy = cache[f'{e}_vy'][i]
                bbox = self._make_bbox(x, y, vx, vy, self.size_map[e], self.color_map[e], alpha)
                ax[0,0].add_patch(bbox)
        ax[0,0].axis('equal')
        ax[0,0].set_xlabel('X position')
        ax[0,0].set_ylabel('Y position')
        ax[0,0].set_title(title or 'Bounding Box Trajectories')
        box_handles = [patches.Patch(edgecolor=self.color_map[e], facecolor='none', label=self.display_names[e]) for e in ents]
        ax[0,0].legend(handles=box_handles)

        # Velocity
        t = np.arange(N)
        for e in ents:
            ax[0,1].plot(t, (cache[f'{e}_vx']**2 + cache[f'{e}_vy']**2)**0.5, label=self.display_names[e], color=self.color_map[e])
        ax[0,1].set_xlabel('Time step')
        ax[0,1].set_ylabel('Velocity')
        ax[0,1].set_title('Velocity vs Time')
        ax[0,1].grid(True)
        ax[0,1].legend()

        for e in ents:
            ax[1,1].plot(t, cache['ttc_x'], label=self.display_names[e], color=self.color_map[e])
        ax[1,1].set_xlabel('Time step')
        ax[1,1].set_ylabel('Time to Collision')
        ax[1,1].set_title('TTC X vs Time')
        ax[1,1].grid(True)
        ax[1,1].legend()

        for e in ents:
            ax[2,1].plot(t, cache['ttc_y'], label=self.display_names[e], color=self.color_map[e])
        ax[2,1].set_xlabel('Time step')
        ax[2,1].set_ylabel('Time to Collision')
        ax[2,1].set_title('TTC Y vs Time')
        ax[2,1].grid(True)
        ax[2,1].legend()

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_episode_with_bboxes_compare(vis1, transitions1, vis2, transitions2,
                                        title=None, global_frame=True):
        """
        Overlay two runs on the same 3×2 figure using each visualizer's styling.
        vis1, vis2: TrajectoryVisualizer instances
        transitions1, transitions2: episode transitions to compare
        """
        cache1 = vis1.load(transitions1)
        cache2 = vis2.load(transitions2)
        # align lengths
        N = min(cache1['N'], cache2['N'])
        # ensure same entities
        if vis1.entities != vis2.entities:
            raise ValueError("Both visualizers must share the same entities")
        ents = vis1.entities

        fig, ax = plt.subplots(3, 2, figsize=(18, 6))
        # --- Bounding boxes comparison ---
        # compute global min/max
        all_x = np.concatenate([
            cache1[f'{e}_x'][:N] for e in ents
        ] + [
            cache2[f'{e}_x'][:N] for e in ents
        ])
        all_y = np.concatenate([
            cache1[f'{e}_y'][:N] for e in ents
        ] + [
            cache2[f'{e}_y'][:N] for e in ents
        ])
        x_min, x_max = all_x.min() - vis1.ego_size[0], all_x.max() + vis1.ego_size[0]
        y_min, y_max = all_y.min() - vis1.ego_size[1], all_y.max() + vis1.ego_size[1]
        ax[0,0].set_xlim(x_min, x_max)
        ax[0,0].set_ylim(y_min, y_max)
        for y_line, style in vis1.lane_markings:
            ax[0,0].hlines(y_line, x_min, x_max, colors='k', linestyles=style)
        # draw both runs
        for i in range(N):
            alpha = max(0.2, i / N)
            # run1 solid
            for e in ents:
                x, y = cache1[f'{e}_x'][i], cache1[f'{e}_y'][i]
                vx, vy = cache1[f'{e}_vx'][i], cache1[f'{e}_vy'][i]
                bbox = vis1._make_bbox(x, y, vx, vy, vis1.size_map[e], vis1.color_map[e], alpha)
                bbox.set_linestyle('-')
                ax[0,0].add_patch(bbox)
            # run2 dashed
            for e in ents:
                x, y = cache2[f'{e}_x'][i], cache2[f'{e}_y'][i]
                vx, vy = cache2[f'{e}_vx'][i], cache2[f'{e}_vy'][i]
                bbox = vis2._make_bbox(x, y, vx, vy, vis2.size_map[e], vis2.color_map[e], alpha)
                bbox.set_linestyle('-')
                ax[0,0].add_patch(bbox)
        ax[0,0].axis('equal')
        ax[0,0].set_xlabel('X position')
        ax[0,0].set_ylabel('Y position')
        ax[0,0].set_title(title or 'Bounding Box Comparison')
        # ax[0,0].legend([
        #     patches.Line2D([0],[0], color='black', linestyle='-', label='Run 1'),
        #     patches.Line2D([0],[0], color='black', linestyle='--', label='Run 2'),
        # ])




        # --- Velocity comparison ---
        t = np.arange(N)
        for e in ents:
            v1 = np.hypot(cache1[f'{e}_vx'][:N], cache1[f'{e}_vy'][:N])
            v2 = np.hypot(cache2[f'{e}_vx'][:N], cache2[f'{e}_vy'][:N])
            ax[0,1].plot(t, v1, label=vis1.display_names[e])
            ax[0,1].plot(t, v2, label=vis2.display_names[e], linestyle='-')
        ax[0,1].set_xlabel('Time step')
        ax[0,1].set_ylabel('Velocity')
        ax[0,1].set_title('Velocity Comparison')
        ax[0,1].grid(True)
        ax[0,1].legend()
        # --- TTC X comparison ---
        for cache, vis, ls, lbl in [(cache1, vis1, '-', vis1.display_names['car2']), (cache2, vis2, '-', vis2.display_names['car2'])]:
            ax[1,1].plot(t, cache['ttc_x'][:N], label=lbl, linestyle=ls)
        ax[1,1].set_xlabel('Time step')
        ax[1,1].set_ylabel('TTC X')
        ax[1,1].set_title('TTC X Comparison')
        ax[1,1].grid(True)
        ax[1,1].legend()
        # --- TTC Y comparison ---
        for cache, vis, ls, lbl in [(cache1, vis1, '-', vis1.display_names['car2']), (cache2, vis2, '-', vis2.display_names['car2'])]:
            ax[2,1].plot(t, cache['ttc_y'][:N], label=lbl, linestyle=ls)
        ax[2,1].set_xlabel('Time step')
        ax[2,1].set_ylabel('TTC Y')
        ax[2,1].set_title('TTC Y Comparison')
        ax[2,1].grid(True)
        ax[2,1].legend()
        plt.tight_layout()
        plt.show()


    def animate_episode(self, transitions, title=None, interval=1000, repeat=False, save_path=None):
        cache = self.load(transitions)
        N = cache['N']
        t = np.arange(N)

        fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(18, 6))

        # BBox panel
        all_x = np.concatenate([cache[f'{e}_x'] for e in self.entities])
        all_y = np.concatenate([cache[f'{e}_y'] for e in self.entities])
        x_min, x_max = all_x.min() - self.ego_size[0], all_x.max() + self.ego_size[0]
        y_min, y_max = all_y.min() - self.ego_size[1], all_y.max() + self.ego_size[1]
        ax0.set_xlim(x_min, x_max)
        ax0.set_ylim(y_min, y_max)
        for y_line, style in self.lane_markings:
            ax0.hlines(y_line, x_min-30, x_max+30, colors='k', linestyles=style)
        ax0.set_xlabel('X position')
        ax0.set_ylabel('Y position')
        ax0.set_title(title or 'Episode Animation')

        # Velocity panel
        ax1.set_xlim(0, N)
        vmax = max(cache[f'{e}_vx'].max() for e in self.entities)
        ax1.set_ylim(0, vmax * 1.1)
        line_handles = {e: ax1.plot([], [], label=self.display_names[e], color=self.color_map[e])[0] for e in self.entities}
        ax1.set_xlabel('Time step')
        ax1.set_ylabel('Velocity')
        ax1.legend()
        ax1.grid(True)

        # Action panel
        action_line, = ax2.step([], [], where='post')
        ax2.set_xlim(0, N)
        ax2.set_xlabel('Time step')
        ax2.set_ylabel('Action')
        ax2.set_title('Actions vs Time')
        ax2.set_yticks([])

        # Pre-create patches
        patches_dict = {}
        for e in self.entities:
            x0 = cache[f'{e}_x'][0]
            y0 = cache[f'{e}_y'][0]
            vx0 = cache[f'{e}_vx'][0]
            vy0 = cache[f'{e}_vy'][0]
            patch = self._make_bbox(x0, y0, vx0, vy0, self.size_map[e], self.color_map[e], alpha=1.0)
            patch.set_label(self.display_names[e])
            ax0.add_patch(patch)
            patches_dict[e] = patch
        ax0.legend()

        def init():
            return list(patches_dict.values()) + list(line_handles.values()) + [action_line]

        def update(i):
            for e in self.entities:
                x = cache[f'{e}_x'][i]
                y = cache[f'{e}_y'][i]
                vx = cache[f'{e}_vx'][i]
                vy = cache[f'{e}_vy'][i]
                patch = patches_dict[e]
                patch.set_xy((x - self.size_map[e][0] / 2, y - self.size_map[e][1] / 2))
                patch.angle = np.degrees(np.arctan2(vy, vx))
                line_handles[e].set_data(t[:i+1], cache[f'{e}_vx'][:i+1])
            acts = cache['actions'][:i+1]
            action_line.set_data(t[:i+1], acts)
            ax2.set_yticks(np.unique(cache['actions']))
            ego_x = cache[f'{self.entities[0]}_x'][i]
            ax0.set_xlim(ego_x - 30, ego_x + 30)
            ax0.set_ylim(-30, 30)
            return list(patches_dict.values()) + list(line_handles.values()) + [action_line]

        anim = FuncAnimation(
            fig,
            update,
            frames=N,
            init_func=init,
            interval=interval,
            blit=False,
            repeat=repeat
        )

        if save_path:
            if save_path.endswith(os.sep):
                os.makedirs(save_path, exist_ok=True)
                fname = f"{self.display_names[self.entities[0]]}_vs_{self.display_names.get(self.entities[1], '')}.mp4"
                save_path = os.path.join(save_path, fname)
            else:
                parent = os.path.dirname(save_path)
                if parent and not os.path.isdir(parent):
                    os.makedirs(parent, exist_ok=True)
            anim.save(save_path, dpi=100)
            plt.close(fig)
        else:
            return anim


def load_visualizer_and_episodes(
    data_dir,
    state_slices,
    relative_map=None,
    ego_size=(5, 2),
    npc_size=(5, 2),
    lane_markings=None,
    scenario=None,
    color_1 = 'green',
    color_2 = 'blue'
):
    # Load metadata and frame_stack
    meta_path = os.path.join(data_dir, 'metadata.json')
    with open(meta_path) as f:
        metadata = json.load(f)
    frame_stack = metadata.get('frame_stack', 1)

    # Determine policy files
    pa = metadata.get('policy_A', [])
    pb = metadata.get('policy_B', [])
    multi = None
    if len(pa) > 1:
        multi = ('A', pa)
    elif len(pb) > 1:
        multi = ('B', pb)


    if multi:
        side, choices = multi
        choices.append('policies.DQNPolicy')
        if scenario is None:
            raise ValueError(f"Multiple entries in policy_{side}, please specify scenario among {choices}")
        if scenario not in choices:
            raise ValueError(f"scenario must be one of {choices}")
        file1 = scenario
        file2 = pb[0] if side == 'A' and pb else pa[0] if side == 'B' and pa else None
    else:
        if len(pa) >= 1:
            file1 = pa[0]
        else:
            raise ValueError("metadata.json must contain at least one entry in policy_A")
        file2 = pb[0] if pb else None

    name1 = file1.split('.')[-1]
    name2 = file2.split('.')[-1] if file2 else 'Opponent'

    # Load episodes
    ep_fn = f'{file1}.jsonl'
    ep_path = os.path.join(data_dir, ep_fn)
    store = TrajectoryStore(ep_path)
    episodes = store.load(ep_path)

    entity_names = {'car1': name1, 'car2': name2}
    vis = TrajectoryVisualizer(
        state_slices,
        relative_map=relative_map,
        ego_size=ego_size,
        npc_size=npc_size,
        lane_markings=lane_markings or [],
        entity_names=entity_names,
        frame_stack=frame_stack,
        color_1 = color_1,
        color_2 = color_2
    )
    return vis, episodes, name1, name2


# Example usage:
STATE_SPEC = [
    ("car1", ["presence", "x", "y", "vx", "vy"]),
    ("car2", ["presence", "x", "y", "vx", "vy"]),
]
RELATIVE_MAP = {
    # "car2": "car1",  # car2's x,y are relative to car1's x,y
}
STATE_SLICES = get_state_slices(STATE_SPEC)


data_dir1 = '../mobil_vs_scenarios/episodes/test'
data_dir2 = '../dqn_vs_scenarios_5stack/episodes/test_100000'
scenario = 'scenarios.CutIn'
mp4_path = os.path.join(data_dir1,'mp4s')
os.makedirs(mp4_path, exist_ok=True)
vis, episodes, name11, name21 = load_visualizer_and_episodes(
    data_dir1,
    STATE_SLICES,
    relative_map=RELATIVE_MAP,
    ego_size=(5,2),
    npc_size=(5,2),
    lane_markings=[(-6,'solid'),(-2,'dashed'),(2,'solid')],
    scenario=scenario
)


vis2, episodes2, name12, name22 = load_visualizer_and_episodes(
    data_dir2,
    STATE_SLICES,
    relative_map=RELATIVE_MAP,
    ego_size=(5,2),
    npc_size=(5,2),
    lane_markings=[(-6,'solid'),(-2,'dashed'),(2,'solid')],
    scenario=scenario,
    color_1 = 'green',
    color_2 = 'red'
)


ep_key = list(episodes.keys())[0]

transition1 = episodes[ep_key]
transition2 = episodes2[ep_key]


for i in range(len(list(episodes.keys()))):
    key = list(episodes.keys())[i]
    transition1 = episodes[ep_key]
    transition2 = episodes2[ep_key]
    TrajectoryVisualizer.plot_episode_with_bboxes_compare(vis, transition1, vis2, transition2)
