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
                 lane_markings=[(-6, 'solid'), (-2, 'dashed'), (2, 'solid')], entity_names=None, frame_stack=1):
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
        self.color_map = {ent: ('green' if i == 0 else 'blue')
                            for i, ent in enumerate(self.entities)}
        self.size_map  = {ent: (ego_size if i == 0 else npc_size)
                            for i, ent in enumerate(self.entities)}
        self.cache = {}

    def load(self, transitions):
        data = self.extract(transitions)
        N = data['action'].shape[0]
        self.cache.clear()
        self.cache['N'] = N
        for ent in self.entities:
            arr = data[ent]
            x, y = arr['x'], arr['y']
            self.cache[f'{ent}_x'] = x
            self.cache[f'{ent}_y'] = -y
            self.cache[f'{ent}_vx'] = arr['vx']
            self.cache[f'{ent}_vy'] = -arr['vy']
        self.cache['actions'] = data['action']
        self.cache['rewards'] = data['reward']
        return self.cache

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

        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(18, 6))

        # Bounding boxes
        all_x = np.concatenate([cache[f'{e}_x'] for e in ents])
        all_y = np.concatenate([cache[f'{e}_y'] for e in ents])
        x_min, x_max = all_x.min() - self.ego_size[0], all_x.max() + self.ego_size[0]
        y_min, y_max = all_y.min() - self.ego_size[1], all_y.max() + self.ego_size[1]
        ax0.set_xlim(x_min, x_max)
        ax0.set_ylim(y_min, y_max)
        for y_line, style in self.lane_markings:
            ax0.hlines(y_line, x_min, x_max, colors='k', linestyles=style)
        for i in range(N):
            alpha = max(0.2, i / N)
            for e in ents:
                x = cache[f'{e}_x'][i]
                y = cache[f'{e}_y'][i]
                vx = cache[f'{e}_vx'][i]
                vy = cache[f'{e}_vy'][i]
                bbox = self._make_bbox(x, y, vx, vy, self.size_map[e], self.color_map[e], alpha)
                ax0.add_patch(bbox)
        ax0.axis('equal')
        ax0.set_xlabel('X position')
        ax0.set_ylabel('Y position')
        ax0.set_title(title or 'Bounding Box Trajectories')
        box_handles = [patches.Patch(edgecolor=self.color_map[e], facecolor='none', label=self.display_names[e]) for e in ents]
        ax0.legend(handles=box_handles)

        # Velocity
        t = np.arange(N)
        for e in ents:
            ax1.plot(t, (cache[f'{e}_vx']**2 + cache[f'{e}_vy']**2)**0.5, label=self.display_names[e], color=self.color_map[e])
        ax1.set_xlabel('Time step')
        ax1.set_ylabel('Velocity')
        ax1.set_title('Velocity vs Time')
        ax1.grid(True)
        ax1.legend()

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
    scenario=None
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
    ep_fn = f'episodes_{file1}.jsonl'
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
        frame_stack=frame_stack
    )
    return vis, episodes, name1, name2


# data_dir = '../mobil_vs_scenarios'
# with open(os.path.join(data_dir, 'metadata.json')) as f:
#     metadata = json.load(f)

# scenario_policies = None
# policy_A_name = None
# policy_B_name = None
# if len(metadata['policy_A']) > 1:
#     scenario_policies = metadata['policy_A']
#     policy_A_name = 'Scenario'
# elif len(metadata['policy_B']) > 1:
#     scenario_policy = metadata['policy_B']
#     policy_B_name = 'Scenario'

# if policy_A_name is not None:
#     policy_B_name = metadata['policy_B'][0].split('.')[1]
# elif policy_B_name is not None:
#     policy_A_name = metadata['policy_A'][0].split('.')[1]

# # Example usage:
# scenario_name = metadata['policy_A'][0]
# store = TrajectoryStore(os.path.join(data_dir, 'episodes_' + scenario_name + '.jsonl'))
# episodes = store.load(os.path.join(data_dir, 'episodes_' + scenario_name + '.jsonl'))
# vis = TrajectoryVisualizer(STATE_SLICES, None, entity_names={'car1': scenario_name, 'car2': policy_B_name})
# vis.plot_episode_with_bboxes(episodes[2], title=f"{scenario_name} vs {policy_B_name}")
# anim = vis.animate_episode(episodes[2], title = f'{scenario_name} vs {policy_B_name}', interval=1000)
# plt.show()


# Example usage:
STATE_SPEC = [
    ("car1", ["presence", "x", "y", "vx", "vy"]),
    ("car2", ["presence", "x", "y", "vx", "vy"]),
]
RELATIVE_MAP = {
    # "car2": "car1",  # car2's x,y are relative to car1's x,y
}
STATE_SLICES = get_state_slices(STATE_SPEC)


data_dir = '../mobil_vs_scenarios'
scenario = 'scenarios.CutIn'
mp4_path = os.path.join(data_dir,'mp4s')
os.makedirs(mp4_path, exist_ok=True)
vis, episodes, name1, name2 = load_visualizer_and_episodes(
    data_dir,
    STATE_SLICES,
    relative_map=RELATIVE_MAP,
    ego_size=(5,2),
    npc_size=(5,2),
    lane_markings=[(-6,'solid'),(-2,'dashed'),(2,'solid')],
    scenario=scenario
)

for i in range(len(list(episodes.keys()))):
    key = list(episodes.keys())[i]
    vis.plot_episode_with_bboxes(episodes[key], title=f"{name1} vs {name2}")
    # save_path = os.path.join(mp4_path, f'{scenario}_episode_{key}.mp4')
    # anim = vis.animate_episode(episodes[key], title=f"{name1} vs {name2}", save_path = save_path)
