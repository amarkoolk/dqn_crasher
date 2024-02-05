from yaml import load, dump
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = load(file, Loader=Loader)
    return config

def save_config(config, config_path):
    with open(config_path, 'w') as file:
        dump(config, file, Dumper=Dumper)

if __name__ == "__main__":
    # Create Example Dictionary, then Save it to a Yaml and then Load it
    example_dict = {
            "controlled_vehicles": 2,
            "observation": {
                "type": "MultiAgentObservation",
                "observation_config": {
                    "type": "Kinematics",
                }
            },
            "action": {
                "type": "MultiAgentAction",
                "action_config": {
                    "type": "DiscreteMetaAction",
                }
            },
            "lanes_count" : 2,
            "vehicles_count" : 0,
            "duration" : 10,
            "initial_lane_id" : None,
            "policy_frequency": 1,
            # Reset Configs
            'spawn_configs' : ['behind_left', 'behind_right', 'behind_center', 'adjacent_left', 'adjacent_right', 'forward_left', 'forward_right', 'forward_center'],
            'mean_distance' : 20,
            'initial_speed' : 20,
            'mean_delta_v' : 0,
            # Crash Configs
            'ttc_x_reward' : 4,
            'ttc_y_reward' : 1,
            'crash_reward' : 400,
            'tolerance' : 1e-3
    }
    save_config(example_dict, 'example.yaml')
    loaded_dict = load_config('example.yaml')