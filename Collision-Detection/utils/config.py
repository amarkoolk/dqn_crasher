import yaml
from yaml import dump, load

try:
    from yaml import CDumper as Dumper
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Dumper, Loader


def load_pkg_yaml(rel_path: str):
    """
    Load a YAML file from the configs directory. Use paths like:
      "configs/model/dqn_vs_scenarios.yaml"
      "configs/env/multi_agent.yaml"
    """
    with open(rel_path, "r") as f:
        return yaml.safe_load(f)


def load_config(config_path):
    with open(config_path, "r") as file:
        config = load(file, Loader=Loader)
    return config


def save_config(config, config_path):
    with open(config_path, "w") as file:
        dump(config, file, Dumper=Dumper)
