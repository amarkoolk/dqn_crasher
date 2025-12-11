import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader


def load_config(config_path):
    """Load a YAML configuration file."""
    with open(config_path, "r") as file:
        config = yaml.load(file, Loader=Loader)
    return config
