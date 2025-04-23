import yaml, random

class DistributionScheduler:
    def __init__(self, entries):
        # entries is a list of dicts with keys: from, to, opponents
        self.entries = entries

    @classmethod
    def from_yaml(cls, path):
        data = yaml.safe_load(open(path))
        return cls(data["training_distribution"])

    def get_weights(self, episode_idx):
        # find the right entry for this episode
        for e in self.entries:
            start, end = e["from"], e["to"]
            if episode_idx >= start and (end is None or episode_idx <= end):
                return e["opponents"]
        raise RuntimeError(f"No curriculum entry for episode {episode_idx}")

    def sample_opponent(self, episode_idx, policy_map):
        """
        policy_map: dict name -> policy instance
        returns one policy sampled according to the current weights
        """
        weights_map = self.get_weights(episode_idx)
        names, weights = [], []
        for name, w in weights_map.items():
            if name not in policy_map:
                raise KeyError(f"No policy named {name}")
            names.append(name)
            weights.append(w)
        choice = random.choices(names, weights=weights, k=1)[0]
        return policy_map[choice]