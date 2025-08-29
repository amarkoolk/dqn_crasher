# Compatibility shim so strings like "policies.DQNPolicy" keep working.
from dqn_crasher.scenarios.policies import (BasePolicy, DQNPolicy, MobilPolicy,
                                            PolicyDistribution, ScenarioPolicy)

__all__ = [
    "BasePolicy",
    "DQNPolicy",
    "MobilPolicy",
    "PolicyDistribution",
    "ScenarioPolicy",
]
