# Compatibility shim so strings like "scenarios.IdleSlower" keep working.
from dqn_crasher.scenarios.scenarios import CutIn, CutInSlowDown, IdleFaster, IdleSlower

__all__ = ["IdleFaster", "IdleSlower", "CutIn", "CutInSlowDown"]
