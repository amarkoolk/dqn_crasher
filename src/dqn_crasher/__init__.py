import importlib
import sys

# Ensure subpackage exists and is importable
# (you added src/dqn_crasher/scenarios/__init__.py in Step 1)
_policies = importlib.import_module("dqn_crasher.scenarios.policies")
_scenarios = importlib.import_module("dqn_crasher.scenarios.scenarios")

# Create top-level module aliases
sys.modules.setdefault("policies", _policies)
sys.modules.setdefault("scenarios", _scenarios)
