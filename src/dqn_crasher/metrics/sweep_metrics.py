import pandas as pd
from wandb.apis.public import Api

# Initialize API and sweep
sweep = Api().sweep("amar-research/dqn_crasher-dqn_crasher/5bd9p8e4")
prefix = "checkpoint_summary"
scenarios = [
    "MobilPolicy.forward_right",
    "MobilPolicy.forward_left",
    "MobilPolicy.behind_right",
    "MobilPolicy.behind_left",
    "IdleSlower",
    "IdleFaster",
    "CutIn",
    "CutInSlowDown",
]
metrics = ["episode_rewards"]  # , 'num_crashes']

# Collect data for scenario-specific best
data = []
for scenario in scenarios:
    best_run = sweep.best_run(order=f"{prefix}/{scenario}/episode_rewards")
    config = best_run.config
    for metric in metrics:
        data.append(
            {
                "Scenario": scenario,
                "Metric": metric,
                "Best Run Name": best_run.name,
                "Best Value": best_run.summary.get(f"{prefix}/{scenario}/{metric}", 0),
                "Total": best_run.summary.get(f"{prefix}/{metric}", 0),
                "LR": config.get("lr", 0),
                "Hidden Layer Size": config.get("hidden_layer", 0),
                "Num Layers": config.get("num_hidden_layers", 0),
                "Batch Size": config.get("batch_size", 0),
            }
        )

# Collect data for best overall (per metric)

best_run_reward = sweep.best_run(order=f"{prefix}/episode_rewards")
config = best_run_reward.config
for metric in metrics:
    for scenario in scenarios:
        data.append(
            {
                "Scenario": scenario,
                "Metric": f"{metric}",
                "Best Run Name": best_run_reward.name,
                "Best Value": best_run_reward.summary.get(
                    f"{prefix}/{scenario}/{metric}", 0
                ),
                "Total": best_run_reward.summary.get(f"{prefix}/{metric}", 0),
                "LR": config.get("lr", 0),
                "Hidden Layer Size": config.get("hidden_layer", 0),
                "Num Layers": config.get("num_hidden_layers", 0),
                "Batch Size": config.get("batch_size", 0),
            }
        )


# Create DataFrame
df = pd.DataFrame(data)

# Print a nice table
print("\n=== Best Runs per Scenario and Metric ===\n")
print(df.to_string(index=False))

# Optional: Pretty HTML in Jupyter
# from IPython.display import display
# display(df)
