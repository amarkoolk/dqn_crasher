import pandas as pd 
import wandb
import numpy as np
import matplotlib.pyplot as plt
from wandb.apis.public import Runs
import sys

if __name__ == "__main__":
    
    api = wandb.Api()
    training_runs : Runs = api.runs("amar-research/safetyh",
                    filters={"tags" : "adj_bl_eval"}
                    )

    run_history = []
    # Should be only one run
    for run in training_runs:
        config = run.config
        history = run.scan_history()
        run_history.append((run.tags,run.history(samples=history.max_step, x_axis="_step", pandas=(True), stream="default")))

    fig, ax = plt.subplots()
    ax.set_title("Ego Speed")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Average Speed (m/s)")
    for tag, history in run_history:
        if(len(tag) != 3):
            continue
        history["smoothed_speed"] = history["rollout/ego_speed_mean"].ewm(span=500).mean()
        history.plot(x="_step",y="smoothed_speed", grid=True, ax=ax, label=f"{tag[1]}, {tag[2]}")

    plt.show()