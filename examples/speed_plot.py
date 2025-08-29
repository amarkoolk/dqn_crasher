import pandas as pd
import wandb
import numpy as np
import matplotlib.pyplot as plt
from wandb.apis.public import Runs
import sys

if __name__ == "__main__":
    api = wandb.Api()
    training_runs: Runs = api.runs(
        "amar-research/safetyh", filters={"tags": "adj_bl_eval"}
    )

    run_history = []
    # Should be only one run
    for run in training_runs:
        config = run.config
        history = run.scan_history()
        run_history.append(
            (
                run.tags,
                run.history(
                    samples=history.max_step,
                    x_axis="_step",
                    pandas=(True),
                    stream="default",
                ),
            )
        )

    fig, ax = plt.subplots()
    ax.set_title("Ego Speed")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Average Speed (m/s)")
    all_histories = []

    for tag, history in run_history:
        if len(tag) != 3:
            continue
        history["smoothed_speed"] = (
            history["rollout/ego_speed_mean"].ewm(span=500).mean()
        )
        all_histories.append(history[["_step", "smoothed_speed"]])

    # Concatenate all histories into a single DataFrame
    combined_history = pd.concat(all_histories)

    # Group by step and calculate mean and standard deviation
    grouped_history = (
        combined_history.groupby("_step")
        .agg(mean_speed=("smoothed_speed", "mean"), std_speed=("smoothed_speed", "std"))
        .reset_index()
    )

    # Plot the mean speed with standard deviation as a shaded area
    ax.plot(
        grouped_history["_step"], grouped_history["mean_speed"], label="Average Speed"
    )
    ax.fill_between(
        grouped_history["_step"],
        grouped_history["mean_speed"] - grouped_history["std_speed"],
        grouped_history["mean_speed"] + grouped_history["std_speed"],
        color="b",
        alpha=0.2,
        label="Standard Deviation",
    )
    ax.legend()
    ax.grid()
    plt.show()
