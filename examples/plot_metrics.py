import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from wandb.apis.public import Runs

import wandb

if __name__ == "__main__":
    api = wandb.Api()
    tag = sys.argv[1]
    eps = float(sys.argv[2])
    ego_version = int(sys.argv[3])
    npc_version = int(sys.argv[4])
    sampling_mode = int(sys.argv[5])
    if sampling_mode == 0:
        sampling = "uniform"
    elif sampling_mode == 1:
        sampling = "prioritized"
    elif sampling_mode == 2:
        sampling = "two_model"
    training_runs: Runs = api.runs(
        "amar-research/safetyh",
        filters={
            "tags": f"{tag}",
            "config.eps_start": eps,
            "config.ego_version": ego_version,
            "config.npc_version": npc_version,
        },
    )

    # Should be only one run
    for run in training_runs:
        config = run.config
        history = run.scan_history()
        run_history = run.history(
            samples=history.max_step, x_axis="_step", pandas=(True), stream="default"
        )

    train_ego = config["train_ego"]

    n_egos = ego_version if train_ego else ego_version + 1
    n_npcs = npc_version if not train_ego else npc_version + 1

    if sampling_mode == 2:
        n_opps = n_npcs + 1 if train_ego else n_egos
    else:
        n_opps = n_npcs if train_ego else n_egos

    run_history["smoothed_rew"] = (
        run_history["rollout/ep_rew_mean"].ewm(span=500).mean()
    )
    run_history["smoothed_sr100"] = run_history["rollout/sr100"].ewm(span=500).mean()

    reward_df = pd.DataFrame(
        run_history, columns=["_step", "rollout/ep_rew_mean", "smoothed_rew"]
    )
    crash_df = pd.DataFrame(
        run_history, columns=["_step", "rollout/sr100", "smoothed_sr100"]
    )

    elo_columns = ["_step", "rollout/opponent_elo"]
    sr_columns = ["_step"]
    freq_columns = ["_step"]
    for i in range(n_opps):
        elo_columns.append(f"rollout/model_{i}_elo")
        freq_columns.append(f"rollout/model_{i}_ep_freq")
        sr_columns.append(f"rollout/model_{i}_sr100")

    elo_df = pd.DataFrame(run_history, columns=elo_columns)
    sr_df = pd.DataFrame(run_history, columns=sr_columns)
    freq_df = pd.DataFrame(run_history, columns=freq_columns)

    model_name = "V" if train_ego else "E"
    freq_mapping = {
        f"rollout/model_{i}_ep_freq": f"{model_name}{i}_EpFreq-{elo_df[f'rollout/model_{i}_elo'].mean():.2f} ELO"
        for i in range(n_opps)
    }
    sr_mapping = {
        f"rollout/model_{i}_sr100": f"{model_name}{i}_SR-{elo_df[f'rollout/model_{i}_elo'].mean():.2f} ELO"
        for i in range(n_opps)
    }

    if sampling_mode == 2:
        if train_ego:
            freq_mapping = {
                f"rollout/model_{i}_ep_freq": f"V{i - 1}_EpFreq-{elo_df[f'rollout/model_{i}_elo'].mean():.2f} ELO"
                for i in range(n_opps)
            }
            sr_mapping = {
                f"rollout/model_{i}_sr100": f"V{i - 1}_SR-{elo_df[f'rollout/model_{i}_elo'].mean():.2f} ELO"
                for i in range(n_opps)
            }

            freq_mapping["rollout/model_0_ep_freq"] = (
                f"MOBIL_EpFreq-{elo_df[f'rollout/model_0_elo'].mean():.2f} ELO"
            )
            sr_mapping["rollout/model_0_sr100"] = (
                f"MOBIL_SR-{elo_df[f'rollout/model_0_elo'].mean():.2f} ELO"
            )

    sr_df = sr_df.rename(columns=sr_mapping)
    freq_df = freq_df.rename(columns=freq_mapping)

    opponent_elo = elo_df["rollout/opponent_elo"].mean()

    fig, ax = plt.subplots(2, 2, sharex=True)
    title_text = (
        f"{sampling} sampling - Behind Left - Training E{ego_version} vs V{npc_version} - Starting Eps {eps} - E{ego_version - 1} ELO {opponent_elo:.0f}"
        if train_ego
        else f"Behind Left - Training V{npc_version} vs E{ego_version} - Starting Eps {eps} - V{npc_version - 1} ELO {opponent_elo:.0f}"
    )
    fig.suptitle(title_text)
    ax[0, 0].set_title("Average Reward/Episode")
    ax[0, 1].set_title("Crash Rate")
    ax[1, 0].set_title("Crash Rate among Models")
    ax[1, 1].set_title("Model Episode Frequency")
    ax[0, 0].set_xlabel("Episode")
    ax[0, 1].set_xlabel("Episode")
    ax[0, 0].set_ylabel("Reward")
    ax[0, 1].set_ylabel("Crash Rate")
    reward_df.plot(x="_step", ax=ax[0, 0], grid=True)
    crash_df.plot(x="_step", ax=ax[0, 1], grid=True)
    sr_df.plot(x="_step", ax=ax[1, 0], grid=True)
    freq_df.plot(x="_step", ax=ax[1, 1], grid=True)

    plt.show()
