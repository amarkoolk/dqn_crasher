import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from wandb.apis.public import Runs

import wandb

figure_dir = "../figures"
fig_dir = os.path.join(os.getcwd(), figure_dir)
results_dir = "results"
res_dir = os.path.join(os.getcwd(), results_dir)

tick_fontsize = 14
label_fontsize = 14
legend_fontsize = 14
title_fontsize = 16

if __name__ == "__main__":
    api = wandb.Api()
    training_runs: Runs = api.runs(
        "amar-research/safetyh", filters={"tags": "additional_state"}
    )

    run_history = {}
    # Should be only one run
    for run in training_runs:
        config = run.config
        history = run.scan_history()
        run_history[str(run.name[-4:])] = run.history(
            samples=history.max_step, x_axis="_step", pandas=(True), stream="default"
        )

    

    state10_mobil_speed_df = pd.DataFrame(
        run_history["3121"], columns=["_step", "rollout/model_0_speed"]
    )
    state10_npc_speed_df = pd.DataFrame(
        run_history["3121"], columns=["_step", "rollout/model_1_speed"]
    )
    state10_mobil_sr_df = pd.DataFrame(
        run_history["3121"], columns=["_step", "rollout/model_0_sr100"]
    )
    state10_npc_sr_df = pd.DataFrame(
        run_history["3121"], columns=["_step", "rollout/model_1_sr100"]
    )

    state11_mobil_speed_df = pd.DataFrame(
        run_history["3122"], columns=["_step", "rollout/model_0_speed"]
    )
    state11_npc_speed_df = pd.DataFrame(
        run_history["3122"], columns=["_step", "rollout/model_1_speed"]
    )
    state11_mobil_sr_df = pd.DataFrame(
        run_history["3122"], columns=["_step", "rollout/model_0_sr100"]
    )
    state11_npc_sr_df = pd.DataFrame(
        run_history["3122"], columns=["_step", "rollout/model_1_sr100"]
    )

    state10_mobil_speed_df["Original State"] = pd.to_numeric(
        state10_mobil_speed_df["rollout/model_0_speed"], errors="coerce"
    )
    state10_mobil_speed_df.fillna(0, inplace=True)
    state11_mobil_speed_df["Augmented State"] = pd.to_numeric(
        state11_mobil_speed_df["rollout/model_0_speed"], errors="coerce"
    )
    state11_mobil_speed_df.fillna(0, inplace=True)

    state10_npc_speed_df["Original State"] = pd.to_numeric(
        state10_npc_speed_df["rollout/model_1_speed"], errors="coerce"
    )
    state10_npc_speed_df.fillna(0, inplace=True)
    state11_npc_speed_df["Augmented State"] = pd.to_numeric(
        state11_npc_speed_df["rollout/model_1_speed"], errors="coerce"
    )
    state11_npc_speed_df.fillna(0, inplace=True)

    state10_mobil_sr_df["Original State"] = pd.to_numeric(
        state10_mobil_sr_df["rollout/model_0_sr100"], errors="coerce"
    )
    state10_mobil_sr_df.fillna(0, inplace=True)
    state11_mobil_sr_df["Augmented State"] = pd.to_numeric(
        state11_mobil_sr_df["rollout/model_0_sr100"], errors="coerce"
    )
    state11_mobil_sr_df.fillna(0, inplace=True)

    state10_npc_sr_df["Original State"] = pd.to_numeric(
        state10_npc_sr_df["rollout/model_1_sr100"], errors="coerce"
    )
    state10_npc_sr_df.fillna(0, inplace=True)
    state11_npc_sr_df["Augmented State"] = pd.to_numeric(
        state11_npc_sr_df["rollout/model_1_sr100"], errors="coerce"
    )
    state11_npc_sr_df.fillna(0, inplace=True)

    fig, ax = plt.subplots(2, 2, sharex=True, figsize=(16, 8))

    fig.suptitle(
        "Speed of Ego Vehicle and Crash Rate vs. MOBIL and NPC", fontsize=title_fontsize
    )
    ax[0, 0].set_title("Ego Speed - Mobil", fontsize=title_fontsize)
    ax[0, 1].set_title("Ego Speed - NPC", fontsize=title_fontsize)
    ax[1, 0].set_title("Crash Rate - Mobil", fontsize=title_fontsize)
    ax[1, 1].set_title("Crash Rate - NPC", fontsize=title_fontsize)
    plt.xlabel("Episode", fontsize=label_fontsize)
    ax[0, 0].set_ylabel("Speed (m/s)", fontsize=label_fontsize)
    # ax[0,1].set_ylabel("Speed (m/s)", fontsize=label_fontsize)
    ax[1, 0].set_ylabel("Crash Rate", fontsize=label_fontsize)
    # ax[1,1].set_ylabel("Crash Rate", fontsize=label_fontsize)
    state10_mobil_speed_df.plot(x="_step", y="Original State", ax=ax[0, 0], grid=True)
    state11_mobil_speed_df.plot(
        x="_step", y="Augmented State", ax=ax[0, 0], grid=True, color="red"
    )
    state10_npc_speed_df.plot(x="_step", y="Original State", ax=ax[0, 1], grid=True)
    state11_npc_speed_df.plot(
        x="_step", y="Augmented State", ax=ax[0, 1], grid=True, color="red"
    )
    state10_mobil_sr_df.plot(x="_step", y="Original State", ax=ax[1, 0], grid=True)
    state11_mobil_sr_df.plot(
        x="_step", y="Augmented State", ax=ax[1, 0], grid=True, color="red"
    )
    state10_npc_sr_df.plot(x="_step", y="Original State", ax=ax[1, 1], grid=True)
    state11_npc_sr_df.plot(
        x="_step", y="Augmented State", ax=ax[1, 1], grid=True, color="red"
    )

    ax[1, 0].set_xlabel("Episode", fontsize=label_fontsize)
    ax[1, 1].set_xlabel("Episode", fontsize=label_fontsize)
    ax[0, 0].set_ylim(10, 35)
    ax[0, 1].set_ylim(10, 35)
    ax[1, 0].set_ylim(0, 1)
    ax[1, 1].set_ylim(0, 1)
    plt.savefig(os.path.join(fig_dir, "state10_state11_speed_sr.pdf"))

    state11_mobil_speed_df["Behind Left"] = state11_mobil_speed_df["Augmented State"]
    state11_npc_speed_df["Behind Left"] = state11_npc_speed_df["Augmented State"]
    state11_mobil_sr_df["Behind Left"] = state11_mobil_sr_df["Augmented State"]
    state11_npc_sr_df["Behind Left"] = state11_npc_sr_df["Augmented State"]

    state11_mobil_speed_df_fl = pd.DataFrame(
        run_history["3123"], columns=["_step", "rollout/model_0_speed"]
    )
    state11_npc_speed_df_fl = pd.DataFrame(
        run_history["3123"], columns=["_step", "rollout/model_1_speed"]
    )
    state11_mobil_sr_df_fl = pd.DataFrame(
        run_history["3123"], columns=["_step", "rollout/model_0_sr100"]
    )
    state11_npc_sr_df_fl = pd.DataFrame(
        run_history["3123"], columns=["_step", "rollout/model_1_sr100"]
    )

    state11_mobil_speed_df_fl["Forward Left"] = pd.to_numeric(
        state11_mobil_speed_df_fl["rollout/model_0_speed"], errors="coerce"
    )
    state11_mobil_speed_df_fl.fillna(0, inplace=True)
    state11_npc_speed_df_fl["Forward Left"] = pd.to_numeric(
        state11_npc_speed_df_fl["rollout/model_1_speed"], errors="coerce"
    )
    state11_npc_speed_df_fl.fillna(0, inplace=True)
    state11_mobil_sr_df_fl["Forward Left"] = pd.to_numeric(
        state11_mobil_sr_df_fl["rollout/model_0_sr100"], errors="coerce"
    )
    state11_mobil_sr_df_fl.fillna(0, inplace=True)
    state11_npc_sr_df_fl["Forward Left"] = pd.to_numeric(
        state11_npc_sr_df_fl["rollout/model_1_sr100"], errors="coerce"
    )
    state11_npc_sr_df_fl.fillna(0, inplace=True)

    state11_mobil_speed_df_fb = pd.DataFrame(
        run_history["3124"],
        columns=["_step", "rollout/model_0_speed", "rollout/spawn_config"],
    )
    state11_npc_speed_df_fb = pd.DataFrame(
        run_history["3124"],
        columns=["_step", "rollout/model_1_speed", "rollout/spawn_config"],
    )
    state11_mobil_sr_df_fb = pd.DataFrame(
        run_history["3124"],
        columns=["_step", "rollout/model_0_sr100", "rollout/spawn_config"],
    )
    state11_npc_sr_df_fb = pd.DataFrame(
        run_history["3124"],
        columns=["_step", "rollout/model_1_sr100", "rollout/spawn_config"],
    )

    state11_mobil_speed_df_fb["Forward/Back Left"] = pd.to_numeric(
        state11_mobil_speed_df_fb["rollout/model_0_speed"], errors="coerce"
    )
    state11_mobil_speed_df_fb.fillna(0, inplace=True)
    state11_npc_speed_df_fb["Forward/Back Left"] = pd.to_numeric(
        state11_npc_speed_df_fb["rollout/model_1_speed"], errors="coerce"
    )
    state11_npc_speed_df_fb.fillna(0, inplace=True)
    state11_mobil_sr_df_fb["Forward/Back Left"] = pd.to_numeric(
        state11_mobil_sr_df_fb["rollout/model_0_sr100"], errors="coerce"
    )
    state11_mobil_sr_df_fb.fillna(0, inplace=True)
    state11_npc_sr_df_fb["Forward/Back Left"] = pd.to_numeric(
        state11_npc_sr_df_fb["rollout/model_1_sr100"], errors="coerce"
    )
    state11_npc_sr_df_fb.fillna(0, inplace=True)

    fig, ax = plt.subplots(2, 2, sharex=True)
    fig.suptitle(
        "Speed of Ego Vehicle and Crash Rate vs. MOBIL and NPC", fontsize=title_fontsize
    )
    ax[0, 0].set_title("Ego Speed - Mobil")
    ax[0, 1].set_title("Ego Speed - NPC")
    ax[1, 0].set_title("Crash Rate - Mobil")
    ax[1, 1].set_title("Crash Rate - NPC")
    plt.xlabel("Episode")
    ax[0, 0].set_ylabel("Speed (m/s)")
    ax[1, 0].set_ylabel("Crash Rate")
    state11_mobil_speed_df_fl.plot(x="_step", y="Forward Left", ax=ax[0, 0], grid=True)
    state11_npc_speed_df_fl.plot(x="_step", y="Forward Left", ax=ax[0, 1], grid=True)
    state11_mobil_sr_df_fl.plot(x="_step", y="Forward Left", ax=ax[1, 0], grid=True)
    state11_npc_sr_df_fl.plot(x="_step", y="Forward Left", ax=ax[1, 1], grid=True)

    ax[0, 0].set_ylim(0, 35)
    ax[0, 1].set_ylim(0, 35)
    ax[1, 0].set_ylim(0, 1)
    ax[1, 1].set_ylim(0, 1)
    plt.savefig(os.path.join(fig_dir, "state11_speed_sr_fl.pdf"))
    plt.show()

    fb_mobil_pivoted = state11_mobil_speed_df_fb.pivot(
        index="_step", columns="rollout/spawn_config", values="rollout/model_0_speed"
    )
    fb_npc_pivoted = state11_npc_speed_df_fb.pivot(
        index="_step", columns="rollout/spawn_config", values="rollout/model_1_speed"
    )
    print(fb_npc_pivoted)

    fig, ax = plt.subplots(2, 2, sharex=True)
    fig.suptitle("Speed and Success Rate - Behind Left")
    ax[0, 0].set_title("Ego Speed - Mobil")
    ax[0, 1].set_title("Ego Speed - NPC")
    ax[1, 0].set_title("Crash Rate - Mobil")
    ax[1, 1].set_title("Crash Rate - NPC")
    plt.xlabel("Episode")
    ax[0, 0].set_ylabel("Speed (m/s)")
    ax[0, 1].set_ylabel("Speed (m/s)")
    ax[1, 0].set_ylabel("Crash Rate")
    ax[1, 1].set_ylabel("Crash Rate")
    state11_mobil_speed_df.plot(x="_step", y="Behind Left", ax=ax[0, 0], grid=True)
    state11_mobil_speed_df_fl.plot(x="_step", y="Forward Left", ax=ax[0, 0], grid=True)
    fb_mobil_pivoted.plot(ax=ax[0, 0], grid=True)

    # state11_npc_speed_df.plot(x="_step",y='Behind Left',ax=ax[0,1], grid=True)
    # state11_npc_speed_df_fl.plot(x="_step",y='Forward Left',ax=ax[0,1], grid=True)
    ax[0, 1].plot(fb_npc_pivoted["behind_left"].dropna().to_numpy()[1:])
    ax[0, 1].plot(fb_npc_pivoted["forward_left"].dropna().to_numpy()[1:])

    state11_mobil_sr_df.plot(x="_step", y="Behind Left", ax=ax[1, 0], grid=True)
    state11_mobil_sr_df_fl.plot(x="_step", y="Forward Left", ax=ax[1, 0], grid=True)
    state11_mobil_sr_df_fb.plot(
        x="_step", y="Forward/Back Left", ax=ax[1, 0], grid=True
    )

    state11_npc_sr_df.plot(x="_step", y="Behind Left", ax=ax[1, 1], grid=True)
    state11_npc_sr_df_fl.plot(x="_step", y="Forward Left", ax=ax[1, 1], grid=True)
    state11_npc_sr_df_fb.plot(x="_step", y="Forward/Back Left", ax=ax[1, 1], grid=True)

    ax[0, 0].set_ylim(0, 35)
    ax[0, 1].set_ylim(0, 35)
    ax[1, 0].set_ylim(0, 1)
    ax[1, 1].set_ylim(0, 1)
    plt.show()
