import pandas as pd 
import wandb
import numpy as np
import matplotlib.pyplot as plt
from wandb.apis.public import Runs
import sys

if __name__ == "__main__":
    
    api = wandb.Api()
    training_runs : Runs = api.runs("amar-research/safetyh",
                    filters={"tags" : "additional_state"}
                    )

    run_history = {}
    # Should be only one run
    for run in training_runs:
        config = run.config
        history = run.scan_history()
        run_history[str(run.name[-4:])] = run.history(samples=history.max_step, x_axis="_step", pandas=(True), stream="default")

    state10_mobil_speed_df = pd.DataFrame(run_history["3121"], columns=["_step", "rollout/model_0_speed"])
    state10_npc_speed_df = pd.DataFrame(run_history["3121"], columns=["_step", "rollout/model_1_speed"])
    state10_mobil_sr_df = pd.DataFrame(run_history["3121"], columns=["_step", "rollout/model_0_sr100"])
    state10_npc_sr_df = pd.DataFrame(run_history["3121"], columns=["_step", "rollout/model_1_sr100"])

    state11_mobil_speed_df = pd.DataFrame(run_history["3122"], columns=["_step", "rollout/model_0_speed"])
    state11_npc_speed_df = pd.DataFrame(run_history["3122"], columns=["_step", "rollout/model_1_speed"])
    state11_mobil_sr_df = pd.DataFrame(run_history["3122"], columns=["_step", "rollout/model_0_sr100"])
    state11_npc_sr_df = pd.DataFrame(run_history["3122"], columns=["_step", "rollout/model_1_sr100"])

    state10_mobil_speed_df['10 State'] = pd.to_numeric(state10_mobil_speed_df['rollout/model_0_speed'], errors='coerce')
    state10_mobil_speed_df.fillna(0, inplace=True)
    state11_mobil_speed_df['11 State'] = pd.to_numeric(state11_mobil_speed_df['rollout/model_0_speed'], errors='coerce')
    state11_mobil_speed_df.fillna(0, inplace=True)

    state10_npc_speed_df['10 State'] = pd.to_numeric(state10_npc_speed_df['rollout/model_1_speed'], errors='coerce')
    state10_npc_speed_df.fillna(0, inplace=True)
    state11_npc_speed_df['11 State'] = pd.to_numeric(state11_npc_speed_df['rollout/model_1_speed'], errors='coerce')
    state11_npc_speed_df.fillna(0, inplace=True)

    state10_mobil_sr_df['10 State'] = pd.to_numeric(state10_mobil_sr_df['rollout/model_0_sr100'], errors='coerce')
    state10_mobil_sr_df.fillna(0, inplace=True)
    state11_mobil_sr_df['11 State'] = pd.to_numeric(state11_mobil_sr_df['rollout/model_0_sr100'], errors='coerce')
    state11_mobil_sr_df.fillna(0, inplace=True)

    state10_npc_sr_df['10 State'] = pd.to_numeric(state10_npc_sr_df['rollout/model_1_sr100'], errors='coerce')
    state10_npc_sr_df.fillna(0, inplace=True)
    state11_npc_sr_df['11 State'] = pd.to_numeric(state11_npc_sr_df['rollout/model_1_sr100'], errors='coerce')
    state11_npc_sr_df.fillna(0, inplace=True)



    fig, ax = plt.subplots(2,2, sharex=True)
    fig.suptitle("Speed and Success Rate - Behind Left")
    ax[0,0].set_title("Ego Speed - Mobil")
    ax[0,1].set_title("Ego Speed - NPC")
    ax[1,0].set_title("Crash Rate - Mobil")
    ax[1,1].set_title("Crash Rate - NPC")
    plt.xlabel("Episode")
    ax[0,0].set_ylabel("Speed (m/s)")
    ax[0,1].set_ylabel("Speed (m/s)")
    ax[1,0].set_ylabel("Crash Rate")
    ax[1,1].set_ylabel("Crash Rate")
    state10_mobil_speed_df.plot(x="_step",y='10 State',ax=ax[0,0], grid=True)
    state11_mobil_speed_df.plot(x="_step",y='11 State',ax=ax[0,0], grid=True, color='red')
    state10_npc_speed_df.plot(x="_step",y='10 State',ax=ax[0,1], grid=True)
    state11_npc_speed_df.plot(x="_step",y='11 State',ax=ax[0,1], grid=True, color='red')
    state10_mobil_sr_df.plot(x="_step",y='10 State',ax=ax[1,0], grid=True)
    state11_mobil_sr_df.plot(x="_step",y='11 State',ax=ax[1,0], grid=True, color='red')
    state10_npc_sr_df.plot(x="_step",y='10 State',ax=ax[1,1], grid=True)
    state11_npc_sr_df.plot(x="_step",y='11 State',ax=ax[1,1], grid=True, color='red')

    ax[0,0].set_ylim(0, 35)
    ax[0,1].set_ylim(0, 35)
    ax[1,0].set_ylim(0, 1)
    ax[1,1].set_ylim(0, 1)
    plt.show()

    state11_mobil_speed_df['Behind Left'] = state11_mobil_speed_df['11 State']
    state11_npc_speed_df['Behind Left'] = state11_npc_speed_df['11 State']
    state11_mobil_sr_df['Behind Left'] = state11_mobil_sr_df['11 State']
    state11_npc_sr_df['Behind Left'] = state11_npc_sr_df['11 State']

    state11_mobil_speed_df_fl = pd.DataFrame(run_history["3123"], columns=["_step", "rollout/model_0_speed"])
    state11_npc_speed_df_fl = pd.DataFrame(run_history["3123"], columns=["_step", "rollout/model_1_speed"])
    state11_mobil_sr_df_fl = pd.DataFrame(run_history["3123"], columns=["_step", "rollout/model_0_sr100"])
    state11_npc_sr_df_fl = pd.DataFrame(run_history["3123"], columns=["_step", "rollout/model_1_sr100"])

    state11_mobil_speed_df_fl['Forward Left'] = pd.to_numeric(state11_mobil_speed_df_fl['rollout/model_0_speed'], errors='coerce')
    state11_mobil_speed_df_fl.fillna(0, inplace=True)
    state11_npc_speed_df_fl['Forward Left'] = pd.to_numeric(state11_npc_speed_df_fl['rollout/model_1_speed'], errors='coerce')
    state11_npc_speed_df_fl.fillna(0, inplace=True)
    state11_mobil_sr_df_fl['Forward Left'] = pd.to_numeric(state11_mobil_sr_df_fl['rollout/model_0_sr100'], errors='coerce')
    state11_mobil_sr_df_fl.fillna(0, inplace=True)
    state11_npc_sr_df_fl['Forward Left'] = pd.to_numeric(state11_npc_sr_df_fl['rollout/model_1_sr100'], errors='coerce')
    state11_npc_sr_df_fl.fillna(0, inplace=True)

    state11_mobil_speed_df_fb = pd.DataFrame(run_history["3124"], columns=["_step", "rollout/model_0_speed"])
    state11_npc_speed_df_fb = pd.DataFrame(run_history["3124"], columns=["_step", "rollout/model_1_speed"])
    state11_mobil_sr_df_fb = pd.DataFrame(run_history["3124"], columns=["_step", "rollout/model_0_sr100"])
    state11_npc_sr_df_fb = pd.DataFrame(run_history["3124"], columns=["_step", "rollout/model_1_sr100"])

    state11_mobil_speed_df_fb['Forward/Back Left'] = pd.to_numeric(state11_mobil_speed_df_fb['rollout/model_0_speed'], errors='coerce')
    state11_mobil_speed_df_fb.fillna(0, inplace=True)
    state11_npc_speed_df_fb['Forward/Back Left'] = pd.to_numeric(state11_npc_speed_df_fb['rollout/model_1_speed'], errors='coerce')
    state11_npc_speed_df_fb.fillna(0, inplace=True)
    state11_mobil_sr_df_fb['Forward/Back Left'] = pd.to_numeric(state11_mobil_sr_df_fb['rollout/model_0_sr100'], errors='coerce')
    state11_mobil_sr_df_fb.fillna(0, inplace=True)
    state11_npc_sr_df_fb['Forward/Back Left'] = pd.to_numeric(state11_npc_sr_df_fb['rollout/model_1_sr100'], errors='coerce')
    state11_npc_sr_df_fb.fillna(0, inplace=True)

    fig, ax = plt.subplots(2,2, sharex=True)
    fig.suptitle("Speed and Success Rate - Behind Left")
    ax[0,0].set_title("Ego Speed - Mobil")
    ax[0,1].set_title("Ego Speed - NPC")
    ax[1,0].set_title("Crash Rate - Mobil")
    ax[1,1].set_title("Crash Rate - NPC")
    plt.xlabel("Episode")
    ax[0,0].set_ylabel("Speed (m/s)")
    ax[0,1].set_ylabel("Speed (m/s)")
    ax[1,0].set_ylabel("Crash Rate")
    ax[1,1].set_ylabel("Crash Rate")
    state11_mobil_speed_df_fl.plot(x="_step",y='Forward Left',ax=ax[0,0], grid=True)
    state11_npc_speed_df_fl.plot(x="_step",y='Forward Left',ax=ax[0,1], grid=True)
    state11_mobil_sr_df_fl.plot(x="_step",y='Forward Left',ax=ax[1,0], grid=True)
    state11_npc_sr_df_fl.plot(x="_step",y='Forward Left',ax=ax[1,1], grid=True)

    ax[0,0].set_ylim(0, 35)
    ax[0,1].set_ylim(0, 35)
    ax[1,0].set_ylim(0, 1)
    ax[1,1].set_ylim(0, 1)
    plt.show()

    fig, ax = plt.subplots(2,2, sharex=True)
    fig.suptitle("Speed and Success Rate - Behind Left")
    ax[0,0].set_title("Ego Speed - Mobil")
    ax[0,1].set_title("Ego Speed - NPC")
    ax[1,0].set_title("Crash Rate - Mobil")
    ax[1,1].set_title("Crash Rate - NPC")
    plt.xlabel("Episode")
    ax[0,0].set_ylabel("Speed (m/s)")
    ax[0,1].set_ylabel("Speed (m/s)")
    ax[1,0].set_ylabel("Crash Rate")
    ax[1,1].set_ylabel("Crash Rate")
    state11_mobil_speed_df.plot(x="_step",y='Behind Left',ax=ax[0,0], grid=True)
    state11_mobil_speed_df_fl.plot(x="_step",y='Forward Left',ax=ax[0,0], grid=True)
    state11_mobil_speed_df_fb.plot(x="_step",y='Forward/Back Left',ax=ax[0,0], grid=True)

    state11_npc_speed_df.plot(x="_step",y='Behind Left',ax=ax[0,1], grid=True)
    state11_npc_speed_df_fl.plot(x="_step",y='Forward Left',ax=ax[0,1], grid=True)
    state11_npc_speed_df_fb.plot(x="_step",y='Forward/Back Left',ax=ax[0,1], grid=True)

    state11_mobil_sr_df.plot(x="_step",y='Behind Left',ax=ax[1,0], grid=True)
    state11_mobil_sr_df_fl.plot(x="_step",y='Forward Left',ax=ax[1,0], grid=True)
    state11_mobil_sr_df_fb.plot(x="_step",y='Forward/Back Left',ax=ax[1,0], grid=True)

    state11_npc_sr_df.plot(x="_step",y='Behind Left',ax=ax[1,1], grid=True)
    state11_npc_sr_df_fl.plot(x="_step",y='Forward Left',ax=ax[1,1], grid=True)
    state11_npc_sr_df_fb.plot(x="_step",y='Forward/Back Left',ax=ax[1,1], grid=True)

    ax[0,0].set_ylim(0, 35)
    ax[0,1].set_ylim(0, 35)
    ax[1,0].set_ylim(0, 1)
    ax[1,1].set_ylim(0, 1)
    plt.show()