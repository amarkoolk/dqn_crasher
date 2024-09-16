import pandas as pd 
import wandb
import numpy as np
import matplotlib.pyplot as plt
from wandb.apis.public import Runs
import sys

if __name__ == "__main__":
    
    api = wandb.Api()
    training_runs : Runs = api.runs("amar-research/safetyh",
                    filters={"tags" : "07152024"}
                    )

    cycles = {}
    # Should be only one run
    for run in training_runs:
        config = run.config
        print(f'Ego Version: {config["ego_version"]}, NPC Version: {config["npc_version"]}')
        history = run.scan_history()
        run_history = run.history(samples=history.max_step, x_axis="_step", pandas=(True), stream="default")
        cycles[(config["ego_version"], config["npc_version"])] = run_history

    # Filter Dataframe to Keep only steps where 'rollout/spawn_config' is 'behind_left'
    # for cycle_key in cycles.keys():

    cycle_key=(1,0)

    use_mobil_filter = cycles[cycle_key]['rollout/use_mobil'] == True
    print(use_mobil_filter)

    fig, ax = plt.subplots(2,2)
    cycles[cycle_key].loc[cycles[cycle_key]['rollout/spawn_config'] == 'behind_left'].plot(ax=ax[0,0],x="_step", y="rollout/ego_speed_mean", grid=True, label = 'BL-EGO')
    cycles[cycle_key].loc[cycles[cycle_key]['rollout/spawn_config'] == 'forward_left'].plot(ax=ax[0,0],x="_step", y="rollout/ego_speed_mean", grid=True, label='FL-EGO')
    cycles[cycle_key].loc[cycles[cycle_key]['rollout/spawn_config'] == 'behind_left'].plot(ax=ax[1,0],x="_step", y="rollout/npc_speed_mean", grid=True, label = 'BL-NPC')
    cycles[cycle_key].loc[cycles[cycle_key]['rollout/spawn_config'] == 'forward_left'].plot(ax=ax[1,0],x="_step", y="rollout/npc_speed_mean", grid=True, label='FL-NPC')

    cycles[cycle_key].loc[cycles[cycle_key]['rollout/spawn_config'] == 'behind_left'].ewm(span=100).mean().plot(ax=ax[0,0],x="_step", y="rollout/ego_speed_mean", grid=True, label = 'BL-EGO-avg')
    cycles[cycle_key].loc[cycles[cycle_key]['rollout/spawn_config'] == 'forward_left'].ewm(span=100).mean().plot(ax=ax[0,0],x="_step", y="rollout/ego_speed_mean", grid=True, label='FL-EGO-avg')
    cycles[cycle_key].loc[cycles[cycle_key]['rollout/spawn_config'] == 'behind_left'].ewm(span=100).mean().plot(ax=ax[1,0],x="_step", y="rollout/npc_speed_mean", grid=True, label = 'BL-NPC-avg')
    cycles[cycle_key].loc[cycles[cycle_key]['rollout/spawn_config'] == 'forward_left'].ewm(span=100).mean().plot(ax=ax[1,0],x="_step", y="rollout/npc_speed_mean", grid=True, label='FL-NPC-avg')
    ax[0,0].legend()
    ax[1,0].legend()

    mobil_df = cycles[cycle_key].loc[cycles[cycle_key]['rollout/use_mobil'] == True]

    cycles[cycle_key].loc[cycles[cycle_key]['rollout/spawn_config'] == 'behind_left'].plot(ax=ax[0,1],x="_step", y="rollout/ego_speed_mean", grid=True, label = 'BL-MOBIL')
    cycles[cycle_key].loc[cycles[cycle_key]['rollout/spawn_config'] == 'forward_left'].plot(ax=ax[0,1],x="_step", y="rollout/ego_speed_mean", grid=True, label='FL-MOBIL')
    mobil_df.loc[mobil_df['rollout/spawn_config'] == 'behind_left'].plot(ax=ax[1,1],x="_step", y="rollout/npc_speed_mean", grid=True, label = 'BL-NPC')
    mobil_df.loc[mobil_df['rollout/spawn_config'] == 'forward_left'].plot(ax=ax[1,1],x="_step", y="rollout/npc_speed_mean", grid=True, label='FL-NPC')

    cycles[cycle_key].loc[cycles[cycle_key]['rollout/spawn_config'] == 'behind_left'].ewm(span=100).mean().plot(ax=ax[0,1],x="_step", y="rollout/ego_speed_mean", grid=True, label = 'BL-MOBIL-avg')
    cycles[cycle_key].loc[cycles[cycle_key]['rollout/spawn_config'] == 'forward_left'].ewm(span=100).mean().plot(ax=ax[0,1],x="_step", y="rollout/ego_speed_mean", grid=True, label='FL-MOBIL-avg')
    mobil_df.loc[mobil_df['rollout/spawn_config'] == 'behind_left'].ewm(span=100).mean().plot(ax=ax[1,1],x="_step", y="rollout/npc_speed_mean", grid=True, label = 'BL-NPC-avg')
    mobil_df.loc[mobil_df['rollout/spawn_config'] == 'forward_left'].ewm(span=100).mean().plot(ax=ax[1,1],x="_step", y="rollout/npc_speed_mean", grid=True, label='FL-NPC-avg')
    ax[1,0].legend()
    ax[1,1].legend()
    plt.show()
        
    bl_df = cycles[cycle_key].loc[cycles[cycle_key]['rollout/spawn_config'] == 'behind_left']
    bl_um_ego = bl_df.loc[bl_df['rollout/use_mobil'] == True]['rollout/ego_speed_mean']
    bl_num_ego = bl_df.loc[bl_df['rollout/use_mobil'] == False]['rollout/ego_speed_mean']

    fl_df = cycles[cycle_key].loc[cycles[cycle_key]['rollout/spawn_config'] == 'forward_left']
    fl_um_ego = fl_df.loc[fl_df['rollout/use_mobil'] == True]['rollout/ego_speed_mean']
    fl_num_ego = fl_df.loc[fl_df['rollout/use_mobil'] == False]['rollout/ego_speed_mean']

    bl_um_npc = bl_df.loc[bl_df['rollout/use_mobil'] == True]['rollout/npc_speed_mean']
    bl_num_npc = bl_df.loc[bl_df['rollout/use_mobil'] == False]['rollout/npc_speed_mean']

    fl_um_npc = fl_df.loc[fl_df['rollout/use_mobil'] == True]['rollout/npc_speed_mean']
    fl_num_npc = fl_df.loc[fl_df['rollout/use_mobil'] == False]['rollout/npc_speed_mean']

    bl_sr_um = bl_df['rollout/sr100']
    bl_sr_num = bl_df.loc[bl_df['rollout/use_mobil'] == False]['rollout/sr100']  
    
    fl_sr_um = fl_df.loc[fl_df['rollout/use_mobil'] == True]['rollout/sr100']
    fl_sr_num = fl_df.loc[fl_df['rollout/use_mobil'] == False]['rollout/sr100']



    fig, ax = plt.subplots(3,2, sharex=True)
    fig.suptitle(f"E{cycle_key[0]} vs V{cycle_key[1]} - E1 Behind Left vs Front Left - MOBIL vs No MOBIL")
    ax[0,0].set_title("Behind Left - MOBIL")
    ax[0,1].set_title("Behind Left - No MOBIL")
    ax[1,0].set_title("Front Left - MOBIL")
    ax[1,1].set_title("Front Left - No MOBIL")
    ax[2,0].set_title("Success Rate - MOBIL")
    ax[2,1].set_title("Success Rate - No MOBIL")

    bl_um_ego.ewm(span=100).mean().plot(x="_step",ax=ax[0,0], grid=True)
    bl_num_ego.ewm(span=100).mean().plot(x="_step",ax=ax[0,1], grid=True)
    fl_um_ego.ewm(span=100).mean().plot(x="_step",ax=ax[1,0], grid=True)
    fl_num_ego.ewm(span=100).mean().plot(x="_step",ax=ax[1,1], grid=True)

    bl_um_npc.ewm(span=100).mean().plot(x="_step",ax=ax[0,0], grid=True)
    bl_num_npc.ewm(span=100).mean().plot(x="_step",ax=ax[0,1], grid=True)
    fl_um_npc.ewm(span=100).mean().plot(x="_step",ax=ax[1,0], grid=True)
    fl_num_npc.ewm(span=100).mean().plot(x="_step",ax=ax[1,1], grid=True)

    bl_sr_um.ewm(span=100).mean().plot(x="_step",ax=ax[2,0], grid=True)
    bl_sr_num.ewm(span=100).mean().plot(x="_step",ax=ax[2,1], grid=True)
    fl_sr_um.ewm(span=100).mean().plot(x="_step",ax=ax[2,0], grid=True)
    fl_sr_num.ewm(span=100).mean().plot(x="_step",ax=ax[2,1], grid=True)

    ax[0,0].legend(["Ego Speed", "NPC Speed"])
    ax[0,1].legend(["Ego Speed", "NPC Speed"])
    ax[1,0].legend(["Ego Speed", "NPC Speed"])
    ax[1,1].legend(["Ego Speed", "NPC Speed"])
    ax[2,0].legend(["MOBIL", "No MOBIL"])
    ax[2,1].legend(["MOBIL", "No MOBIL"])

    ax[0,0].set_ylim([10, 35])
    ax[0,1].set_ylim([10, 35])
    ax[1,0].set_ylim([10, 35])
    ax[1,1].set_ylim([10, 35])

    ax[2,0].set_ylim([0, 1])
    ax[2,1].set_ylim([0, 1])
    plt.show()


    # train_ego = config["train_ego"]

    # n_egos = ego_version if train_ego else ego_version + 1
    # n_npcs = npc_version if not train_ego else npc_version + 1

    # if sampling_mode == 2:
    #     n_opps = n_npcs+1 if train_ego else n_egos
    # else:
    #     n_opps = n_npcs if train_ego else n_egos

    # run_history["smoothed_rew"] = run_history["rollout/ep_rew_mean"].ewm(span=500).mean()
    # run_history["smoothed_sr100"] = run_history["rollout/sr100"].ewm(span=500).mean()

    # reward_df = pd.DataFrame(run_history, columns=["_step", "rollout/ep_rew_mean", "smoothed_rew"])
    # crash_df = pd.DataFrame(run_history, columns=["_step", "rollout/sr100", "smoothed_sr100"])

    # elo_columns = ["_step","rollout/opponent_elo"]
    # sr_columns = ["_step"]
    # freq_columns = ["_step"]
    # for i in range(n_opps):
    #     elo_columns.append(f"rollout/model_{i}_elo")
    #     freq_columns.append(f"rollout/model_{i}_ep_freq")
    #     sr_columns.append(f"rollout/model_{i}_sr100")

    # elo_df = pd.DataFrame(run_history, columns=elo_columns)
    # sr_df = pd.DataFrame(run_history, columns=sr_columns)
    # freq_df = pd.DataFrame(run_history, columns=freq_columns)

    # model_name = "V" if train_ego else "E"
    # freq_mapping = {f"rollout/model_{i}_ep_freq": f"{model_name}{i}_EpFreq-{elo_df[f'rollout/model_{i}_elo'].mean():.2f} ELO" for i in range(n_opps)}
    # sr_mapping = {f"rollout/model_{i}_sr100": f"{model_name}{i}_SR-{elo_df[f'rollout/model_{i}_elo'].mean():.2f} ELO" for i in range(n_opps)}

    # if sampling_mode == 2:
    #     if train_ego:
    #         freq_mapping = {f"rollout/model_{i}_ep_freq": f"V{i-1}_EpFreq-{elo_df[f'rollout/model_{i}_elo'].mean():.2f} ELO" for i in range(n_opps)}
    #         sr_mapping = {f"rollout/model_{i}_sr100": f"V{i-1}_SR-{elo_df[f'rollout/model_{i}_elo'].mean():.2f} ELO" for i in range(n_opps)}

    #         freq_mapping["rollout/model_0_ep_freq"] = f"MOBIL_EpFreq-{elo_df[f'rollout/model_0_elo'].mean():.2f} ELO"
    #         sr_mapping["rollout/model_0_sr100"] = f"MOBIL_SR-{elo_df[f'rollout/model_0_elo'].mean():.2f} ELO"

    # sr_df = sr_df.rename(columns=sr_mapping)
    # freq_df = freq_df.rename(columns=freq_mapping)

    # opponent_elo = elo_df["rollout/opponent_elo"].mean()

    # fig, ax = plt.subplots(2,2, sharex=True)
    # title_text = f"{sampling} sampling - Behind Left - Training E{ego_version} vs V{npc_version} - Starting Eps {eps} - E{ego_version-1} ELO {opponent_elo:.0f}" if train_ego else \
    #       f"Behind Left - Training V{npc_version} vs E{ego_version} - Starting Eps {eps} - V{npc_version-1} ELO {opponent_elo:.0f}"
    # fig.suptitle(title_text)
    # ax[0,0].set_title("Average Reward/Episode")
    # ax[0,1].set_title("Crash Rate")
    # ax[1,0].set_title("Crash Rate among Models")
    # ax[1,1].set_title("Model Episode Frequency")
    # ax[0,0].set_xlabel("Episode")
    # ax[0,1].set_xlabel("Episode")
    # ax[0,0].set_ylabel("Reward")
    # ax[0,1].set_ylabel("Crash Rate")
    # reward_df.plot(x="_step",ax=ax[0,0], grid=True)
    # crash_df.plot(x="_step",ax=ax[0,1], grid=True)
    # sr_df.plot(x="_step",ax=ax[1,0], grid=True)
    # freq_df.plot(x="_step",ax=ax[1,1], grid=True)

    # plt.show()