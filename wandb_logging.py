import wandb

def initialize_logging(config, train_ego = None, eval = False, npc_pool_size = None, ego_pool_size = None):
    if wandb.run is not None:
        wandb.finish()

    gym_config = config.get("gym_config", {})

    run_config ={
                "learning_rate": config.get('learning_rate', 5e-4),
                "architecture": config.get('architecture', 'DQN'),
                "dataset": "Highway-Env",
                "max_steps": config.get('total_timesteps', 100000),
                "collision_reward": gym_config.get('collision_reward', 400),
                "ttc_x_reward": gym_config.get('ttc_x_reward', 4),
                "ttc_y_reward": gym_config.get('ttc_y_reward', 1),
                "batch_size": config.get('batch_size', 32),
                "gamma": config.get('gamma', 0.8),
                "eps_start": config.get('start_e', 1.0),
                "eps_end": config.get('end_e', 0.05),
                "eps_decay": config.get('decay_e', 6000),
                "tau": config.get('tau', 0.005),
                "ReplayBuffer": config.get('buffer_type', 'ER'),
                "eval": eval
    }

    run_config['ego_version'] = config.get('ego_version', 0)
    run_config['npc_version'] = config.get('npc_version', 0)
    if train_ego is not None:
        run_config['train_ego'] = train_ego
    if npc_pool_size is not None:
        run_config['npc_pool'] = True
        run_config['npc_pool_size'] = npc_pool_size
    else:
        run_config['npc_pool'] = False
    if ego_pool_size is not None:
        run_config['ego_pool'] = True
        run_config['ego_pool_size'] = ego_pool_size
    else:
        run_config['ego_pool'] = False

    if config.get('wandb_tag', None) is not None:
        tags = config.get('wandb_tag').split(',')
    else:
        tags = []
    
        
    run = wandb.init(
            # set the wandb project where this run will be logged
            project=config.get('wandb_project_name', 'safetyh'),
            
            # track hyperparameters and run metadata
            config=run_config,

            tags=tags
        )
    return run

def log_evaluation(args, n_cycle):
    if wandb.run is not None:
        wandb.finish()


    run_config ={
        "cycle": n_cycle
    }
        
    run = wandb.init(
            # set the wandb project where this run will be logged
            project=args.wandb_project_name,
            
            # track hyperparameters and run metadata
            config=run_config
        )
    return run

def log_stats(info, episode_statistics: dict, ego: bool):

    spawn_config = info['spawn_config']
    if(spawn_config not in episode_statistics['num_crashes']):
        episode_statistics['num_crashes'][spawn_config] = []
    episode_statistics['num_crashes'][spawn_config].append(float(info['crashed']))
    episode_statistics['total_crashes'].append(float(info['crashed']))
    episode_statistics['ep_rew_total'].append(episode_statistics['episode_rewards'])
    episode_statistics['ep_len_total'].append(episode_statistics['episode_duration'])
    episode_statistics['ego_speed_total'].append(episode_statistics['ego_speed']/episode_statistics['episode_duration'])
    episode_statistics['npc_speed_total'].append(episode_statistics['npc_speed']/episode_statistics['episode_duration'])

    wandb_log = {
        "rollout/ep_rew_mean": sum(episode_statistics['ep_rew_total'])/len(episode_statistics['ep_rew_total']),
        "rollout/ep_len_mean": sum(episode_statistics['ep_len_total'])/len(episode_statistics['ep_len_total']),
        "rollout/num_crashes": sum(episode_statistics['total_crashes']),
        "rollout/sr100": sum(episode_statistics['total_crashes'][-100:])/100,
        "rollout/ego_speed_mean": episode_statistics['ego_speed']/episode_statistics['episode_duration'],
        "rollout/npc_speed_mean": episode_statistics['npc_speed']/episode_statistics['episode_duration'],
        "rollout/spawn_config": spawn_config,
        "rollout/epsilon": episode_statistics['epsilon'],
        "rollout/collision_reward": episode_statistics['collision_reward']/episode_statistics['episode_duration'],
        f"rollout/{spawn_config}/ego_speed_mean": episode_statistics['ego_speed']/episode_statistics['episode_duration'],
        f"rollout/{spawn_config}/npc_speed_mean": episode_statistics['npc_speed']/episode_statistics['episode_duration'],
        f"rollout/{spawn_config}/num_crashes": sum(episode_statistics['num_crashes'][spawn_config]),
        f"rollout/{spawn_config}/sr100": sum(episode_statistics['num_crashes'][spawn_config][-100:])/100,
        f"rollout/{spawn_config}/collision_reward": episode_statistics['collision_reward']/episode_statistics['episode_duration'],
    }

    if ego:
        wandb_log.update({
                    "rollout/right_lane_reward": episode_statistics['right_lane_reward']/episode_statistics['episode_duration'],
                    "rollout/high_speed_reward": episode_statistics['high_speed_reward']/episode_statistics['episode_duration'],
                    f"rollout/{spawn_config}/right_lane_reward": episode_statistics['right_lane_reward']/episode_statistics['episode_duration'],
                    f"rollout/{spawn_config}/high_speed_reward": episode_statistics['high_speed_reward']/episode_statistics['episode_duration'],
        })
    else:
        wandb_log.update({
                    "rollout/ttc_x_reward": episode_statistics['ttc_x_reward']/episode_statistics['episode_duration'],
                    "rollout/ttc_y_reward": episode_statistics['ttc_y_reward']/episode_statistics['episode_duration'],
                    f"rollout/{spawn_config}/ttc_x_reward": episode_statistics['ttc_x_reward']/episode_statistics['episode_duration'],
                    f"rollout/{spawn_config}/ttc_y_reward": episode_statistics['ttc_y_reward']/episode_statistics['episode_duration'],
        })
    wandb.log(wandb_log, step = episode_statistics['episode_num'])