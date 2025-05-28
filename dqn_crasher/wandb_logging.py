import wandb

def initialize_logging(config, train_ego = None, eval = False, checkpoint = False, npc_pool_size = None, ego_pool_size = None):
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
                "eval": eval,
                "checkpoint_testing": config.get('enable_checkpoint_testing', True),
                "test_interval": config.get('test_interval', 10000)
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

    # Define separate x-axes for different metric types
    # Training metrics use training_step
    wandb.define_metric("training_step")
    wandb.define_metric("training/*", step_metric="training_step")

    # Testing episode metrics use testing_step
    wandb.define_metric("testing_step")
    wandb.define_metric("checkpoint_episode/*", step_metric="testing_step")

    # Checkpoint summary metrics use checkpoint_step
    wandb.define_metric("checkpoint_step")
    wandb.define_metric("checkpoint_summary/*", step_metric="checkpoint_step")

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


def log_stats(info, episode_statistics: dict, policy_prefix=None, checkpoint=False, checkpoint_step=None):

    # Add crash data
    if info['scenario'] not in episode_statistics:
        episode_statistics['num_crashes'][info['scenario']] = []

    episode_statistics['num_crashes'][info['scenario']].append(info['crashed'])
    episode_statistics['total_crashes'].append(info['crashed'])

    # Calculate episode statistics safely
    episode_duration = max(1, episode_statistics.get('episode_duration', 1))  # Avoid div/0

    # Get statistics with defaults to prevent KeyError
    ep_rewards = episode_statistics.get('episode_rewards', 0)
    ep_duration = episode_statistics.get('episode_duration', 1)
    ego_speed = episode_statistics.get('ego_speed', 0)
    npc_speed = episode_statistics.get('npc_speed', 0)

    # Initialize deque collections if they don't exist
    for key in ['ep_rew_total', 'ep_len_total', 'ego_speed_total', 'npc_speed_total']:
        if key not in episode_statistics:
            from collections import deque
            episode_statistics[key] = deque([], maxlen=100)

    # Add statistics to deques
    episode_statistics['ep_rew_total'].append(ep_rewards)
    episode_statistics['ep_len_total'].append(ep_duration)
    episode_statistics['ego_speed_total'].append(ego_speed/episode_duration)
    episode_statistics['npc_speed_total'].append(npc_speed/episode_duration)

    # Use more descriptive prefixes for different metric types
    if checkpoint:
        prefix = "checkpoint"  # Individual checkpoint testing episodes
    else:
        prefix = "training"  # Regular training metrics

    if policy_prefix:
        prefix = f"{prefix}/{policy_prefix}"

    # Determine which step metric to use
    if checkpoint:
        step_metric = "testing_step"
    else:
        step_metric = "training_step"


    # For crash rate calculation, use min to avoid division by zero
    total_crashes = episode_statistics.get('total_crashes', [])
    crash_rate_divisor = min(100, len(total_crashes))
    config_crash_rate_divisor = min(100, len(episode_statistics.get('num_crashes', {}).get(info['scenario'], [])))

    # Get all values with safe defaults
    ep_rew_total = episode_statistics.get('ep_rew_total', [0])
    ep_len_total = episode_statistics.get('ep_len_total', [0])
    ego_speed = episode_statistics.get('ego_speed', 0)
    npc_speed = episode_statistics.get('npc_speed', 0)
    epsilon = episode_statistics.get('epsilon', 0)
    collision_reward = episode_statistics.get('collision_reward', 0)
    episode_duration = max(1, episode_statistics.get('episode_duration', 1))

    # Build log dictionary with safe calculations
    wandb_log = {
        f"{prefix}/ep_rew_mean": sum(ep_rew_total)/max(1, len(ep_rew_total)),
        f"{prefix}/ep_len_mean": sum(ep_len_total)/max(1, len(ep_len_total)),
        f"{prefix}/num_crashes": sum(total_crashes),
        f"{prefix}/sr100": sum(total_crashes[-crash_rate_divisor:])/max(1, crash_rate_divisor),
        f"{prefix}/ego_speed_mean": ego_speed/episode_duration,
        f"{prefix}/npc_speed_mean": npc_speed/episode_duration,
        f"{prefix}/scenario": info['scenario'],
        f"{prefix}/epsilon": epsilon,
        f"{prefix}/collision_reward": collision_reward/episode_duration,
        f"{prefix}/{info['scenario']}/ego_speed_mean": ego_speed/episode_duration,
        f"{prefix}/{info['scenario']}/npc_speed_mean": npc_speed/episode_duration,
    }

    # Only add these metrics if we have crash data for this spawn config
    num_crashes = episode_statistics.get('num_crashes', {})
    if info['scenario'] in num_crashes and len(num_crashes[info['scenario']]) > 0:
        # Get safe crashes for this config and calculate metrics
        config_crashes = num_crashes[info['scenario']]
        crash_count = sum(config_crashes)
        sr100 = sum(config_crashes[-config_crash_rate_divisor:])/max(1, config_crash_rate_divisor)

        wandb_log.update({
            f"{prefix}/{info['scenario']}/num_crashes": crash_count,
            f"{prefix}/{info['scenario']}/sr100": sr100,
            f"{prefix}/{info['scenario']}/collision_reward": collision_reward/episode_duration,
        })

    # Get the current episode number or checkpoint step
    current_step = episode_statistics.get('episode_num', 0)

    # Add the appropriate step metric to the log dict
    if step_metric == "checkpoint_step":
        wandb_log["checkpoint_step"] = current_step
    elif step_metric == "testing_step":
        wandb_log["testing_step"] = current_step
    else:
        wandb_log["training_step"] = current_step

    # Add identifiers to clarify metric types and grouping
    if checkpoint:
        wandb_log["checkpoint_id"] = checkpoint_step
        wandb_log["metric_collection"] = "checkpoint_testing" if checkpoint else "checkpoint_summary"
    else:
        wandb_log["metric_collection"] = "training"

    # Log with custom tags (step is automatically handled by define_metric)
    wandb.log(wandb_log)


def log_checkpoint_summary(all_stats, policy_stats, checkpoint_step, ego=False, specific_policy_stats=None):
    """
    Log a summary of checkpoint metrics across all episodes

    This creates aggregated statistics showing overall performance at a checkpoint,
    separating them clearly from individual episode metrics with distinct prefixes.

    Args:
        all_stats: Combined statistics across all episodes
        policy_stats: Dictionary mapping policy names to their aggregate statistics
        checkpoint_step: The training step at which this checkpoint was run
        ego: Whether this is for the ego agent
        specific_policy_stats: Dictionary mapping specific policy names to their statistics
    """
    if not all_stats:
        print("Warning: Empty statistics provided to log_checkpoint_summary")
        return

    # Create a deep copy to avoid modifying the original
    summary_stats = all_stats.copy()

    # Calculate additional summary metrics
    # These will make the aggregated summary more useful
    if 'total_crashes' in summary_stats and summary_stats['total_crashes']:
        success_rate = sum(summary_stats['total_crashes']) / max(1, len(summary_stats['total_crashes']))
        summary_stats['success_rate'] = success_rate

    if 'ep_rew_total' in summary_stats and summary_stats['ep_rew_total']:
        avg_reward = sum(summary_stats['ep_rew_total']) / max(1, len(summary_stats['ep_rew_total']))
        summary_stats['avg_reward'] = avg_reward

    if 'ep_len_total' in summary_stats and summary_stats['ep_len_total']:
        avg_episode_length = sum(summary_stats['ep_len_total']) / max(1, len(summary_stats['ep_len_total']))
        summary_stats['avg_episode_length'] = avg_episode_length

    # Add checkpoint metadata for clearer organization
    summary_stats['checkpoint_step'] = checkpoint_step
    summary_stats['summary_type'] = 'overall_performance'

    # Set episode number to checkpoint step to ensure proper ordering
    summary_stats['episode_num'] = checkpoint_step

    # Log the combined stats as a checkpoint summary
    log_stats(None, summary_stats, ego=ego, checkpoint=False,
              checkpoint_summary=True, checkpoint_step=checkpoint_step, aggregated=True)

    # Log per-policy summaries
    if policy_stats:
        for policy_name, policy_stat in policy_stats.items():
            if policy_stat:
                # Create a copy to avoid modifying the original
                policy_summary = policy_stat.copy()

                # Set episode number to checkpoint step to ensure proper ordering
                policy_summary['episode_num'] = checkpoint_step

                # Calculate policy-specific summary metrics
                if 'total_crashes' in policy_summary and policy_summary['total_crashes']:
                    policy_success_rate = sum(policy_summary['total_crashes']) / max(1, len(policy_summary['total_crashes']))
                    policy_summary['success_rate'] = policy_success_rate

                if 'ep_rew_total' in policy_summary and policy_summary['ep_rew_total']:
                    policy_avg_reward = sum(policy_summary['ep_rew_total']) / max(1, len(policy_summary['ep_rew_total']))
                    policy_summary['avg_reward'] = policy_avg_reward

                if 'ep_len_total' in policy_summary and policy_summary['ep_len_total']:
                    policy_avg_episode_length = sum(policy_summary['ep_len_total']) / max(1, len(policy_summary['ep_len_total']))
                    policy_summary['avg_episode_length'] = policy_avg_episode_length

                # Add policy-specific metadata
                policy_summary['checkpoint_step'] = checkpoint_step
                policy_summary['policy_name'] = policy_name
                policy_summary['summary_type'] = 'policy_performance'

                log_stats(None, policy_summary, ego=ego, policy_prefix=policy_name,
                          checkpoint=False, checkpoint_summary=True,
                          checkpoint_step=checkpoint_step, aggregated=True)

    # Log specific policy summaries
    if specific_policy_stats:
        for specific_policy_name, policy_stat in specific_policy_stats.items():
            if policy_stat:
                # Create a copy to avoid modifying the original
                specific_policy_summary = policy_stat.copy()

                # Calculate policy-specific summary metrics
                if 'total_crashes' in specific_policy_summary and specific_policy_summary['total_crashes']:
                    policy_success_rate = sum(specific_policy_summary['total_crashes']) / max(1, len(specific_policy_summary['total_crashes']))
                    specific_policy_summary['success_rate'] = policy_success_rate

                if 'ep_rew_total' in specific_policy_summary and specific_policy_summary['ep_rew_total']:
                    policy_avg_reward = sum(specific_policy_summary['ep_rew_total']) / max(1, len(specific_policy_summary['ep_rew_total']))
                    specific_policy_summary['avg_reward'] = policy_avg_reward

                if 'ep_len_total' in specific_policy_summary and specific_policy_summary['ep_len_total']:
                    policy_avg_episode_length = sum(specific_policy_summary['ep_len_total']) / max(1, len(specific_policy_summary['ep_len_total']))
                    specific_policy_summary['avg_episode_length'] = policy_avg_episode_length

                # Add policy-specific metadata
                specific_policy_summary['checkpoint_step'] = checkpoint_step
                specific_policy_summary['specific_policy_name'] = specific_policy_name
                specific_policy_summary['summary_type'] = 'specific_policy_performance'

                # Log with specific policy prefix
                log_stats(None, specific_policy_summary, ego=ego,
                          policy_prefix=f"specific/{specific_policy_name}",
                          checkpoint=False, checkpoint_summary=True,
                          checkpoint_step=checkpoint_step, aggregated=True)
