import os

import numpy as np

import wandb


def initialize_logging(
    config,
    train_ego=None,
    eval=False,
    checkpoint=False,
    npc_pool_size=None,
    ego_pool_size=None,
):
    if wandb.run is not None:
        wandb.finish()

    is_sweep = bool(os.getenv("WANDB_SWEEP_ID"))

    gym_config = config.get("gym_config", {})

    run_config = {
        "learning_rate": config.get("learning_rate", 5e-4),
        "architecture": config.get("architecture", "DQN"),
        "dataset": "Highway-Env",
        "max_steps": config.get("total_timesteps", 100000),
        "collision_reward": gym_config.get("collision_reward", 400),
        "ttc_x_reward": gym_config.get("ttc_x_reward", 4),
        "ttc_y_reward": gym_config.get("ttc_y_reward", 1),
        "batch_size": config.get("batch_size", 32),
        "gamma": config.get("gamma", 0.8),
        "eps_start": config.get("start_e", 1.0),
        "eps_end": config.get("end_e", 0.05),
        "eps_decay": config.get("decay_e", 6000),
        "tau": config.get("tau", 0.005),
        "ReplayBuffer": config.get("buffer_type", "ER"),
        "eval": eval,
        "checkpoint_testing": config.get("enable_checkpoint_testing", True),
        "test_interval": config.get("test_interval", 10000),
    }

    run_config["ego_version"] = config.get("ego_version", 0)
    run_config["npc_version"] = config.get("npc_version", 0)
    if train_ego is not None:
        run_config["train_ego"] = train_ego
    if npc_pool_size is not None:
        run_config["npc_pool"] = True
        run_config["npc_pool_size"] = npc_pool_size
    else:
        run_config["npc_pool"] = False
    if ego_pool_size is not None:
        run_config["ego_pool"] = True
        run_config["ego_pool_size"] = ego_pool_size
    else:
        run_config["ego_pool"] = False

    if config.get("wandb_tag", None) is not None:
        tags = config.get("wandb_tag").split(",")
    else:
        tags = []

    init_args = {"config": run_config, "tags": tags}

    if not is_sweep:
        init_args["project"] = config.get("wandb_project_name", "collision-detection")

    run = wandb.init(**init_args)

    if is_sweep:
        config.update(dict(wandb.config))

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


def log_stats(info, episode_statistics, checkpoint=False, checkpoint_step=None):
    """Log episode statistics to wandb"""
    if not wandb.run:
        return
        
    episode_duration = episode_statistics.get("episode_duration", 1)
    
    # Determine prefix based on context
    if checkpoint:
        prefix = "checkpoint_episode"
        step_key = "testing_step"
    else:
        prefix = "training"
        step_key = "training_step"
    
    # Basic episode metrics
    metrics = {
        f"{prefix}/episode_reward": episode_statistics.get("episode_rewards", 0),
        f"{prefix}/episode_duration": episode_duration,
        f"{prefix}/avg_ego_speed": episode_statistics.get("ego_speed", 0) / episode_duration,
        f"{prefix}/avg_npc_speed": episode_statistics.get("npc_speed", 0) / episode_duration,
        f"{prefix}/collision_reward": episode_statistics.get("collision_reward", 0),
        f"{prefix}/right_lane_reward": episode_statistics.get("right_lane_reward", 0),
        f"{prefix}/high_speed_reward": episode_statistics.get("high_speed_reward", 0),
        f"{prefix}/ttc_x_reward": episode_statistics.get("ttc_x_reward", 0),
        f"{prefix}/ttc_y_reward": episode_statistics.get("ttc_y_reward", 0),
        f"{prefix}/epsilon": episode_statistics.get("epsilon", 0),
    }
    
    # Add scenario-specific metrics if available
    scenario = episode_statistics.get("scenario")
    if scenario:
        metrics[f"{prefix}/scenario"] = scenario
    
    # Log with appropriate step
    if checkpoint and checkpoint_step is not None:
        metrics[step_key] = checkpoint_step
    else:
        metrics[step_key] = episode_statistics.get(step_key, 0)
    
    wandb.log(metrics)


def log_checkpoint_summary(stats, checkpoint_step):
    """Log checkpoint summary metrics"""
    if not wandb.run:
        return
        
    # This would contain aggregated metrics across all checkpoint episodes
    summary_metrics = {
        "checkpoint_summary/step": checkpoint_step,
        "checkpoint_step": checkpoint_step,
    }
    
    wandb.log(summary_metrics)
