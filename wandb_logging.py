import wandb

def initialize_logging(args):
    if wandb.run is not None:
        wandb.finish()
        
    run = wandb.init(
            # set the wandb project where this run will be logged
            project=args.wandb_project_name,
            
            # track hyperparameters and run metadata
            config={
                "learning_rate": args.learning_rate,
                "architecture": args.model_type,
                "max_duration": args.max_duration,
                "dataset": "Highway-Env",
                "max_steps": args.total_timesteps,
                "collision_reward": args.crash_reward,
                "ttc_x_reward": args.ttc_x_reward,
                "ttc_y_reward": args.ttc_y_reward,
                "BATCH_SIZE": args.batch_size,
                "GAMMA": args.gamma,
                "EPS_START": args.start_e,
                "EPS_END": args.end_e,
                "EPS_DECAY": args.decay_e,
                "TAU": args.tau,
                "ReplayBuffer": args.buffer_type
            }
        )
    return run