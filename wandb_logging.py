import wandb

def initialize_logging(args, ego_version = None, npc_version = None, train_ego = None, eval = False):
    if wandb.run is not None:
        wandb.finish()

    run_config ={
                "learning_rate": args.learning_rate,
                "architecture": args.model_type,
                "max_duration": args.max_duration,
                "dataset": "Highway-Env",
                "max_steps": args.total_timesteps,
                "collision_reward": args.crash_reward,
                "ttc_x_reward": args.ttc_x_reward,
                "ttc_y_reward": args.ttc_y_reward,
                "batch_size": args.batch_size,
                "gamma": args.gamma,
                "eps_start": args.start_e,
                "eps_end": args.end_e,
                "eps_decay": args.decay_e,
                "tau": args.tau,
                "ReplayBuffer": args.buffer_type,
                "eval": eval
    }

    if ego_version is not None:
        run_config['ego_version'] = ego_version
    if npc_version is not None:
        run_config['npc_version'] = npc_version
    if train_ego is not None:
        run_config['train_ego'] = train_ego
    
        
    run = wandb.init(
            # set the wandb project where this run will be logged
            project=args.wandb_project_name,
            
            # track hyperparameters and run metadata
            config=run_config
        )
    return run