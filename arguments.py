from dataclasses import dataclass
import os

@dataclass
class Args:

    # General arguments

    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    metal: bool = False
    """if toggled, metal will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "safetyh"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""

    # Environment specific arguments
    max_duration: int = 100
    """maximum duration of one episode"""
    adversarial: bool = True
    """if toggled, the environment will be adversarial"""
    crash_reward: float = 400
    """reward for collision"""
    ttc_x_reward: float = 4
    """coefficient for ttx-x reward"""
    ttc_y_reward: float = 1
    """coefficient for ttx-y reward"""

    # Algorithm specific arguments
    model_type: str = "DQN"
    """the type of the model"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 1e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = 10000
    """the replay memory buffer size"""
    buffer_type: str = "ER"
    """the type of replay memory buffer"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """the target network update rate"""
    batch_size: int = 128
    """the batch size of sample from the reply memory"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.05
    """the ending epsilon for exploration"""
    decay_e: int = 1000
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""