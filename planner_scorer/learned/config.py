"""
Configuration for learned trajectory scorers.
"""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class DataConfig:
    """Dataset configuration."""
    trajectory_length: int = 16       # waypoints per trajectory (8s at 2Hz)
    trajectory_dim: int = 4           # (x, y, heading, velocity)
    num_agents_max: int = 32          # max agents in scene
    agent_history_len: int = 10       # past timesteps for agents
    agent_feature_dim: int = 7        # (x, y, heading, vx, vy, length, width)
    map_polyline_max: int = 64        # max map polylines
    map_points_per_polyline: int = 20 # points per polyline
    map_feature_dim: int = 5          # (x, y, dx, dy, type)
    num_candidates: int = 64          # candidate trajectories per scene
    num_negatives: int = 15           # negative samples per positive
    bev_size: Tuple[int, int] = (200, 200)  # BEV raster size
    bev_resolution: float = 0.5      # meters per pixel


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    # Trajectory encoder
    traj_embed_dim: int = 128
    traj_num_layers: int = 2
    traj_num_heads: int = 4

    # Scene encoder
    scene_embed_dim: int = 256
    scene_num_layers: int = 4
    scene_num_heads: int = 8

    # Cross-attention (trajectory attends to scene)
    cross_attn_layers: int = 3
    cross_attn_heads: int = 8

    # Scorer MLP head
    scorer_hidden_dims: List[int] = field(default_factory=lambda: [256, 128, 64])
    scorer_dropout: float = 0.1

    # General
    hidden_dim: int = 256
    dropout: float = 0.1
    activation: str = 'gelu'


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Optimization
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 100
    warmup_epochs: int = 5
    lr_scheduler: str = 'cosine'  # 'cosine', 'step', 'plateau'
    gradient_clip: float = 1.0

    # Loss
    loss_type: str = 'combined'  # 'bce', 'ranking', 'contrastive', 'combined'
    contrastive_temperature: float = 0.07
    ranking_margin: float = 0.5
    loss_weights: dict = field(default_factory=lambda: {
        'classification': 1.0,
        'ranking': 0.5,
        'contrastive': 0.3,
    })

    # Data augmentation
    noise_std: float = 0.1        # Gaussian noise on trajectory waypoints
    drop_agent_prob: float = 0.1  # randomly drop agents

    # Logging
    log_interval: int = 50
    val_interval: int = 1  # epochs
    save_interval: int = 5  # epochs
    checkpoint_dir: str = 'checkpoints'
    log_dir: str = 'logs'

    # Hardware
    device: str = 'cuda'
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class FullConfig:
    """Complete configuration."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
