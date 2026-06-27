"""UniAD Model Configuration."""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class BEVConfig:
    """BEV encoder configuration."""
    bev_h: int = 200
    bev_w: int = 200
    bev_z: int = 1
    embed_dims: int = 256
    num_cameras: int = 6
    img_size: Tuple[int, int] = (900, 1600)
    backbone: str = 'resnet101'
    num_points_in_pillar: int = 4
    num_levels: int = 4
    temporal_num_frames: int = 4


@dataclass
class TrackConfig:
    """TrackFormer configuration."""
    num_queries: int = 900
    num_classes: int = 10
    embed_dims: int = 256
    num_heads: int = 8
    num_layers: int = 6
    ffn_dim: int = 2048
    num_track_queries: int = 300  # persistent track queries


@dataclass
class MapConfig:
    """MapFormer configuration."""
    num_queries: int = 100
    num_classes: int = 3  # lane divider, road boundary, pedestrian crossing
    embed_dims: int = 256
    num_points_per_polyline: int = 20
    num_layers: int = 6


@dataclass
class MotionConfig:
    """MotionFormer configuration."""
    embed_dims: int = 256
    num_heads: int = 8
    num_layers: int = 6
    num_modes: int = 6  # multimodal predictions
    future_steps: int = 12  # 6s at 2Hz
    num_agents_max: int = 300


@dataclass
class PlannerConfig:
    """Planning module configuration."""
    embed_dims: int = 256
    num_future_steps: int = 6  # 3s at 2Hz
    gru_hidden_dim: int = 512
    use_ego_query: bool = True
    collision_loss_weight: float = 5.0
    l2_loss_weight: float = 1.0


@dataclass
class UniADConfig:
    """Full UniAD configuration."""
    bev: BEVConfig = field(default_factory=BEVConfig)
    track: TrackConfig = field(default_factory=TrackConfig)
    map: MapConfig = field(default_factory=MapConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)

    # Training
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    batch_size: int = 1  # per GPU (limited by memory)
    num_epochs: int = 24  # per stage
    warmup_epochs: int = 1
    grad_clip: float = 35.0

    # Loss weights
    loss_weights: dict = field(default_factory=lambda: {
        'detection': 1.0,
        'tracking': 1.0,
        'mapping': 5.0,
        'motion': 1.0,
        'occupancy': 1.0,
        'planning_l2': 1.0,
        'planning_collision': 5.0,
    })
