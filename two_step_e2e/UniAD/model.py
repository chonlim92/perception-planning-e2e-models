"""
UniAD: Simplified Implementation of the Planning-Oriented Autonomous Driving Framework

This is a reference implementation capturing UniAD's core architecture:
- BEV feature extraction (simplified)
- TrackFormer (object detection + tracking)
- MapFormer (online vectorized mapping)
- MotionFormer (trajectory prediction)
- Planner (GRU-based ego planning)

For the full implementation, see: https://github.com/OpenDriveLab/UniAD
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
from config import UniADConfig


class SimplifiedBEVEncoder(nn.Module):
    """
    Simplified BEV encoder (replaces BEVFormer for demonstration).

    In practice, BEVFormer uses:
    - Spatial cross-attention with deformable attention
    - Temporal self-attention across historical BEV features
    - Multi-scale image features from backbone

    This simplified version uses a ConvNet to simulate BEV feature extraction.
    """

    def __init__(self, config: UniADConfig):
        super().__init__()
        cfg = config.bev
        self.bev_h = cfg.bev_h
        self.bev_w = cfg.bev_w
        self.embed_dims = cfg.embed_dims

        # Simplified: project multi-view concatenated features to BEV
        # Real BEVFormer uses spatial cross-attention with 3D reference points
        self.img_encoder = nn.Sequential(
            nn.Conv2d(3 * cfg.num_cameras, 128, 7, stride=4, padding=3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((cfg.bev_h, cfg.bev_w)),
            nn.Conv2d(256, cfg.embed_dims, 1),
        )

        # Temporal fusion (simplified)
        self.temporal_attn = nn.MultiheadAttention(
            cfg.embed_dims, num_heads=8, batch_first=True)
        self.temporal_norm = nn.LayerNorm(cfg.embed_dims)

    def forward(self, multi_view_imgs: torch.Tensor,
                prev_bev: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            multi_view_imgs: (B, num_cameras, 3, H, W) multi-view images
            prev_bev: (B, bev_h*bev_w, embed_dims) previous BEV features
        Returns:
            bev_features: (B, bev_h*bev_w, embed_dims) BEV feature tokens
        """
        B, N, C, H, W = multi_view_imgs.shape
        # Concatenate camera views along channel dim
        imgs = multi_view_imgs.reshape(B, N * C, H, W)
        bev = self.img_encoder(imgs)  # (B, embed_dims, bev_h, bev_w)
        bev = bev.flatten(2).permute(0, 2, 1)  # (B, bev_h*bev_w, embed_dims)

        # Temporal fusion with previous BEV
        if prev_bev is not None:
            residual = bev
            bev_normed = self.temporal_norm(bev)
            bev = residual + self.temporal_attn(bev_normed, prev_bev, prev_bev)[0]

        return bev


class TrackFormer(nn.Module):
    """
    Simplified TrackFormer for joint detection + tracking.

    Core idea: maintain persistent track queries that carry object identity
    across frames, plus newborn queries for detecting new objects.
    """

    def __init__(self, config: UniADConfig):
        super().__init__()
        cfg = config.track
        self.embed_dims = cfg.embed_dims
        self.num_queries = cfg.num_queries
        self.num_classes = cfg.num_classes

        # Object queries (newborn + track)
        self.queries = nn.Embedding(cfg.num_queries, cfg.embed_dims)

        # Decoder layers (simplified transformer decoder)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=cfg.embed_dims, nhead=cfg.num_heads,
            dim_feedforward=cfg.ffn_dim, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=cfg.num_layers)

        # Output heads
        self.class_head = nn.Linear(cfg.embed_dims, cfg.num_classes + 1)  # +1 for no-object
        self.bbox_head = nn.Sequential(
            nn.Linear(cfg.embed_dims, cfg.embed_dims),
            nn.ReLU(),
            nn.Linear(cfg.embed_dims, 10),  # (cx, cy, cz, w, l, h, sin, cos, vx, vy)
        )

    def forward(self, bev_features: torch.Tensor,
                track_queries: Optional[torch.Tensor] = None) -> Dict:
        """
        Args:
            bev_features: (B, HW, D) BEV feature tokens
            track_queries: (B, N_track, D) persistent track queries from prev frame
        Returns:
            dict with 'agent_features', 'classes', 'boxes', 'queries'
        """
        B = bev_features.shape[0]

        # Initialize queries
        queries = self.queries.weight.unsqueeze(0).expand(B, -1, -1)
        if track_queries is not None:
            queries = torch.cat([track_queries, queries[:, :self.num_queries - track_queries.shape[1]]], dim=1)

        # Decode
        agent_features = self.decoder(queries, bev_features)  # (B, Q, D)

        # Predict
        classes = self.class_head(agent_features)  # (B, Q, num_classes+1)
        boxes = self.bbox_head(agent_features)     # (B, Q, 10)

        return {
            'agent_features': agent_features,
            'classes': classes,
            'boxes': boxes,
            'queries': agent_features,  # for next-frame tracking
        }


class MapFormer(nn.Module):
    """
    Simplified MapFormer for online vectorized map prediction.

    Predicts polylines for: lane dividers, road boundaries, pedestrian crossings.
    """

    def __init__(self, config: UniADConfig):
        super().__init__()
        cfg = config.map
        self.embed_dims = cfg.embed_dims
        self.num_queries = cfg.num_queries
        self.num_points = cfg.num_points_per_polyline

        # Map queries
        self.queries = nn.Embedding(cfg.num_queries, cfg.embed_dims)

        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=cfg.embed_dims, nhead=8,
            dim_feedforward=cfg.embed_dims * 4, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=cfg.num_layers)

        # Prediction heads
        self.class_head = nn.Linear(cfg.embed_dims, cfg.num_classes + 1)
        self.point_head = nn.Sequential(
            nn.Linear(cfg.embed_dims, cfg.embed_dims),
            nn.ReLU(),
            nn.Linear(cfg.embed_dims, cfg.num_points_per_polyline * 2),  # (x, y) per point
        )

    def forward(self, bev_features: torch.Tensor) -> Dict:
        """
        Args:
            bev_features: (B, HW, D) BEV feature tokens
        Returns:
            dict with 'map_features', 'classes', 'polylines'
        """
        B = bev_features.shape[0]
        queries = self.queries.weight.unsqueeze(0).expand(B, -1, -1)

        map_features = self.decoder(queries, bev_features)

        classes = self.class_head(map_features)  # (B, Q, num_classes+1)
        polylines = self.point_head(map_features)  # (B, Q, num_points*2)
        polylines = polylines.reshape(B, self.num_queries, self.num_points, 2)

        return {
            'map_features': map_features,
            'classes': classes,
            'polylines': polylines,
        }


class MotionFormer(nn.Module):
    """
    MotionFormer for multi-agent trajectory prediction.

    Takes agent features from TrackFormer and map features from MapFormer,
    models agent-agent and agent-map interactions to predict futures.
    """

    def __init__(self, config: UniADConfig):
        super().__init__()
        cfg = config.motion
        self.embed_dims = cfg.embed_dims
        self.num_modes = cfg.num_modes
        self.future_steps = cfg.future_steps

        # Agent-agent interaction (self-attention among agents)
        self.agent_interaction = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=cfg.embed_dims, nhead=cfg.num_heads,
                dim_feedforward=cfg.embed_dims * 4, batch_first=True),
            num_layers=3)

        # Agent-map interaction (cross-attention)
        self.agent_map_attn = nn.MultiheadAttention(
            cfg.embed_dims, cfg.num_heads, batch_first=True)
        self.map_norm = nn.LayerNorm(cfg.embed_dims)

        # Motion prediction head (multi-modal)
        self.mode_queries = nn.Embedding(cfg.num_modes, cfg.embed_dims)
        self.trajectory_head = nn.Sequential(
            nn.Linear(cfg.embed_dims * 2, cfg.embed_dims),
            nn.ReLU(),
            nn.Linear(cfg.embed_dims, cfg.future_steps * 2),  # (x, y) per step
        )
        self.mode_prob_head = nn.Linear(cfg.embed_dims, 1)

    def forward(self, agent_features: torch.Tensor,
                map_features: torch.Tensor) -> Dict:
        """
        Args:
            agent_features: (B, Na, D) from TrackFormer
            map_features: (B, Nm, D) from MapFormer
        Returns:
            dict with 'predicted_trajectories', 'mode_probs', 'agent_features'
        """
        B, Na, D = agent_features.shape

        # Agent-agent interaction
        agent_feat = self.agent_interaction(agent_features)

        # Agent-map interaction
        agent_normed = self.map_norm(agent_feat)
        agent_feat = agent_feat + self.agent_map_attn(
            agent_normed, map_features, map_features)[0]

        # Multi-modal predictions
        mode_q = self.mode_queries.weight.unsqueeze(0).unsqueeze(1)  # (1, 1, K, D)
        mode_q = mode_q.expand(B, Na, -1, -1)  # (B, Na, K, D)

        agent_exp = agent_feat.unsqueeze(2).expand(-1, -1, self.num_modes, -1)  # (B, Na, K, D)
        combined = torch.cat([agent_exp, mode_q], dim=-1)  # (B, Na, K, 2D)

        # Predict trajectories and probabilities
        combined_flat = combined.reshape(B * Na * self.num_modes, -1)
        traj_flat = self.trajectory_head(combined_flat)
        trajectories = traj_flat.reshape(B, Na, self.num_modes, self.future_steps, 2)

        mode_probs = self.mode_prob_head(agent_exp.reshape(-1, D))
        mode_probs = mode_probs.reshape(B, Na, self.num_modes)
        mode_probs = F.softmax(mode_probs, dim=-1)

        return {
            'predicted_trajectories': trajectories,  # (B, Na, K, T, 2)
            'mode_probs': mode_probs,                # (B, Na, K)
            'agent_features': agent_feat,            # (B, Na, D) updated features
        }


class Planner(nn.Module):
    """
    GRU-based ego-vehicle planner.

    Uses ego query that attends to predicted agent futures and scene features
    to autoregressively decode future waypoints.
    """

    def __init__(self, config: UniADConfig):
        super().__init__()
        cfg = config.planner
        self.embed_dims = cfg.embed_dims
        self.num_future_steps = cfg.num_future_steps

        # Ego query
        self.ego_query = nn.Parameter(torch.randn(1, 1, cfg.embed_dims) * 0.02)

        # Cross-attention to agent predictions
        self.ego_agent_attn = nn.MultiheadAttention(
            cfg.embed_dims, num_heads=8, batch_first=True)
        self.ego_bev_attn = nn.MultiheadAttention(
            cfg.embed_dims, num_heads=8, batch_first=True)

        # GRU decoder for autoregressive waypoint generation
        self.gru = nn.GRUCell(cfg.embed_dims, cfg.gru_hidden_dim)
        self.hidden_proj = nn.Linear(cfg.embed_dims, cfg.gru_hidden_dim)

        # Waypoint output head
        self.waypoint_head = nn.Sequential(
            nn.Linear(cfg.gru_hidden_dim, cfg.embed_dims),
            nn.ReLU(),
            nn.Linear(cfg.embed_dims, 2),  # (dx, dy) relative to previous
        )

        # Embed waypoint back to hidden dim for next step
        self.waypoint_embed = nn.Linear(2, cfg.embed_dims)

    def forward(self, bev_features: torch.Tensor,
                agent_features: torch.Tensor,
                predicted_trajectories: torch.Tensor) -> Dict:
        """
        Args:
            bev_features: (B, HW, D) BEV features
            agent_features: (B, Na, D) updated agent features from MotionFormer
            predicted_trajectories: (B, Na, K, T, 2) predicted agent futures
        Returns:
            dict with 'trajectory' (B, T, 2), 'waypoints' list
        """
        B = bev_features.shape[0]

        # Initialize ego query
        ego_q = self.ego_query.expand(B, -1, -1)  # (B, 1, D)

        # Attend to agents
        ego_q = ego_q + self.ego_agent_attn(ego_q, agent_features, agent_features)[0]

        # Attend to BEV
        ego_q = ego_q + self.ego_bev_attn(ego_q, bev_features, bev_features)[0]

        # Initialize GRU hidden state
        hidden = self.hidden_proj(ego_q.squeeze(1))  # (B, gru_hidden)

        # Autoregressive decoding
        waypoints = []
        input_token = ego_q.squeeze(1)  # (B, D)

        for t in range(self.num_future_steps):
            hidden = self.gru(input_token, hidden)
            waypoint = self.waypoint_head(hidden)  # (B, 2)
            waypoints.append(waypoint)

            # Embed waypoint for next step
            input_token = self.waypoint_embed(waypoint)  # (B, D)

        trajectory = torch.stack(waypoints, dim=1)  # (B, T, 2)

        return {
            'trajectory': trajectory,
            'waypoints': waypoints,
        }


class UniAD(nn.Module):
    """
    Full UniAD model combining all modules.

    Pipeline:
        Images → BEV → TrackFormer → MapFormer → MotionFormer → Planner → Trajectory
    """

    def __init__(self, config: Optional[UniADConfig] = None):
        super().__init__()
        self.config = config or UniADConfig()

        self.bev_encoder = SimplifiedBEVEncoder(self.config)
        self.track_former = TrackFormer(self.config)
        self.map_former = MapFormer(self.config)
        self.motion_former = MotionFormer(self.config)
        self.planner = Planner(self.config)

    def forward(self, multi_view_imgs: torch.Tensor,
                prev_bev: Optional[torch.Tensor] = None,
                prev_track_queries: Optional[torch.Tensor] = None) -> Dict:
        """
        Full forward pass through UniAD.

        Args:
            multi_view_imgs: (B, 6, 3, H, W)
            prev_bev: optional previous BEV features for temporal fusion
            prev_track_queries: optional track queries from previous frame
        Returns:
            dict with all intermediate and final outputs
        """
        # 1. BEV encoding
        bev_features = self.bev_encoder(multi_view_imgs, prev_bev)

        # 2. Detection + Tracking
        track_output = self.track_former(bev_features, prev_track_queries)

        # 3. Online Mapping
        map_output = self.map_former(bev_features)

        # 4. Motion Prediction
        motion_output = self.motion_former(
            track_output['agent_features'], map_output['map_features'])

        # 5. Planning
        plan_output = self.planner(
            bev_features,
            motion_output['agent_features'],
            motion_output['predicted_trajectories'])

        return {
            'bev_features': bev_features,
            'track': track_output,
            'map': map_output,
            'motion': motion_output,
            'plan': plan_output,
        }


def compute_planning_loss(plan_output: Dict, gt_trajectory: torch.Tensor,
                          predicted_occupancy: Optional[torch.Tensor] = None,
                          config: Optional[UniADConfig] = None) -> Dict:
    """
    Compute planning losses.

    Args:
        plan_output: from Planner.forward()
        gt_trajectory: (B, T, 2) expert trajectory
        predicted_occupancy: (B, T, H, W) future occupancy predictions
    """
    cfg = config or UniADConfig()
    pred_traj = plan_output['trajectory']  # (B, T, 2)

    # L2 regression loss
    l2_loss = F.mse_loss(pred_traj, gt_trajectory)

    # Collision loss: penalize overlap with predicted occupancy (differentiable via grid_sample)
    collision_loss = torch.tensor(0.0, device=pred_traj.device)
    if predicted_occupancy is not None:
        B, T_occ, H, W = predicted_occupancy.shape
        T_plan = min(T_occ, pred_traj.shape[1])
        grid_x = (pred_traj[:, :T_plan, 0:1] / 30.0).clamp(-1, 1)
        grid_y = (pred_traj[:, :T_plan, 1:2] / 15.0).clamp(-1, 1)
        grid = torch.cat([grid_x, grid_y], dim=-1).unsqueeze(2)  # (B, T, 1, 2)
        occ_for_sample = predicted_occupancy[:, :T_plan].unsqueeze(2)  # (B, T, 1, H, W)
        sampled = F.grid_sample(
            occ_for_sample.reshape(B * T_plan, 1, H, W),
            grid.reshape(B * T_plan, 1, 1, 2),
            align_corners=True, mode='bilinear', padding_mode='zeros'
        )
        collision_loss = sampled.mean()

    total_loss = (cfg.planner.l2_loss_weight * l2_loss +
                  cfg.planner.collision_loss_weight * collision_loss)

    return {
        'total': total_loss,
        'l2': l2_loss,
        'collision': collision_loss,
    }


def demo():
    """Demo UniAD forward pass."""
    print("UniAD - Simplified Implementation Demo")
    print("=" * 50)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = UniADConfig()
    model = UniAD(config).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {num_params:,}")
    print(f"Device: {device}")

    # Dummy input
    B = 1
    imgs = torch.randn(B, 6, 3, 224, 400, device=device)  # 6 cameras (downscaled)

    print(f"\nInput: {B} scenes × 6 cameras × (3, 224, 400)")

    with torch.no_grad():
        output = model(imgs)

    print(f"\nOutputs:")
    print(f"  BEV features: {output['bev_features'].shape}")
    print(f"  Track - agent features: {output['track']['agent_features'].shape}")
    print(f"  Track - classes: {output['track']['classes'].shape}")
    print(f"  Track - boxes: {output['track']['boxes'].shape}")
    print(f"  Map - features: {output['map']['map_features'].shape}")
    print(f"  Map - polylines: {output['map']['polylines'].shape}")
    print(f"  Motion - trajectories: {output['motion']['predicted_trajectories'].shape}")
    print(f"  Motion - mode probs: {output['motion']['mode_probs'].shape}")
    print(f"  Plan - trajectory: {output['plan']['trajectory'].shape}")

    # Planning loss
    gt_traj = torch.randn(B, config.planner.num_future_steps, 2, device=device)
    loss = compute_planning_loss(output['plan'], gt_traj, config=config)
    print(f"\n  Planning loss: {loss['total'].item():.4f} (L2={loss['l2'].item():.4f})")


if __name__ == '__main__':
    demo()
