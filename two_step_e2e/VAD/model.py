"""
VAD: Vectorized Scene Representation for Efficient Autonomous Driving

Key components:
1. BEV feature extraction (simplified)
2. Vectorized agent prediction (motion vectors)
3. Vectorized map prediction (polyline vectors)
4. Ego planning with K trajectory candidates + scoring

Reference: https://github.com/hustvl/VAD
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class VectorizedAgentDecoder(nn.Module):
    """
    Predicts agent motion as vectorized representations.
    Each agent query attends to BEV and produces motion vectors.
    """

    def __init__(self, embed_dim: int = 256, num_queries: int = 300,
                 num_heads: int = 8, num_layers: int = 6,
                 future_steps: int = 12, num_modes: int = 6):
        super().__init__()
        self.num_queries = num_queries
        self.future_steps = future_steps
        self.num_modes = num_modes

        self.queries = nn.Embedding(num_queries, embed_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 4, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Output heads
        self.class_head = nn.Linear(embed_dim, 10 + 1)  # nuScenes classes + background
        self.motion_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, num_modes * future_steps * 2),  # K modes × T steps × (dx,dy)
        )
        self.mode_prob_head = nn.Linear(embed_dim, num_modes)

    def forward(self, bev_features: torch.Tensor) -> Dict:
        """
        Args:
            bev_features: (B, HW, D) BEV feature tokens
        Returns:
            agent_features: (B, Q, D)
            motion_vectors: (B, Q, K, T, 2) predicted displacements
            mode_probs: (B, Q, K)
            classes: (B, Q, num_classes)
        """
        B = bev_features.shape[0]
        queries = self.queries.weight.unsqueeze(0).expand(B, -1, -1)

        agent_features = self.decoder(queries, bev_features)

        classes = self.class_head(agent_features)
        motion = self.motion_head(agent_features)
        motion = motion.reshape(B, self.num_queries, self.num_modes, self.future_steps, 2)
        mode_probs = F.softmax(self.mode_prob_head(agent_features), dim=-1)

        return {
            'agent_features': agent_features,  # (B, Q, D)
            'motion_vectors': motion,          # (B, Q, K, T, 2)
            'mode_probs': mode_probs,          # (B, Q, K)
            'classes': classes,                # (B, Q, 11)
        }


class VectorizedMapDecoder(nn.Module):
    """
    Predicts map elements as vectorized polylines.
    """

    def __init__(self, embed_dim: int = 256, num_queries: int = 100,
                 num_heads: int = 8, num_layers: int = 6,
                 num_points: int = 20, num_classes: int = 3):
        super().__init__()
        self.num_queries = num_queries
        self.num_points = num_points

        self.queries = nn.Embedding(num_queries, embed_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 4, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.class_head = nn.Linear(embed_dim, num_classes + 1)
        self.polyline_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, num_points * 2),
        )

    def forward(self, bev_features: torch.Tensor) -> Dict:
        """
        Args:
            bev_features: (B, HW, D)
        Returns:
            map_features: (B, Q, D)
            polylines: (B, Q, num_points, 2)
            classes: (B, Q, num_classes+1)
        """
        B = bev_features.shape[0]
        queries = self.queries.weight.unsqueeze(0).expand(B, -1, -1)

        map_features = self.decoder(queries, bev_features)

        classes = self.class_head(map_features)
        polylines = self.polyline_head(map_features)
        polylines = polylines.reshape(B, self.num_queries, self.num_points, 2)

        return {
            'map_features': map_features,
            'polylines': polylines,
            'classes': classes,
        }


class EgoPlanningHead(nn.Module):
    """
    VAD's ego planning: K learnable ego queries produce K candidate
    trajectories. A scoring head selects the best one.

    Key innovation: ego queries attend to both agent vectors and map vectors
    (not dense BEV), making planning efficient.
    """

    def __init__(self, embed_dim: int = 256, num_ego_queries: int = 6,
                 num_heads: int = 8, num_plan_layers: int = 3,
                 num_waypoints: int = 6):
        super().__init__()
        self.num_ego_queries = num_ego_queries
        self.num_waypoints = num_waypoints

        # Learnable ego queries
        self.ego_queries = nn.Embedding(num_ego_queries, embed_dim)

        # Cross-attention to agent vectors
        self.ego_agent_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
            for _ in range(num_plan_layers)
        ])
        self.ego_agent_norms = nn.ModuleList([
            nn.LayerNorm(embed_dim) for _ in range(num_plan_layers)
        ])

        # Cross-attention to map vectors
        self.ego_map_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
            for _ in range(num_plan_layers)
        ])
        self.ego_map_norms = nn.ModuleList([
            nn.LayerNorm(embed_dim) for _ in range(num_plan_layers)
        ])

        # FFN after each attention pair
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 4),
                nn.GELU(),
                nn.Linear(embed_dim * 4, embed_dim),
            ) for _ in range(num_plan_layers)
        ])
        self.ffn_norms = nn.ModuleList([
            nn.LayerNorm(embed_dim) for _ in range(num_plan_layers)
        ])

        # Trajectory output: each ego query → trajectory
        self.trajectory_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, num_waypoints * 2),
        )

        # Scoring head: predict quality of each trajectory
        self.score_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1),
        )

    def forward(self, agent_features: torch.Tensor,
                map_features: torch.Tensor) -> Dict:
        """
        Args:
            agent_features: (B, Na, D) from VectorizedAgentDecoder
            map_features: (B, Nm, D) from VectorizedMapDecoder
        Returns:
            trajectories: (B, K, T, 2) K candidate trajectories
            scores: (B, K) quality scores
            best_trajectory: (B, T, 2) selected best
        """
        B = agent_features.shape[0]
        ego_q = self.ego_queries.weight.unsqueeze(0).expand(B, -1, -1)  # (B, K, D)

        # Multi-layer cross-attention
        for i in range(len(self.ego_agent_layers)):
            # Attend to agents
            residual = ego_q
            ego_normed = self.ego_agent_norms[i](ego_q)
            ego_q = residual + self.ego_agent_layers[i](
                ego_normed, agent_features, agent_features)[0]

            # Attend to map
            residual = ego_q
            ego_normed = self.ego_map_norms[i](ego_q)
            ego_q = residual + self.ego_map_layers[i](
                ego_normed, map_features, map_features)[0]

            # FFN
            residual = ego_q
            ego_normed = self.ffn_norms[i](ego_q)
            ego_q = residual + self.ffn_layers[i](ego_normed)

        # Predict trajectories and scores
        trajectories = self.trajectory_head(ego_q)  # (B, K, T*2)
        trajectories = trajectories.reshape(B, self.num_ego_queries, self.num_waypoints, 2)

        scores = self.score_head(ego_q).squeeze(-1)  # (B, K)

        # Select best trajectory at inference
        best_idx = scores.argmax(dim=-1)  # (B,)
        best_trajectory = torch.stack([
            trajectories[b, best_idx[b]] for b in range(B)
        ])  # (B, T, 2)

        return {
            'trajectories': trajectories,
            'scores': scores,
            'best_trajectory': best_trajectory,
            'best_idx': best_idx,
        }


class VAD(nn.Module):
    """
    VAD: Vectorized Autonomous Driving

    Full pipeline:
        Images → BEV → Agent Vectors + Map Vectors → Ego Planning → Trajectory
    """

    def __init__(self, embed_dim: int = 256, bev_h: int = 200, bev_w: int = 200,
                 num_cameras: int = 6, num_ego_queries: int = 6,
                 num_waypoints: int = 6):
        super().__init__()
        self.embed_dim = embed_dim

        # Simplified BEV encoder
        self.bev_encoder = nn.Sequential(
            nn.Conv2d(3 * num_cameras, 128, 7, stride=4, padding=3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, embed_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((bev_h // 4, bev_w // 4)),
        )

        self.agent_decoder = VectorizedAgentDecoder(embed_dim=embed_dim)
        self.map_decoder = VectorizedMapDecoder(embed_dim=embed_dim)
        self.planner = EgoPlanningHead(
            embed_dim=embed_dim, num_ego_queries=num_ego_queries,
            num_waypoints=num_waypoints)

    def forward(self, images: torch.Tensor) -> Dict:
        """
        Args:
            images: (B, num_cameras, 3, H, W) multi-view images
        Returns:
            dict with all outputs
        """
        B, N, C, H, W = images.shape
        imgs = images.reshape(B, N * C, H, W)

        # BEV encoding
        bev = self.bev_encoder(imgs)  # (B, D, h, w)
        bev_tokens = bev.flatten(2).permute(0, 2, 1)  # (B, HW, D)

        # Vectorized perception
        agent_output = self.agent_decoder(bev_tokens)
        map_output = self.map_decoder(bev_tokens)

        # Planning
        plan_output = self.planner(
            agent_output['agent_features'],
            map_output['map_features'])

        return {
            'agents': agent_output,
            'map': map_output,
            'plan': plan_output,
        }


def compute_vad_loss(outputs: Dict, gt_trajectory: torch.Tensor,
                     gt_agent_motions: Optional[torch.Tensor] = None,
                     gt_map_polylines: Optional[torch.Tensor] = None) -> Dict:
    """Compute VAD training losses."""
    plan = outputs['plan']
    trajectories = plan['trajectories']  # (B, K, T, 2)
    scores = plan['scores']              # (B, K)

    B, K, T, _ = trajectories.shape

    # Find best matching trajectory to GT (winner-take-all)
    gt_exp = gt_trajectory.unsqueeze(1).expand_as(trajectories)
    distances = torch.norm(trajectories - gt_exp, dim=-1).mean(dim=-1)  # (B, K)
    best_match = distances.argmin(dim=-1)  # (B,)

    # Planning regression loss (on best-matching trajectory only)
    plan_loss = torch.tensor(0.0, device=trajectories.device)
    for b in range(B):
        plan_loss = plan_loss + F.l1_loss(
            trajectories[b, best_match[b]], gt_trajectory[b])
    plan_loss = plan_loss / B

    # Scoring loss: best_match should have highest score
    score_target = torch.zeros_like(scores)
    for b in range(B):
        score_target[b, best_match[b]] = 1.0
    score_loss = F.binary_cross_entropy_with_logits(scores, score_target)

    total = plan_loss + 0.5 * score_loss

    return {
        'total': total,
        'planning': plan_loss,
        'scoring': score_loss,
    }


def demo():
    """Demo VAD."""
    print("VAD: Vectorized Autonomous Driving Demo")
    print("=" * 45)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = VAD(embed_dim=256, num_ego_queries=6, num_waypoints=6).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")

    B = 2
    images = torch.randn(B, 6, 3, 224, 400, device=device)

    with torch.no_grad():
        output = model(images)

    plan = output['plan']
    print(f"\nOutputs:")
    print(f"  Agent motion vectors: {output['agents']['motion_vectors'].shape}")
    print(f"  Map polylines: {output['map']['polylines'].shape}")
    print(f"  Candidate trajectories: {plan['trajectories'].shape}")
    print(f"  Scores: {plan['scores'].shape}")
    print(f"  Best trajectory: {plan['best_trajectory'].shape}")
    print(f"  Best indices: {plan['best_idx'].tolist()}")

    # Loss demo
    gt_traj = torch.randn(B, 6, 2, device=device)
    loss = compute_vad_loss(output, gt_traj)
    print(f"\n  Loss: {loss['total'].item():.4f} "
          f"(plan={loss['planning'].item():.4f}, score={loss['scoring'].item():.4f})")


if __name__ == '__main__':
    demo()
