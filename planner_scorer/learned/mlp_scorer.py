"""
MLP-based Trajectory Scorer

Encodes trajectory and scene features, concatenates, and scores via MLP.

Architecture:
    trajectory_features = MLP(flatten(waypoints))
    scene_features = MLP(pool(agents + map))
    score = ScorerHead(concat(trajectory_features, scene_features))
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class MLPScorer(nn.Module):
    """
    MLP trajectory scorer. Higher score = better trajectory.
    """

    def __init__(self, traj_points: int = 16, traj_dim: int = 4,
                 agent_dim: int = 7, map_dim: int = 5,
                 hidden_dim: int = 256, max_agents: int = 32,
                 max_map_polylines: int = 64, dropout: float = 0.1):
        super().__init__()

        # Trajectory encoder: flatten waypoints -> features
        traj_input_dim = traj_points * traj_dim
        self.traj_encoder = nn.Sequential(
            nn.Linear(traj_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
        )

        # Agent encoder: per-agent MLP + max-pool
        self.agent_encoder = nn.Sequential(
            nn.Linear(agent_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
        )

        # Map encoder: per-polyline feature -> pool
        self.map_encoder = nn.Sequential(
            nn.Linear(map_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 4),
        )

        # Scene fusion
        scene_dim = hidden_dim // 2 + hidden_dim // 4
        self.scene_fusion = nn.Sequential(
            nn.Linear(scene_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
        )

        # Scoring head: concat(traj_feat, scene_feat) -> score
        scorer_input = hidden_dim // 2 + hidden_dim // 2
        self.scorer = nn.Sequential(
            nn.Linear(scorer_input, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, trajectory: torch.Tensor,
                agents: torch.Tensor,
                agent_mask: torch.Tensor,
                map_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            trajectory: (B, T, traj_dim) waypoints
            agents: (B, N, agent_dim) agent states
            agent_mask: (B, N) True = valid agent
            map_features: (B, M, map_dim) polyline features
        Returns:
            score: (B, 1) raw logit (apply sigmoid for [0,1])
        """
        B = trajectory.shape[0]

        # Encode trajectory
        traj_flat = trajectory.reshape(B, -1)
        traj_feat = self.traj_encoder(traj_flat)  # (B, hidden/2)

        # Encode agents with masking (use -inf for invalid agents before max-pool)
        agent_feat = self.agent_encoder(agents)  # (B, N, hidden/2)
        agent_feat = agent_feat.masked_fill(~agent_mask.unsqueeze(-1), float('-inf'))
        agent_pooled = agent_feat.max(dim=1)[0]  # (B, hidden/2)

        # Encode map
        map_feat = self.map_encoder(map_features)  # (B, M, hidden/4)
        map_pooled = map_feat.max(dim=1)[0]  # (B, hidden/4)

        # Fuse scene
        scene = torch.cat([agent_pooled, map_pooled], dim=-1)
        scene_feat = self.scene_fusion(scene)  # (B, hidden/2)

        # Score
        combined = torch.cat([traj_feat, scene_feat], dim=-1)
        return self.scorer(combined)

    def score_candidates(self, candidates: torch.Tensor,
                         agents: torch.Tensor,
                         agent_mask: torch.Tensor,
                         map_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Score K candidate trajectories.

        Args:
            candidates: (B, K, T, D) K candidate trajectories
        Returns:
            scores: (B, K)
            best_idx: (B,)
        """
        B, K, T, D = candidates.shape

        # Flatten for batch scoring
        cands_flat = candidates.reshape(B * K, T, D)
        agents_exp = agents.unsqueeze(1).expand(-1, K, -1, -1).reshape(B * K, -1, agents.shape[-1])
        mask_exp = agent_mask.unsqueeze(1).expand(-1, K, -1).reshape(B * K, -1)
        map_exp = map_features.unsqueeze(1).expand(-1, K, -1, -1).reshape(B * K, -1, map_features.shape[-1])

        scores = self.forward(cands_flat, agents_exp, mask_exp, map_exp)
        scores = scores.reshape(B, K)
        best_idx = scores.argmax(dim=-1)
        return scores, best_idx


def demo():
    """Quick demo."""
    print("MLP Trajectory Scorer Demo")
    print("=" * 40)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MLPScorer().to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Synthetic data
    B, K, T = 4, 64, 16
    candidates = torch.randn(B, K, T, 4, device=device)
    agents = torch.randn(B, 32, 7, device=device)
    mask = torch.ones(B, 32, dtype=torch.bool, device=device)
    map_feat = torch.randn(B, 64, 5, device=device)

    with torch.no_grad():
        scores, best = model.score_candidates(candidates, agents, mask, map_feat)
    print(f"Scores: {scores.shape}, Best: {best.tolist()}")
    print(f"Range: [{scores.min():.3f}, {scores.max():.3f}]")


if __name__ == '__main__':
    demo()
