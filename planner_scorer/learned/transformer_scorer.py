"""
Transformer-based Trajectory Scorer with Cross-Attention

Trajectory waypoints attend to scene elements (agents + map) through
cross-attention layers, enabling spatial reasoning about interactions.

Architecture:
    1. Trajectory tokens (positional encoded waypoints)
    2. Scene tokens (agents + map polylines)
    3. Cross-attention: trajectory queries attend to scene keys
    4. Pool → MLP → scalar score
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple


class TransformerScorer(nn.Module):
    """
    Transformer scorer with cross-attention between trajectory and scene.
    """

    def __init__(self, traj_dim: int = 4, agent_dim: int = 7, map_dim: int = 5,
                 d_model: int = 256, n_heads: int = 8,
                 num_cross_layers: int = 3, num_scene_layers: int = 2,
                 max_traj_len: int = 16, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model

        # Trajectory embedding + positional encoding
        self.traj_proj = nn.Linear(traj_dim, d_model)
        pe = torch.zeros(max_traj_len, d_model)
        pos = torch.arange(0, max_traj_len).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

        # Agent & map projections
        self.agent_proj = nn.Linear(agent_dim, d_model)
        self.map_proj = nn.Linear(map_dim, d_model)

        # Scene self-attention
        scene_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, activation='gelu', batch_first=True)
        self.scene_encoder = nn.TransformerEncoder(scene_layer, num_layers=num_scene_layers)

        # Cross-attention layers (trajectory attends to scene)
        self.cross_layers = nn.ModuleList()
        for _ in range(num_cross_layers):
            self.cross_layers.append(nn.ModuleDict({
                'norm_q': nn.LayerNorm(d_model),
                'norm_kv': nn.LayerNorm(d_model),
                'attn': nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True),
                'norm_ff': nn.LayerNorm(d_model),
                'ff': nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model * 4, d_model),
                    nn.Dropout(dropout),
                ),
            }))

        # Scoring head
        self.output_norm = nn.LayerNorm(d_model)
        self.score_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def encode_scene(self, agents: torch.Tensor, agent_mask: torch.Tensor,
                     map_features: torch.Tensor, map_mask: torch.Tensor
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode scene into contextualized tokens."""
        agent_tokens = self.agent_proj(agents)
        map_tokens = self.map_proj(map_features)

        # Concatenate into scene tokens
        scene_tokens = torch.cat([agent_tokens, map_tokens], dim=1)
        # Padding mask: True = ignore
        padding_mask = torch.cat([~agent_mask, ~map_mask], dim=1)

        # Scene self-attention
        scene_tokens = self.scene_encoder(scene_tokens, src_key_padding_mask=padding_mask)
        return scene_tokens, padding_mask

    def forward(self, trajectory: torch.Tensor,
                agents: torch.Tensor, agent_mask: torch.Tensor,
                map_features: torch.Tensor, map_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            trajectory: (B, T, traj_dim)
            agents: (B, Na, agent_dim)
            agent_mask: (B, Na) True = valid
            map_features: (B, Nm, map_dim)
            map_mask: (B, Nm) True = valid
        Returns:
            score: (B, 1)
        """
        # Embed trajectory with positional encoding
        traj_tokens = self.traj_proj(trajectory) + self.pe[:, :trajectory.shape[1]]

        # Encode scene
        scene_tokens, scene_pad_mask = self.encode_scene(
            agents, agent_mask, map_features, map_mask)

        # Cross-attention: trajectory attends to scene
        for layer in self.cross_layers:
            # Pre-norm cross attention
            q = layer['norm_q'](traj_tokens)
            kv = layer['norm_kv'](scene_tokens)
            attended, _ = layer['attn'](q, kv, kv, key_padding_mask=scene_pad_mask)
            traj_tokens = traj_tokens + attended

            # FFN
            residual = traj_tokens
            traj_tokens = layer['norm_ff'](traj_tokens)
            traj_tokens = residual + layer['ff'](traj_tokens)

        # Pool and score
        pooled = traj_tokens.mean(dim=1)  # (B, d_model)
        pooled = self.output_norm(pooled)
        return self.score_head(pooled)

    def score_candidates(self, candidates: torch.Tensor,
                         agents: torch.Tensor, agent_mask: torch.Tensor,
                         map_features: torch.Tensor, map_mask: torch.Tensor
                         ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Efficiently score K candidates (shares scene encoding).

        Args:
            candidates: (B, K, T, D)
        Returns:
            scores: (B, K), best_idx: (B,)
        """
        B, K, T, D = candidates.shape

        # Encode scene ONCE
        scene_tokens, scene_pad_mask = self.encode_scene(
            agents, agent_mask, map_features, map_mask)

        # Expand scene for K candidates
        scene_exp = scene_tokens.unsqueeze(1).expand(-1, K, -1, -1).reshape(B * K, -1, self.d_model)
        mask_exp = scene_pad_mask.unsqueeze(1).expand(-1, K, -1).reshape(B * K, -1)

        # Embed all candidate trajectories
        cands_flat = candidates.reshape(B * K, T, D)
        traj_tokens = self.traj_proj(cands_flat) + self.pe[:, :T]

        # Cross-attention
        for layer in self.cross_layers:
            q = layer['norm_q'](traj_tokens)
            kv = layer['norm_kv'](scene_exp)
            attended, _ = layer['attn'](q, kv, kv, key_padding_mask=mask_exp)
            traj_tokens = traj_tokens + attended
            residual = traj_tokens
            traj_tokens = layer['norm_ff'](traj_tokens)
            traj_tokens = residual + layer['ff'](traj_tokens)

        pooled = traj_tokens.mean(dim=1)
        pooled = self.output_norm(pooled)
        scores = self.score_head(pooled).reshape(B, K)

        return scores, scores.argmax(dim=-1)


def demo():
    """Demo transformer scorer."""
    print("Transformer Trajectory Scorer Demo")
    print("=" * 45)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TransformerScorer(d_model=256, n_heads=8).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    B, K, T = 4, 64, 16
    candidates = torch.randn(B, K, T, 4, device=device)
    agents = torch.randn(B, 32, 7, device=device)
    agent_mask = torch.ones(B, 32, dtype=torch.bool, device=device)
    agent_mask[:, 20:] = False
    map_feat = torch.randn(B, 64, 5, device=device)
    map_mask = torch.ones(B, 64, dtype=torch.bool, device=device)

    with torch.no_grad():
        scores, best = model.score_candidates(
            candidates, agents, agent_mask, map_feat, map_mask)
    print(f"Scores: {scores.shape}, Best: {best.tolist()}")
    print(f"Range: [{scores.min():.3f}, {scores.max():.3f}]")


if __name__ == '__main__':
    demo()
