"""
GenAD: Generative End-to-End Autonomous Driving

Uses diffusion models to generate diverse trajectory proposals,
naturally capturing the multi-modal nature of driving decisions.

Key idea: instead of regressing a single trajectory, learn the
DISTRIBUTION of valid trajectories and sample from it.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal embedding for diffusion timestep."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        emb = math.log(10000) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t.unsqueeze(-1) * emb.unsqueeze(0)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class TrajectoryDiffusionModel(nn.Module):
    """
    Diffusion model for trajectory generation.

    Learns to denoise trajectories conditioned on scene context.
    At inference: start from Gaussian noise, iteratively denoise
    to get a valid trajectory.
    """

    def __init__(self, traj_dim: int = 2, num_waypoints: int = 12,
                 scene_dim: int = 256, hidden_dim: int = 512,
                 num_layers: int = 4, num_heads: int = 8):
        super().__init__()
        self.traj_dim = traj_dim
        self.num_waypoints = num_waypoints

        # Timestep embedding
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Trajectory embedding (noisy trajectory input)
        self.traj_proj = nn.Linear(traj_dim, hidden_dim)
        self.traj_pos = nn.Parameter(torch.randn(1, num_waypoints, hidden_dim) * 0.02)

        # Scene conditioning
        self.scene_proj = nn.Linear(scene_dim, hidden_dim)

        # Denoising network (transformer)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                'self_attn': nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True),
                'cross_attn': nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True),
                'ffn': nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.GELU(),
                    nn.Linear(hidden_dim * 4, hidden_dim),
                ),
                'norm1': nn.LayerNorm(hidden_dim),
                'norm2': nn.LayerNorm(hidden_dim),
                'norm3': nn.LayerNorm(hidden_dim),
                'time_mlp': nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, hidden_dim),
                ),
            }))

        # Output: predict noise (epsilon prediction)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_head = nn.Linear(hidden_dim, traj_dim)

    def forward(self, noisy_traj: torch.Tensor, timestep: torch.Tensor,
                scene_context: torch.Tensor) -> torch.Tensor:
        """
        Predict noise in the trajectory.

        Args:
            noisy_traj: (B, T, 2) noisy trajectory
            timestep: (B,) diffusion timestep [0, 1000]
            scene_context: (B, N, scene_dim) scene feature tokens
        Returns:
            predicted_noise: (B, T, 2) noise prediction
        """
        B = noisy_traj.shape[0]

        # Embed timestep
        t_emb = self.time_embed(timestep.float())  # (B, hidden)

        # Embed trajectory
        x = self.traj_proj(noisy_traj) + self.traj_pos  # (B, T, hidden)

        # Project scene context
        scene = self.scene_proj(scene_context)  # (B, N, hidden)

        # Denoising layers
        for layer in self.layers:
            # Add time embedding
            time_scale = layer['time_mlp'](t_emb).unsqueeze(1)
            x = x + time_scale

            # Self-attention
            residual = x
            x = layer['norm1'](x)
            x = residual + layer['self_attn'](x, x, x)[0]

            # Cross-attention to scene
            residual = x
            x = layer['norm2'](x)
            x = residual + layer['cross_attn'](x, scene, scene)[0]

            # FFN
            residual = x
            x = layer['norm3'](x)
            x = residual + layer['ffn'](x)

        x = self.output_norm(x)
        return self.output_head(x)  # (B, T, 2) predicted noise


class GenAD(nn.Module):
    """
    GenAD: Full generative autonomous driving model.

    Components:
    1. Scene encoder: images → scene context features
    2. Diffusion trajectory model: generates diverse trajectories
    3. Trajectory scorer: selects best trajectory
    """

    def __init__(self, scene_dim: int = 256, hidden_dim: int = 512,
                 num_waypoints: int = 12, num_diffusion_steps: int = 100):
        super().__init__()
        self.num_waypoints = num_waypoints
        self.num_steps = num_diffusion_steps

        # Scene encoder (simplified)
        self.scene_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=4, padding=3), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, scene_dim, 3, stride=2, padding=1), nn.ReLU(),
        )

        # Diffusion model
        self.diffusion = TrajectoryDiffusionModel(
            traj_dim=2, num_waypoints=num_waypoints,
            scene_dim=scene_dim, hidden_dim=hidden_dim)

        # Simple trajectory scorer
        self.scorer = nn.Sequential(
            nn.Linear(num_waypoints * 2 + scene_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Noise schedule (linear beta schedule)
        betas = torch.linspace(1e-4, 0.02, num_diffusion_steps)
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1 - alphas_cumprod))

    def encode_scene(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to scene context tokens."""
        feat = self.scene_encoder(images)  # (B, D, h, w)
        return feat.flatten(2).permute(0, 2, 1)  # (B, hw, D)

    def add_noise(self, x0: torch.Tensor, t: torch.Tensor) -> tuple:
        """Add noise to trajectory (forward diffusion)."""
        noise = torch.randn_like(x0)
        sqrt_alpha = self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1)
        noisy = sqrt_alpha * x0 + sqrt_one_minus * noise
        return noisy, noise

    def training_loss(self, expert_traj: torch.Tensor,
                      scene_context: torch.Tensor) -> torch.Tensor:
        """Compute diffusion training loss."""
        B = expert_traj.shape[0]
        t = torch.randint(0, self.num_steps, (B,), device=expert_traj.device)

        noisy_traj, noise = self.add_noise(expert_traj, t)
        predicted_noise = self.diffusion(noisy_traj, t, scene_context)

        return F.mse_loss(predicted_noise, noise)

    @torch.no_grad()
    def sample(self, scene_context: torch.Tensor,
               num_samples: int = 16) -> torch.Tensor:
        """
        Generate trajectory samples via reverse diffusion (DDPM sampling).

        Args:
            scene_context: (B, N, D) scene features
            num_samples: number of trajectories to generate per scene
        Returns:
            trajectories: (B, num_samples, T, 2)
        """
        B = scene_context.shape[0]
        device = scene_context.device

        # Expand scene for multiple samples
        scene_exp = scene_context.unsqueeze(1).expand(-1, num_samples, -1, -1)
        scene_exp = scene_exp.reshape(B * num_samples, *scene_context.shape[1:])

        # Start from noise
        x = torch.randn(B * num_samples, self.num_waypoints, 2, device=device)

        # Reverse diffusion
        for t_idx in reversed(range(self.num_steps)):
            t = torch.full((B * num_samples,), t_idx, device=device, dtype=torch.long)

            predicted_noise = self.diffusion(x, t, scene_exp)

            # DDPM update
            alpha = self.alphas_cumprod[t_idx]
            alpha_prev = self.alphas_cumprod[t_idx - 1] if t_idx > 0 else torch.tensor(1.0, device=device)
            beta = self.betas[t_idx]

            x0_pred = (x - (1 - alpha).sqrt() * predicted_noise) / alpha.sqrt()
            x0_pred = x0_pred.clamp(-10, 10)  # stability

            if t_idx > 0:
                noise = torch.randn_like(x)
                sigma = ((1 - alpha_prev) / (1 - alpha) * beta).sqrt()
                x = alpha_prev.sqrt() * x0_pred + (1 - alpha_prev - sigma**2).clamp(min=0).sqrt() * predicted_noise + sigma * noise
            else:
                x = x0_pred

        return x.reshape(B, num_samples, self.num_waypoints, 2)

    def forward(self, images: torch.Tensor, num_samples: int = 16) -> Dict:
        """
        Full forward: encode scene, generate diverse trajectories, score them.

        Args:
            images: (B, 3, H, W)
            num_samples: trajectories to generate
        Returns:
            dict with trajectories, scores, best_trajectory
        """
        scene_context = self.encode_scene(images)

        if self.training:
            return {'scene_context': scene_context}

        # Generate diverse trajectories
        trajectories = self.sample(scene_context, num_samples)  # (B, K, T, 2)

        # Score trajectories
        B, K, T, D = trajectories.shape
        traj_flat = trajectories.reshape(B, K, T * D)  # (B, K, T*2)
        scene_global = scene_context.mean(dim=1)  # (B, scene_dim)
        scene_exp = scene_global.unsqueeze(1).expand(-1, K, -1)  # (B, K, scene_dim)

        scorer_input = torch.cat([traj_flat, scene_exp], dim=-1)
        scores = self.scorer(scorer_input).squeeze(-1)  # (B, K)

        best_idx = scores.argmax(dim=-1)
        best_traj = torch.stack([trajectories[b, best_idx[b]] for b in range(B)])

        return {
            'trajectories': trajectories,
            'scores': scores,
            'best_trajectory': best_traj,
            'best_idx': best_idx,
        }


def demo():
    print("GenAD: Generative End-to-End Driving Demo")
    print("=" * 50)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = GenAD(scene_dim=128, hidden_dim=256,
                  num_waypoints=6, num_diffusion_steps=20).to(device)
    model.eval()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    B = 2
    images = torch.randn(B, 3, 128, 256, device=device)

    with torch.no_grad():
        out = model(images, num_samples=8)

    print(f"\nGenerated {out['trajectories'].shape[1]} diverse trajectories")
    print(f"Trajectories shape: {out['trajectories'].shape}")
    print(f"Scores: {out['scores'].shape}")
    print(f"Best trajectory: {out['best_trajectory'].shape}")
    print(f"\nKey advantage: captures MULTIPLE valid driving behaviors!")
    print(f"(e.g., lane change vs slow down — both generated as options)")


if __name__ == '__main__':
    demo()
