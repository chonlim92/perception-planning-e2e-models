"""
GAIA-1 Style World Model for Autonomous Driving

Demonstrates the generative world model paradigm:
1. VQ-VAE tokenizes video frames into discrete tokens
2. Autoregressive transformer predicts next frame tokens
3. Planning uses the world model to imagine outcomes of actions

Simplified implementation for educational purposes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class VectorQuantizer(nn.Module):
    """Vector Quantization layer for VQ-VAE."""

    def __init__(self, num_embeddings: int = 512, embedding_dim: int = 64,
                 commitment_cost: float = 0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.uniform_(self.embeddings.weight, -1/num_embeddings, 1/num_embeddings)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z: (B, D, H, W) continuous latents from encoder
        Returns:
            quantized: (B, D, H, W) quantized latents
            loss: VQ loss
            indices: (B, H, W) codebook indices
        """
        B, D, H, W = z.shape
        z_flat = z.permute(0, 2, 3, 1).reshape(-1, D)  # (B*H*W, D)

        # Find nearest codebook entries
        distances = torch.cdist(z_flat, self.embeddings.weight)  # (B*H*W, num_embed)
        indices = distances.argmin(dim=-1)  # (B*H*W,)

        quantized = self.embeddings(indices).reshape(B, H, W, D).permute(0, 3, 1, 2)

        # Losses
        commitment_loss = F.mse_loss(z, quantized.detach())
        codebook_loss = F.mse_loss(quantized, z.detach())
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss

        # Straight-through estimator
        quantized = z + (quantized - z).detach()

        indices = indices.reshape(B, H, W)
        return quantized, vq_loss, indices


class VideoTokenizer(nn.Module):
    """
    VQ-VAE video tokenizer.
    Encodes frames to discrete tokens, decodes tokens back to frames.
    """

    def __init__(self, in_channels: int = 3, latent_dim: int = 64,
                 num_codes: int = 512):
        super().__init__()

        # Encoder: image → continuous latents
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, latent_dim, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(latent_dim, latent_dim, 3, padding=1),
        )

        # Vector quantizer
        self.vq = VectorQuantizer(num_codes, latent_dim)

        # Decoder: quantized latents → image
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_dim, latent_dim, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(latent_dim, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, in_channels, 4, stride=2, padding=1),
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode image to discrete tokens."""
        z = self.encoder(x)
        quantized, vq_loss, indices = self.vq(z)
        return indices, vq_loss

    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode discrete tokens back to image."""
        quantized = self.vq.embeddings(indices)  # (B, H, W, D)
        quantized = quantized.permute(0, 3, 1, 2)  # (B, D, H, W)
        return self.decoder(quantized)

    def forward(self, x: torch.Tensor) -> Dict:
        z = self.encoder(x)
        quantized, vq_loss, indices = self.vq(z)
        reconstructed = self.decoder(quantized)
        recon_loss = F.mse_loss(reconstructed, x)
        return {
            'reconstructed': reconstructed,
            'indices': indices,
            'vq_loss': vq_loss,
            'recon_loss': recon_loss,
            'total_loss': recon_loss + vq_loss,
        }


class WorldModelTransformer(nn.Module):
    """
    Autoregressive transformer world model.

    Predicts next frame tokens given past frame tokens and actions.
    This is the core "world model" — it learns how the world evolves.
    """

    def __init__(self, num_codes: int = 512, action_dim: int = 3,
                 d_model: int = 512, n_heads: int = 8,
                 num_layers: int = 6, tokens_per_frame: int = 64,
                 max_frames: int = 16):
        super().__init__()
        self.d_model = d_model
        self.tokens_per_frame = tokens_per_frame
        self.num_codes = num_codes

        # Token embeddings
        self.token_embed = nn.Embedding(num_codes, d_model)
        self.pos_embed = nn.Embedding(max_frames * (tokens_per_frame + 1), d_model)

        # Action embedding (continuous → token space)
        self.action_embed = nn.Sequential(
            nn.Linear(action_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Frame separator token
        self.frame_sep = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4, batch_first=True,
            activation='gelu', dropout=0.1)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output head: predict next token
        self.output_head = nn.Linear(d_model, num_codes)

    def forward(self, frame_tokens: torch.Tensor,
                actions: torch.Tensor) -> torch.Tensor:
        """
        Predict next frame tokens.

        Args:
            frame_tokens: (B, num_frames, tokens_per_frame) discrete token indices
            actions: (B, num_frames, action_dim) actions taken at each frame
        Returns:
            logits: (B, tokens_per_frame, num_codes) logits for next frame tokens
        """
        B, T, N = frame_tokens.shape

        # Embed tokens
        token_embeds = self.token_embed(frame_tokens)  # (B, T, N, d_model)

        # Embed actions and interleave
        action_embeds = self.action_embed(actions)  # (B, T, d_model)

        # Build sequence: [frame1_tokens, action1, frame2_tokens, action2, ...]
        seq_parts = []
        for t in range(T):
            seq_parts.append(token_embeds[:, t])  # (B, N, d_model)
            act = action_embeds[:, t:t+1]  # (B, 1, d_model)
            seq_parts.append(act)

        sequence = torch.cat(seq_parts, dim=1)  # (B, T*(N+1), d_model)
        seq_len = sequence.shape[1]

        # Add positional encoding
        positions = torch.arange(seq_len, device=sequence.device)
        sequence = sequence + self.pos_embed(positions).unsqueeze(0)

        # Causal mask
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            seq_len, device=sequence.device)

        # Forward
        hidden = self.transformer(sequence, mask=causal_mask)

        # Take last N positions as prediction for next frame
        last_n = hidden[:, -N:]  # (B, N, d_model)
        logits = self.output_head(last_n)  # (B, N, num_codes)

        return logits

    @torch.no_grad()
    def imagine(self, initial_tokens: torch.Tensor,
                action_sequence: torch.Tensor,
                num_future_frames: int = 5) -> torch.Tensor:
        """
        Imagine future frames given initial observation and planned actions.

        Args:
            initial_tokens: (B, 1, N) initial frame tokens
            action_sequence: (B, num_future, action_dim) planned actions
            num_future_frames: number of frames to imagine
        Returns:
            imagined_tokens: (B, num_future, N) predicted future frame tokens
        """
        B = initial_tokens.shape[0]
        N = initial_tokens.shape[-1]

        all_frames = initial_tokens  # (B, 1, N)
        all_actions = action_sequence[:, :1]  # start with first action

        imagined = []
        for t in range(num_future_frames):
            # Predict next frame
            logits = self.forward(all_frames, all_actions)  # (B, N, num_codes)
            next_tokens = logits.argmax(dim=-1)  # (B, N)
            imagined.append(next_tokens)

            # Append to context
            all_frames = torch.cat([all_frames, next_tokens.unsqueeze(1)], dim=1)
            if t + 1 < action_sequence.shape[1]:
                all_actions = action_sequence[:, :t+2]

        return torch.stack(imagined, dim=1)  # (B, num_future, N)


class WorldModelPlanner:
    """
    Planning via world model imagination.

    Strategy: sample candidate action sequences, imagine their outcomes,
    score the outcomes, select the best action sequence.
    """

    def __init__(self, world_model: WorldModelTransformer,
                 tokenizer: VideoTokenizer,
                 num_candidates: int = 64,
                 horizon: int = 5):
        self.world_model = world_model
        self.tokenizer = tokenizer
        self.num_candidates = num_candidates
        self.horizon = horizon

    @torch.no_grad()
    def plan(self, current_frame: torch.Tensor,
             current_tokens: torch.Tensor) -> torch.Tensor:
        """
        Plan best action sequence by imagining futures.

        Args:
            current_frame: (1, 3, H, W) current camera frame
            current_tokens: (1, 1, N) tokenized current frame
        Returns:
            best_actions: (horizon, 3) best action sequence [steer, gas, brake]
        """
        device = current_tokens.device

        # Sample random candidate action sequences
        candidates = torch.randn(
            self.num_candidates, self.horizon, 3, device=device) * 0.3
        candidates[:, :, 0] = candidates[:, :, 0].clamp(-1, 1)  # steer
        candidates[:, :, 1] = candidates[:, :, 1].clamp(0, 1)   # gas
        candidates[:, :, 2] = candidates[:, :, 2].clamp(0, 1)   # brake

        # Expand current observation for all candidates
        tokens_exp = current_tokens.expand(self.num_candidates, -1, -1)

        # Imagine futures for all candidates
        imagined = self.world_model.imagine(tokens_exp, candidates, self.horizon)
        # imagined: (num_candidates, horizon, N)

        # Score each imagined future (simplified: forward progress proxy)
        scores = self._score_imagined_futures(imagined, candidates)

        # Select best
        best_idx = scores.argmax()
        return candidates[best_idx]  # (horizon, 3)

    def _score_imagined_futures(self, imagined_tokens: torch.Tensor,
                                actions: torch.Tensor) -> torch.Tensor:
        """
        Score imagined futures. In practice this would decode frames and
        check for collisions, progress, etc.

        Simplified: reward smooth actions and penalize extreme steering.
        """
        K = actions.shape[0]
        scores = torch.zeros(K, device=actions.device)

        # Penalize harsh actions
        steer_penalty = (actions[:, :, 0] ** 2).mean(dim=1)
        brake_penalty = actions[:, :, 2].mean(dim=1)
        forward_reward = actions[:, :, 1].mean(dim=1)

        scores = forward_reward - 0.5 * steer_penalty - 0.3 * brake_penalty

        # Diversity bonus based on imagined tokens (crude proxy)
        token_variance = imagined_tokens.float().var(dim=-1).mean(dim=-1)
        scores = scores + 0.1 * token_variance

        return scores


def demo():
    """Demo GAIA-1 world model."""
    print("GAIA-1 Style World Model Demo")
    print("=" * 50)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Small model for demo
    tokenizer = VideoTokenizer(latent_dim=32, num_codes=256).to(device)
    world_model = WorldModelTransformer(
        num_codes=256, d_model=256, n_heads=8,
        num_layers=4, tokens_per_frame=16, max_frames=16).to(device)

    tok_params = sum(p.numel() for p in tokenizer.parameters())
    wm_params = sum(p.numel() for p in world_model.parameters())
    print(f"Tokenizer params: {tok_params:,}")
    print(f"World model params: {wm_params:,}")
    print(f"(Real GAIA-1: ~9B total parameters)")

    # Tokenize a frame
    frame = torch.randn(2, 3, 64, 64, device=device)
    tok_out = tokenizer(frame)
    print(f"\nTokenizer:")
    print(f"  Input: {frame.shape}")
    print(f"  Tokens: {tok_out['indices'].shape} (discrete codes)")
    print(f"  Reconstruction: {tok_out['reconstructed'].shape}")
    print(f"  VQ Loss: {tok_out['vq_loss'].item():.4f}")

    # World model: predict next frame
    B = 2
    frames = torch.randint(0, 256, (B, 4, 16), device=device)  # 4 past frames, 16 tokens each
    actions = torch.randn(B, 4, 3, device=device)

    logits = world_model(frames, actions)
    print(f"\nWorld Model:")
    print(f"  Input frames: {frames.shape} (B, T, tokens_per_frame)")
    print(f"  Actions: {actions.shape}")
    print(f"  Output logits: {logits.shape} (next frame token predictions)")

    # Imagination
    initial = frames[:, :1]  # first frame only
    action_plan = torch.randn(B, 5, 3, device=device)
    imagined = world_model.imagine(initial, action_plan, num_future_frames=5)
    print(f"\n  Imagined future: {imagined.shape} (5 future frames)")

    print(f"\n{'='*50}")
    print("PARADIGM: World Model for Planning")
    print("="*50)
    print("""
    The world model learns: "what happens if I do X?"

    Planning algorithm:
    1. Sample K candidate action sequences
    2. For each: imagine the future using world model
    3. Score each imagined future (safety, progress, comfort)
    4. Execute the best action sequence
    5. Repeat at next timestep

    This is "model-based reinforcement learning" applied to driving.
    The world model IS the environment model.
    """)


if __name__ == '__main__':
    demo()
