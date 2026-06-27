"""
DriveVLM: Vision-Language Model for Autonomous Driving

Demonstrates the foundation model paradigm for driving:
- Vision encoder extracts visual tokens from multi-view cameras
- Language model reasons about the scene and generates trajectories
- Chain-of-thought reasoning makes decisions interpretable

This is a simplified reference implementation.
Full VLM models require InternVL/LLaMA backbones (7B+ parameters).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional, List, Tuple


class VisionEncoder(nn.Module):
    """
    Vision encoder that converts multi-view camera images into visual tokens.

    In practice: ViT-Large or InternViT (300M-1B parameters)
    Here: simplified CNN + ViT-style patch embedding
    """

    def __init__(self, img_size: int = 224, patch_size: int = 16,
                 embed_dim: int = 768, num_layers: int = 6, num_heads: int = 12):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # Patch embedding
        self.patch_embed = nn.Conv2d(3, embed_dim, patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches + 1, embed_dim) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 4, batch_first=True,
            activation='gelu')
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (B, num_views, 3, H, W)
        Returns:
            visual_tokens: (B, num_tokens, embed_dim)
        """
        B, N, C, H, W = images.shape

        # Process each view
        all_tokens = []
        for v in range(N):
            patches = self.patch_embed(images[:, v])  # (B, D, h, w)
            tokens = patches.flatten(2).permute(0, 2, 1)  # (B, num_patches, D)

            # Add CLS token
            cls = self.cls_token.expand(B, -1, -1)
            tokens = torch.cat([cls, tokens], dim=1)
            tokens = tokens + self.pos_embed[:, :tokens.shape[1]]

            tokens = self.encoder(tokens)
            tokens = self.norm(tokens)
            all_tokens.append(tokens)

        # Concatenate all view tokens
        visual_tokens = torch.cat(all_tokens, dim=1)  # (B, N*num_patches, D)
        return visual_tokens


class SpatialAdapter(nn.Module):
    """
    Projects visual tokens into a unified spatial (BEV-like) representation.
    Bridges the gap between 2D image patches and 3D driving scene.
    """

    def __init__(self, visual_dim: int = 768, output_dim: int = 4096,
                 num_query_tokens: int = 64):
        super().__init__()
        self.query_tokens = nn.Parameter(
            torch.randn(1, num_query_tokens, visual_dim) * 0.02)

        self.cross_attn = nn.MultiheadAttention(
            visual_dim, num_heads=12, batch_first=True)
        self.norm = nn.LayerNorm(visual_dim)

        self.proj = nn.Sequential(
            nn.Linear(visual_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, visual_tokens: torch.Tensor) -> torch.Tensor:
        """
        Compress visual tokens into fixed-size spatial representation.

        Args:
            visual_tokens: (B, N_tokens, visual_dim)
        Returns:
            spatial_tokens: (B, num_query, output_dim)
        """
        B = visual_tokens.shape[0]
        queries = self.query_tokens.expand(B, -1, -1)

        # Cross-attention: queries attend to visual tokens
        attended, _ = self.cross_attn(queries, visual_tokens, visual_tokens)
        attended = self.norm(attended)

        return self.proj(attended)


class DrivingLLM(nn.Module):
    """
    Language model head for driving reasoning and planning.

    In practice: LLaMA-7B or InternLM-7B
    Here: simplified transformer decoder demonstrating the paradigm
    """

    def __init__(self, vocab_size: int = 32000, embed_dim: int = 4096,
                 num_layers: int = 8, num_heads: int = 16,
                 max_seq_len: int = 512):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        # Token embedding (for text tokens)
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(max_seq_len, embed_dim)

        # Transformer decoder layers
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 4, batch_first=True,
            activation='gelu')
        self.layers = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(embed_dim)

        # Output heads
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

        # Trajectory output head (parallel to language)
        self.trajectory_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 6 * 2),  # 6 waypoints × (x, y)
        )

    def forward(self, visual_tokens: torch.Tensor,
                text_tokens: Optional[torch.Tensor] = None,
                generate_trajectory: bool = True) -> Dict:
        """
        Args:
            visual_tokens: (B, Nv, D) from spatial adapter
            text_tokens: (B, Nt) text token ids (prompt/command)
            generate_trajectory: whether to output trajectory
        Returns:
            dict with logits, trajectory, hidden_states
        """
        B = visual_tokens.shape[0]

        # Build input sequence: [visual_tokens, text_tokens]
        if text_tokens is not None:
            text_embed = self.token_embed(text_tokens)  # (B, Nt, D)
            seq_len = visual_tokens.shape[1] + text_embed.shape[1]
            positions = torch.arange(seq_len, device=visual_tokens.device)
            pos_embed = self.pos_embed(positions).unsqueeze(0)

            input_seq = torch.cat([visual_tokens, text_embed], dim=1)
            input_seq = input_seq + pos_embed[:, :seq_len]
        else:
            seq_len = visual_tokens.shape[1]
            positions = torch.arange(seq_len, device=visual_tokens.device)
            pos_embed = self.pos_embed(positions).unsqueeze(0)
            input_seq = visual_tokens + pos_embed[:, :seq_len]

        # Causal mask
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            seq_len, device=visual_tokens.device)

        # Forward through transformer
        hidden = self.layers(input_seq, mask=causal_mask)
        hidden = self.norm(hidden)

        # Language modeling logits
        logits = self.lm_head(hidden)  # (B, seq_len, vocab_size)

        output = {
            'logits': logits,
            'hidden_states': hidden,
        }

        # Trajectory prediction from final hidden state
        if generate_trajectory:
            final_hidden = hidden[:, -1]  # (B, D) last token
            trajectory = self.trajectory_head(final_hidden)  # (B, 12)
            trajectory = trajectory.reshape(B, 6, 2)  # (B, 6, 2)
            output['trajectory'] = trajectory

        return output


class DriveVLM(nn.Module):
    """
    DriveVLM: Full Vision-Language Model for Driving

    Pipeline:
        Multi-view images → Vision Encoder → Spatial Adapter → LLM → Trajectory + Explanation
    """

    def __init__(self, visual_dim: int = 768, llm_dim: int = 4096,
                 num_query_tokens: int = 64, vocab_size: int = 32000,
                 num_waypoints: int = 6):
        super().__init__()
        self.num_waypoints = num_waypoints

        # Vision
        self.vision_encoder = VisionEncoder(
            img_size=224, patch_size=16, embed_dim=visual_dim,
            num_layers=6, num_heads=12)

        # Spatial adapter (bridge vision → LLM)
        self.spatial_adapter = SpatialAdapter(
            visual_dim=visual_dim, output_dim=llm_dim,
            num_query_tokens=num_query_tokens)

        # Driving LLM
        self.llm = DrivingLLM(
            vocab_size=vocab_size, embed_dim=llm_dim,
            num_layers=8, num_heads=16)

    def forward(self, images: torch.Tensor,
                text_tokens: Optional[torch.Tensor] = None) -> Dict:
        """
        Args:
            images: (B, num_views, 3, H, W) multi-view camera images
            text_tokens: (B, seq_len) optional text prompt tokens
        Returns:
            dict with trajectory, logits, etc.
        """
        # Visual encoding
        visual_tokens = self.vision_encoder(images)  # (B, N_tokens, visual_dim)

        # Spatial adaptation
        spatial_tokens = self.spatial_adapter(visual_tokens)  # (B, num_query, llm_dim)

        # LLM reasoning + planning
        output = self.llm(spatial_tokens, text_tokens, generate_trajectory=True)

        return output

    @torch.no_grad()
    def generate_with_reasoning(self, images: torch.Tensor,
                                prompt_tokens: torch.Tensor,
                                max_new_tokens: int = 100) -> Dict:
        """
        Generate chain-of-thought reasoning + trajectory.

        In practice, this would use autoregressive decoding like GPT.
        Simplified here for demonstration.
        """
        output = self.forward(images, prompt_tokens)
        return {
            'trajectory': output['trajectory'],
            'reasoning_logits': output['logits'],
        }


def compute_drivevlm_loss(output: Dict, gt_trajectory: torch.Tensor,
                          gt_text_tokens: Optional[torch.Tensor] = None) -> Dict:
    """
    Multi-task loss for DriveVLM.

    Combines:
    - Trajectory regression loss (L2/L1 on waypoints)
    - Language modeling loss (cross-entropy on text tokens)
    """
    losses = {}

    # Trajectory loss
    if 'trajectory' in output and gt_trajectory is not None:
        traj_loss = F.l1_loss(output['trajectory'], gt_trajectory)
        losses['trajectory'] = traj_loss

    # Language modeling loss
    if gt_text_tokens is not None and 'logits' in output:
        logits = output['logits']  # (B, seq_len, vocab)
        # Shift for next-token prediction
        shift_logits = logits[:, :-1].contiguous()
        shift_labels = gt_text_tokens[:, 1:].contiguous()
        lm_loss = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.shape[-1]),
            shift_labels.reshape(-1),
            ignore_index=-100)
        losses['language'] = lm_loss

    losses['total'] = sum(losses.values())
    return losses


def demo():
    """Demo DriveVLM."""
    print("DriveVLM: Vision-Language Model for Driving")
    print("=" * 50)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Smaller model for demo (real model would be 7B+ params)
    model = DriveVLM(
        visual_dim=384, llm_dim=512,
        num_query_tokens=32, vocab_size=1000,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Demo model parameters: {num_params:,}")
    print(f"(Real DriveVLM: ~7B parameters)")
    print(f"Device: {device}")

    # Inputs
    B = 2
    images = torch.randn(B, 6, 3, 224, 224, device=device)
    # Simulate prompt: "Drive forward and turn left at intersection"
    prompt = torch.randint(0, 1000, (B, 20), device=device)

    with torch.no_grad():
        output = model(images, prompt)

    print(f"\nInputs:")
    print(f"  Images: {images.shape} (6 cameras)")
    print(f"  Prompt tokens: {prompt.shape}")
    print(f"\nOutputs:")
    print(f"  Trajectory: {output['trajectory'].shape}")
    print(f"  Language logits: {output['logits'].shape}")
    print(f"\n  Planned waypoints (batch 0):")
    for i, wp in enumerate(output['trajectory'][0]):
        print(f"    t={0.5*(i+1):.1f}s: ({wp[0].item():.3f}, {wp[1].item():.3f})")

    # Loss
    gt_traj = torch.randn(B, 6, 2, device=device)
    gt_text = torch.randint(0, 1000, (B, 20), device=device)
    loss = compute_drivevlm_loss(output, gt_traj, gt_text)
    print(f"\n  Loss: {loss['total'].item():.4f} "
          f"(traj={loss['trajectory'].item():.4f}, "
          f"lang={loss['language'].item():.4f})")

    print("\n" + "=" * 50)
    print("PARADIGM: Foundation Model for Driving")
    print("=" * 50)
    print("""
    Stage 1: Pre-train vision encoder (CLIP/InternVL)
             → General visual understanding

    Stage 2: Fine-tune on driving scene descriptions
             → "There's a pedestrian on the left crosswalk"

    Stage 3: Fine-tune for trajectory generation
             → Images + command → waypoints as tokens

    Stage 4: Reinforcement Learning from driving rewards
             → PPO/DPO improves safety and comfort

    Key Advantage: Chain-of-thought reasoning
             → Model explains WHY it makes each decision
             → "Slowing down because pedestrian is crossing"
    """)


if __name__ == '__main__':
    demo()
