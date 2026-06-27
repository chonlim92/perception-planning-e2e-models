"""
GAIA-1 Training Script
=======================
ATTRIBUTION:
- Loss functions: Based on GAIA-1 paper (Hu et al., Wayve 2023)
  - Video tokenizer: Reconstruction + perceptual + VQ commitment (Section 3.1)
  - World model: Cross-entropy on next-token prediction (Section 3.2)
  - Planning: Reward-weighted trajectory selection (Section 3.3)
- Training strategy: Two-phase from paper:
  - Phase 1: Train video tokenizer (VQ-VAE) on driving video frames
  - Phase 2: Train world model transformer on tokenized sequences
- Implementation: Self-implemented in PyTorch (heavily simplified from 9B model)
- Synthetic dataset: Self-implemented for demonstration
"""

import argparse
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm not installed  # [SELF-IMPLEMENTED]
    def tqdm(iterable, **kwargs):
        return iterable

from model import VideoTokenizer, WorldModelTransformer, WorldModelPlanner


# =============================================================================
# Synthetic Dataset
# =============================================================================

class GAIAVideoDataset(Dataset):  # [SELF-IMPLEMENTED]
    """
    Synthetic driving video dataset for GAIA-1 training demonstration.

    Generates:
    - Video sequences (B, T, 3, H, W): sequences of driving camera frames
    - Action sequences (B, T, action_dim): ego vehicle actions [steer, gas, brake]
    - Text descriptions (B, T, text_dim): encoded scene description vectors

    In practice, this would load real driving video data from nuScenes, Waymo, etc.
    """

    def __init__(
        self,
        num_samples: int = 1000,
        seq_length: int = 8,
        image_size: int = 64,
        action_dim: int = 3,
        text_dim: int = 64,
        seed: int = 42,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.image_size = image_size
        self.action_dim = action_dim
        self.text_dim = text_dim

        # Pre-generate synthetic data with deterministic seed  # [SELF-IMPLEMENTED]
        rng = torch.Generator().manual_seed(seed)

        # Generate smooth synthetic video sequences (simulated ego-motion)
        self.videos = []
        self.actions = []
        self.texts = []

        for i in range(num_samples):
            # Create a base scene with smooth temporal evolution
            # Simulate forward motion with gradual scene changes
            base_scene = torch.randn(3, image_size, image_size, generator=rng) * 0.5
            frames = []
            for t in range(seq_length):
                # Simulate temporal change: shift + noise  # [SELF-IMPLEMENTED]
                shift = t * 0.05
                frame = base_scene + shift + torch.randn_like(base_scene) * 0.1
                frame = frame.clamp(-1, 1)
                frames.append(frame)
            self.videos.append(torch.stack(frames))  # (T, 3, H, W)

            # Generate correlated action sequences (smooth driving)
            action_seq = torch.randn(seq_length, action_dim, generator=rng) * 0.3
            # Smooth actions temporally
            for t in range(1, seq_length):
                action_seq[t] = 0.7 * action_seq[t - 1] + 0.3 * action_seq[t]
            action_seq[:, 0] = action_seq[:, 0].clamp(-1, 1)  # steer
            action_seq[:, 1] = action_seq[:, 1].clamp(0, 1)   # gas
            action_seq[:, 2] = action_seq[:, 2].clamp(0, 1)   # brake
            self.actions.append(action_seq)

            # Generate text description embeddings  # [SELF-IMPLEMENTED]
            text_embed = torch.randn(seq_length, text_dim, generator=rng) * 0.1
            self.texts.append(text_embed)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'video': self.videos[idx],      # (T, 3, H, W)
            'actions': self.actions[idx],    # (T, action_dim)
            'text': self.texts[idx],         # (T, text_dim)
        }


# =============================================================================
# Loss Functions
# =============================================================================

class TokenizerLoss(nn.Module):  # [FROM PAPER] Section 3.1
    """
    Combined loss for VQ-VAE video tokenizer training.

    From GAIA-1 paper Section 3.1:
    L_tokenizer = L_recon + lambda_perceptual * L_perceptual + lambda_vq * L_vq

    Where:
    - L_recon: pixel-wise reconstruction loss (L2/MSE)
    - L_perceptual: feature-level perceptual loss (simplified VGG-style)
    - L_vq: VQ commitment + codebook loss from VectorQuantizer
    """

    def __init__(
        self,
        lambda_recon: float = 1.0,
        lambda_perceptual: float = 0.1,
        lambda_vq: float = 1.0,
    ):
        super().__init__()
        self.lambda_recon = lambda_recon
        self.lambda_perceptual = lambda_perceptual
        self.lambda_vq = lambda_vq

        # Simplified perceptual feature extractor  # [SIMPLIFIED]
        # In practice, this would be a pretrained VGG or LPIPS network
        self.perceptual_net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
        )
        # Freeze perceptual network  # [SELF-IMPLEMENTED]
        for param in self.perceptual_net.parameters():
            param.requires_grad = False

    def forward(
        self,
        reconstructed: torch.Tensor,
        target: torch.Tensor,
        vq_loss: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined tokenizer loss.

        Args:
            reconstructed: (B, 3, H, W) reconstructed frames
            target: (B, 3, H, W) original frames
            vq_loss: scalar VQ loss from VectorQuantizer
        Returns:
            Dict with total_loss, recon_loss, perceptual_loss, vq_loss
        """
        # Reconstruction loss (L2)  # [FROM PAPER]
        recon_loss = F.mse_loss(reconstructed, target)

        # Perceptual loss (feature matching)  # [FROM PAPER] / [SIMPLIFIED]
        with torch.no_grad():
            target_features = self.perceptual_net(target)
        recon_features = self.perceptual_net(reconstructed)
        perceptual_loss = F.mse_loss(recon_features, target_features.detach())

        # Total loss  # [FROM PAPER]
        total_loss = (
            self.lambda_recon * recon_loss
            + self.lambda_perceptual * perceptual_loss
            + self.lambda_vq * vq_loss
        )

        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'perceptual_loss': perceptual_loss,
            'vq_loss': vq_loss,
        }


class WorldModelLoss(nn.Module):  # [FROM PAPER] Section 3.2
    """
    Cross-entropy loss for next-token prediction in the world model.

    From GAIA-1 paper Section 3.2:
    The world model is trained to predict the next discrete token in the sequence
    using standard autoregressive cross-entropy loss:

    L_wm = -sum_t log P(z_t | z_{<t}, a_{<t})

    where z_t are frame tokens and a_t are actions.
    """

    def __init__(self, label_smoothing: float = 0.0):  # [FROM PAPER]
        super().__init__()
        self.label_smoothing = label_smoothing
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(
        self,
        logits: torch.Tensor,
        target_tokens: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute cross-entropy loss for next token prediction.

        Args:
            logits: (B, N, num_codes) predicted token logits for next frame
            target_tokens: (B, N) ground truth token indices for next frame
        Returns:
            Dict with total_loss, accuracy, perplexity
        """
        B, N, C = logits.shape

        # Flatten for cross-entropy  # [FROM PAPER]
        logits_flat = logits.reshape(B * N, C)
        targets_flat = target_tokens.reshape(B * N)

        # Cross-entropy loss  # [FROM PAPER]
        loss = self.criterion(logits_flat, targets_flat)

        # Metrics  # [SELF-IMPLEMENTED]
        with torch.no_grad():
            predictions = logits_flat.argmax(dim=-1)
            accuracy = (predictions == targets_flat).float().mean()
            perplexity = torch.exp(loss)

        return {
            'total_loss': loss,
            'accuracy': accuracy,
            'perplexity': perplexity,
        }


class PlanningLoss(nn.Module):  # [FROM PAPER] Section 3.3
    """
    Planning loss: reward-weighted trajectory selection.

    From GAIA-1 paper Section 3.3:
    The planner is trained to select actions that lead to high-reward
    imagined futures. This uses a reward-weighted regression approach:

    L_plan = -sum_k w_k * log pi(a_k | s)

    where w_k = exp(R_k / temperature) / Z is the normalized reward weight
    for the k-th candidate trajectory.
    """

    def __init__(
        self,
        temperature: float = 1.0,
        num_candidates: int = 64,
    ):
        super().__init__()
        self.temperature = temperature  # [FROM PAPER]
        self.num_candidates = num_candidates

    def forward(
        self,
        candidate_actions: torch.Tensor,
        rewards: torch.Tensor,
        action_log_probs: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute reward-weighted planning loss.

        Args:
            candidate_actions: (K, horizon, action_dim) sampled action sequences
            rewards: (K,) reward scores for each candidate trajectory
            action_log_probs: (K,) log probabilities of actions under policy
                              If None, uses uniform (no policy gradient)
        Returns:
            Dict with total_loss, best_reward, mean_reward
        """
        K = rewards.shape[0]

        # Compute reward weights (softmax with temperature)  # [FROM PAPER]
        reward_weights = F.softmax(rewards / self.temperature, dim=0)  # (K,)

        if action_log_probs is not None:
            # Policy gradient with reward weighting  # [FROM PAPER]
            loss = -(reward_weights.detach() * action_log_probs).sum()
        else:
            # Simplified: just return negative expected reward  # [SIMPLIFIED]
            loss = -(reward_weights * rewards).sum()

        with torch.no_grad():
            best_reward = rewards.max()
            mean_reward = rewards.mean()
            best_idx = rewards.argmax()

        return {
            'total_loss': loss,
            'best_reward': best_reward,
            'mean_reward': mean_reward,
            'best_trajectory_idx': best_idx,
        }


# =============================================================================
# Validation Metrics
# =============================================================================

@torch.no_grad()
def compute_psnr(reconstructed: torch.Tensor, target: torch.Tensor) -> float:  # [SELF-IMPLEMENTED]
    """Compute Peak Signal-to-Noise Ratio for reconstruction quality."""
    mse = F.mse_loss(reconstructed, target).item()
    if mse < 1e-10:
        return 100.0
    # Assuming pixel values in [-1, 1], max value range = 2
    psnr = 10 * math.log10(4.0 / mse)  # max_val^2 = 2^2 = 4
    return psnr


@torch.no_grad()
def compute_codebook_utilization(indices: torch.Tensor, num_codes: int) -> float:  # [SELF-IMPLEMENTED]
    """
    Compute codebook utilization: fraction of codes actively used.
    Low utilization indicates codebook collapse (a common VQ-VAE failure mode).
    """
    unique_codes = indices.unique().numel()
    utilization = unique_codes / num_codes
    return utilization


@torch.no_grad()
def compute_fid_proxy(generated_tokens: torch.Tensor, real_tokens: torch.Tensor) -> float:  # [SIMPLIFIED]
    """
    Simplified FID proxy: measures distributional distance between generated
    and real token sequences. Real FID requires an Inception network on decoded frames.

    Here we approximate by comparing token frequency distributions.
    """
    num_codes = max(generated_tokens.max().item(), real_tokens.max().item()) + 1

    # Token frequency histograms
    gen_hist = torch.histc(generated_tokens.float(), bins=num_codes, min=0, max=num_codes - 1)
    real_hist = torch.histc(real_tokens.float(), bins=num_codes, min=0, max=num_codes - 1)

    # Normalize to probability distributions
    gen_dist = gen_hist / gen_hist.sum()
    real_dist = real_hist / real_hist.sum()

    # Use L2 distance as FID proxy (lower is better)
    fid_proxy = torch.sqrt(((gen_dist - real_dist) ** 2).sum()).item()
    return fid_proxy


# =============================================================================
# Training Functions
# =============================================================================

def train_tokenizer(  # [FROM PAPER] Phase 1
    tokenizer: VideoTokenizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
) -> VideoTokenizer:
    """
    Phase 1: Train the VQ-VAE video tokenizer.

    From GAIA-1 paper: The video tokenizer is trained first on individual frames
    to learn a discrete visual vocabulary. This enables the world model to operate
    in a compressed token space rather than raw pixel space.
    """
    print("\n" + "=" * 70)
    print("PHASE 1: Training Video Tokenizer (VQ-VAE)")
    print("=" * 70)

    # Loss function  # [FROM PAPER]
    loss_fn = TokenizerLoss(
        lambda_recon=args.lambda_recon,
        lambda_perceptual=args.lambda_perceptual,
        lambda_vq=args.lambda_vq,
    ).to(device)

    # Optimizer  # [SELF-IMPLEMENTED]
    optimizer = torch.optim.AdamW(
        tokenizer.parameters(),
        lr=args.lr_tokenizer,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.99),
    )

    # Learning rate scheduler  # [SELF-IMPLEMENTED]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs_tokenizer, eta_min=args.lr_tokenizer * 0.01
    )

    # AMP scaler  # [SELF-IMPLEMENTED]
    scaler = GradScaler(enabled=args.use_amp)

    best_val_loss = float('inf')

    for epoch in range(1, args.epochs_tokenizer + 1):
        # --- Training ---
        tokenizer.train()
        train_metrics = train_one_epoch_tokenizer(
            tokenizer, loss_fn, train_loader, optimizer, scaler, device, args, epoch
        )

        # --- Validation ---
        tokenizer.eval()
        val_metrics = validate_tokenizer(tokenizer, loss_fn, val_loader, device, args)

        # --- Scheduler step ---
        scheduler.step()

        # --- Logging ---
        print(
            f"  Epoch {epoch}/{args.epochs_tokenizer} | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"PSNR: {val_metrics['psnr']:.2f} dB | "
            f"Codebook Util: {val_metrics['codebook_util']:.2%} | "
            f"LR: {scheduler.get_last_lr()[0]:.2e}"
        )

        # --- Checkpoint ---  # [SELF-IMPLEMENTED]
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            save_checkpoint(
                tokenizer, optimizer, epoch, val_metrics,
                Path(args.checkpoint_dir) / "tokenizer_best.pt"
            )

    # Save final checkpoint
    save_checkpoint(
        tokenizer, optimizer, args.epochs_tokenizer, val_metrics,
        Path(args.checkpoint_dir) / "tokenizer_final.pt"
    )

    print(f"\n  Tokenizer training complete. Best val loss: {best_val_loss:.4f}")
    return tokenizer


def train_one_epoch_tokenizer(  # [SELF-IMPLEMENTED]
    tokenizer: VideoTokenizer,
    loss_fn: TokenizerLoss,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    args: argparse.Namespace,
    epoch: int,
) -> Dict[str, float]:
    """Train tokenizer for one epoch over all video frames."""
    total_loss = 0.0
    total_recon = 0.0
    total_vq = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"  [Tokenizer] Epoch {epoch}", leave=False)
    for batch in pbar:
        video = batch['video'].to(device)  # (B, T, 3, H, W)
        B, T, C, H, W = video.shape

        # Flatten temporal dimension to train on individual frames  # [FROM PAPER]
        frames = video.reshape(B * T, C, H, W)

        optimizer.zero_grad()

        with autocast(enabled=args.use_amp):
            # Forward pass through tokenizer
            z = tokenizer.encoder(frames)
            quantized, vq_loss, indices = tokenizer.vq(z)
            reconstructed = tokenizer.decoder(quantized)

            # Compute loss
            losses = loss_fn(reconstructed, frames, vq_loss)

        # Backward pass with gradient scaling  # [SELF-IMPLEMENTED]
        scaler.scale(losses['total_loss']).backward()

        # Gradient clipping  # [SELF-IMPLEMENTED]
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(tokenizer.parameters(), args.grad_clip)

        scaler.step(optimizer)
        scaler.update()

        # Accumulate metrics
        total_loss += losses['total_loss'].item()
        total_recon += losses['recon_loss'].item()
        total_vq += losses['vq_loss'].item()
        num_batches += 1

        if hasattr(pbar, 'set_postfix'):
            pbar.set_postfix(
                loss=f"{losses['total_loss'].item():.4f}",
                recon=f"{losses['recon_loss'].item():.4f}",
            )

    return {
        'loss': total_loss / max(num_batches, 1),
        'recon_loss': total_recon / max(num_batches, 1),
        'vq_loss': total_vq / max(num_batches, 1),
    }


@torch.no_grad()
def validate_tokenizer(  # [SELF-IMPLEMENTED]
    tokenizer: VideoTokenizer,
    loss_fn: TokenizerLoss,
    dataloader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, float]:
    """Validate tokenizer: compute losses + reconstruction quality metrics."""
    total_loss = 0.0
    total_psnr = 0.0
    all_indices = []
    num_batches = 0

    for batch in dataloader:
        video = batch['video'].to(device)
        B, T, C, H, W = video.shape
        frames = video.reshape(B * T, C, H, W)

        # Forward pass
        z = tokenizer.encoder(frames)
        quantized, vq_loss, indices = tokenizer.vq(z)
        reconstructed = tokenizer.decoder(quantized)

        # Loss
        losses = loss_fn(reconstructed, frames, vq_loss)
        total_loss += losses['total_loss'].item()

        # PSNR  # [SELF-IMPLEMENTED]
        total_psnr += compute_psnr(reconstructed, frames)

        # Collect indices for codebook utilization
        all_indices.append(indices.cpu())
        num_batches += 1

    # Codebook utilization  # [SELF-IMPLEMENTED]
    all_indices_cat = torch.cat(all_indices, dim=0)
    codebook_util = compute_codebook_utilization(all_indices_cat, tokenizer.vq.num_embeddings)

    return {
        'loss': total_loss / max(num_batches, 1),
        'psnr': total_psnr / max(num_batches, 1),
        'codebook_util': codebook_util,
    }


def train_world_model(  # [FROM PAPER] Phase 2
    world_model: WorldModelTransformer,
    tokenizer: VideoTokenizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
) -> WorldModelTransformer:
    """
    Phase 2: Train the autoregressive world model transformer.

    From GAIA-1 paper: The world model is trained on tokenized video sequences
    with teacher forcing. Given past frame tokens and actions, it predicts the
    next frame's tokens using cross-entropy loss.
    """
    print("\n" + "=" * 70)
    print("PHASE 2: Training World Model Transformer")
    print("=" * 70)

    # Freeze tokenizer during world model training  # [FROM PAPER]
    tokenizer.eval()
    for param in tokenizer.parameters():
        param.requires_grad = False

    # Loss function  # [FROM PAPER]
    loss_fn = WorldModelLoss(label_smoothing=args.label_smoothing)

    # Optimizer  # [SELF-IMPLEMENTED]
    optimizer = torch.optim.AdamW(
        world_model.parameters(),
        lr=args.lr_world_model,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    # Cosine schedule with warmup  # [SELF-IMPLEMENTED]
    total_steps = args.epochs_world_model * len(train_loader)
    warmup_steps = int(0.05 * total_steps)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # AMP scaler
    scaler = GradScaler(enabled=args.use_amp)

    best_val_loss = float('inf')
    global_step = 0

    for epoch in range(1, args.epochs_world_model + 1):
        # --- Training ---
        world_model.train()
        train_metrics, global_step = train_one_epoch_world_model(
            world_model, tokenizer, loss_fn, train_loader,
            optimizer, scheduler, scaler, device, args, epoch, global_step
        )

        # --- Validation ---
        world_model.eval()
        val_metrics = validate_world_model(
            world_model, tokenizer, loss_fn, val_loader, device, args
        )

        # --- Logging ---
        print(
            f"  Epoch {epoch}/{args.epochs_world_model} | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.2%} | "
            f"Val PPL: {val_metrics['perplexity']:.2f} | "
            f"FID Proxy: {val_metrics['fid_proxy']:.4f}"
        )

        # --- Checkpoint ---  # [SELF-IMPLEMENTED]
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            save_checkpoint(
                world_model, optimizer, epoch, val_metrics,
                Path(args.checkpoint_dir) / "world_model_best.pt"
            )

    # Save final
    save_checkpoint(
        world_model, optimizer, args.epochs_world_model, val_metrics,
        Path(args.checkpoint_dir) / "world_model_final.pt"
    )

    print(f"\n  World model training complete. Best val loss: {best_val_loss:.4f}")
    return world_model


def train_one_epoch_world_model(  # [SELF-IMPLEMENTED]
    world_model: WorldModelTransformer,
    tokenizer: VideoTokenizer,
    loss_fn: WorldModelLoss,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: GradScaler,
    device: torch.device,
    args: argparse.Namespace,
    epoch: int,
    global_step: int,
) -> Tuple[Dict[str, float], int]:
    """Train world model for one epoch."""
    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"  [World Model] Epoch {epoch}", leave=False)
    for batch in pbar:
        video = batch['video'].to(device)   # (B, T, 3, H, W)
        actions = batch['actions'].to(device)  # (B, T, action_dim)
        B, T, C, H, W = video.shape

        # Tokenize all frames with frozen tokenizer  # [FROM PAPER]
        with torch.no_grad():
            frames = video.reshape(B * T, C, H, W)
            z = tokenizer.encoder(frames)
            _, _, indices = tokenizer.vq(z)  # (B*T, h, w)
            h, w = indices.shape[1], indices.shape[2]
            # Flatten spatial tokens per frame
            tokens_per_frame = h * w
            frame_tokens = indices.reshape(B, T, tokens_per_frame)  # (B, T, N)

        # Teacher forcing: predict frame t+1 from frames 0..t  # [FROM PAPER]
        input_tokens = frame_tokens[:, :-1]    # (B, T-1, N)
        target_tokens = frame_tokens[:, -1]    # (B, N) - last frame as target
        input_actions = actions[:, :-1]        # (B, T-1, action_dim)

        optimizer.zero_grad()

        with autocast(enabled=args.use_amp):
            # World model forward  # [FROM PAPER]
            logits = world_model(input_tokens, input_actions)  # (B, N, num_codes)
            losses = loss_fn(logits, target_tokens)

        # Backward
        scaler.scale(losses['total_loss']).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(world_model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        # Metrics
        total_loss += losses['total_loss'].item()
        total_acc += losses['accuracy'].item()
        num_batches += 1
        global_step += 1

        if hasattr(pbar, 'set_postfix'):
            pbar.set_postfix(
                loss=f"{losses['total_loss'].item():.4f}",
                acc=f"{losses['accuracy'].item():.2%}",
            )

    return {
        'loss': total_loss / max(num_batches, 1),
        'accuracy': total_acc / max(num_batches, 1),
    }, global_step


@torch.no_grad()
def validate_world_model(  # [SELF-IMPLEMENTED]
    world_model: WorldModelTransformer,
    tokenizer: VideoTokenizer,
    loss_fn: WorldModelLoss,
    dataloader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, float]:
    """Validate world model: loss, accuracy, perplexity, FID proxy."""
    total_loss = 0.0
    total_acc = 0.0
    total_ppl = 0.0
    all_generated = []
    all_real = []
    num_batches = 0

    for batch in dataloader:
        video = batch['video'].to(device)
        actions = batch['actions'].to(device)
        B, T, C, H, W = video.shape

        # Tokenize
        frames = video.reshape(B * T, C, H, W)
        z = tokenizer.encoder(frames)
        _, _, indices = tokenizer.vq(z)
        h, w = indices.shape[1], indices.shape[2]
        tokens_per_frame = h * w
        frame_tokens = indices.reshape(B, T, tokens_per_frame)

        # Predict next frame
        input_tokens = frame_tokens[:, :-1]
        target_tokens = frame_tokens[:, -1]
        input_actions = actions[:, :-1]

        logits = world_model(input_tokens, input_actions)
        losses = loss_fn(logits, target_tokens)

        total_loss += losses['total_loss'].item()
        total_acc += losses['accuracy'].item()
        total_ppl += losses['perplexity'].item()

        # Collect for FID proxy  # [SIMPLIFIED]
        predicted_tokens = logits.argmax(dim=-1)  # (B, N)
        all_generated.append(predicted_tokens.cpu())
        all_real.append(target_tokens.cpu())
        num_batches += 1

    # FID proxy  # [SIMPLIFIED]
    gen_cat = torch.cat(all_generated, dim=0).flatten()
    real_cat = torch.cat(all_real, dim=0).flatten()
    fid_proxy = compute_fid_proxy(gen_cat, real_cat)

    return {
        'loss': total_loss / max(num_batches, 1),
        'accuracy': total_acc / max(num_batches, 1),
        'perplexity': total_ppl / max(num_batches, 1),
        'fid_proxy': fid_proxy,
    }


def train_planner(  # [FROM PAPER] Phase 3 / Section 3.3
    world_model: WorldModelTransformer,
    tokenizer: VideoTokenizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    """
    Phase 3: Train planning by imagination.

    From GAIA-1 paper Section 3.3:
    The planner samples candidate action sequences, uses the world model
    to imagine future outcomes, scores them, and learns to select better actions.

    This phase requires a trained tokenizer (Phase 1) and world model (Phase 2).
    """
    print("\n" + "=" * 70)
    print("PHASE 3: Training Planner (Imagination-Based)")
    print("=" * 70)

    # Freeze tokenizer and world model  # [FROM PAPER]
    tokenizer.eval()
    world_model.eval()
    for param in tokenizer.parameters():
        param.requires_grad = False
    for param in world_model.parameters():
        param.requires_grad = False

    # Simple learned action prior (policy network)  # [SIMPLIFIED]
    # In the full GAIA-1, this would be a more sophisticated policy
    action_prior = nn.Sequential(
        nn.Linear(world_model.d_model, 256),
        nn.GELU(),
        nn.Linear(256, 128),
        nn.GELU(),
        nn.Linear(128, 3 * args.planning_horizon),  # predict full action sequence
    ).to(device)

    # Loss function  # [FROM PAPER]
    planning_loss_fn = PlanningLoss(
        temperature=args.planning_temperature,
        num_candidates=args.num_candidates,
    )

    # Optimizer for action prior  # [SELF-IMPLEMENTED]
    optimizer = torch.optim.Adam(action_prior.parameters(), lr=args.lr_planner)

    # Create planner utility  # [FROM PAPER]
    planner = WorldModelPlanner(
        world_model=world_model,
        tokenizer=tokenizer,
        num_candidates=args.num_candidates,
        horizon=args.planning_horizon,
    )

    best_reward = float('-inf')

    for epoch in range(1, args.epochs_planner + 1):
        action_prior.train()
        epoch_rewards = []
        epoch_losses = []

        pbar = tqdm(train_loader, desc=f"  [Planner] Epoch {epoch}", leave=False)
        for batch in pbar:
            video = batch['video'].to(device)
            B, T, C, H, W = video.shape

            # Get current frame tokens  # [FROM PAPER]
            with torch.no_grad():
                current_frame = video[:, 0]  # (B, 3, H, W)
                z = tokenizer.encoder(current_frame)
                _, _, indices = tokenizer.vq(z)
                h, w = indices.shape[1], indices.shape[2]
                tokens_per_frame = h * w
                current_tokens = indices.reshape(B, 1, tokens_per_frame)

            # Sample candidate actions using prior + noise  # [FROM PAPER]
            # Use first token embedding as context for the action prior
            with torch.no_grad():
                context = world_model.token_embed(current_tokens[:, 0]).mean(dim=1)  # (B, d_model)

            # Generate action proposals  # [SIMPLIFIED]
            action_params = action_prior(context)  # (B, 3*horizon)
            action_mean = action_params.reshape(B, args.planning_horizon, 3)

            # Sample K candidates around the mean  # [FROM PAPER]
            candidates_list = []
            for b in range(B):
                noise = torch.randn(
                    args.num_candidates, args.planning_horizon, 3, device=device
                ) * 0.2
                candidates_b = action_mean[b:b+1].expand(args.num_candidates, -1, -1) + noise
                candidates_b[:, :, 0] = candidates_b[:, :, 0].clamp(-1, 1)
                candidates_b[:, :, 1] = candidates_b[:, :, 1].clamp(0, 1)
                candidates_b[:, :, 2] = candidates_b[:, :, 2].clamp(0, 1)
                candidates_list.append(candidates_b)

            # Imagine futures and score  # [FROM PAPER]
            batch_rewards = []
            for b in range(B):
                candidates_b = candidates_list[b]
                tokens_b = current_tokens[b:b+1].expand(args.num_candidates, -1, -1)

                with torch.no_grad():
                    imagined = world_model.imagine(
                        tokens_b, candidates_b, num_future_frames=args.planning_horizon
                    )
                    rewards = planner._score_imagined_futures(imagined, candidates_b)
                batch_rewards.append(rewards)

            # Compute planning loss with policy gradient  # [FROM PAPER]
            optimizer.zero_grad()
            total_plan_loss = torch.tensor(0.0, device=device)

            for b in range(B):
                rewards = batch_rewards[b]
                # Compute log-probs of candidates under the prior (Gaussian with std=0.2)
                diff = candidates_list[b] - action_mean[b:b+1]
                action_log_probs = -0.5 * (diff / 0.2).pow(2).sum(dim=(1, 2))
                losses = planning_loss_fn(candidates_list[b], rewards, action_log_probs)
                total_plan_loss = total_plan_loss + losses['total_loss']
                epoch_rewards.append(losses['best_reward'].item())

            total_plan_loss = total_plan_loss / B

            # Backward through action prior  # [SIMPLIFIED]
            # We use the loss as a signal to update the prior towards better actions
            total_plan_loss.backward()
            torch.nn.utils.clip_grad_norm_(action_prior.parameters(), args.grad_clip)
            optimizer.step()

            epoch_losses.append(total_plan_loss.item())

            if hasattr(pbar, 'set_postfix'):
                pbar.set_postfix(
                    loss=f"{total_plan_loss.item():.4f}",
                    reward=f"{epoch_rewards[-1]:.4f}",
                )

        # --- Validation: evaluate trajectory quality ---
        val_metrics = validate_planner(
            world_model, tokenizer, action_prior, planner, val_loader, device, args
        )

        mean_train_reward = sum(epoch_rewards) / max(len(epoch_rewards), 1)
        mean_loss = sum(epoch_losses) / max(len(epoch_losses), 1)

        print(
            f"  Epoch {epoch}/{args.epochs_planner} | "
            f"Loss: {mean_loss:.4f} | "
            f"Train Reward: {mean_train_reward:.4f} | "
            f"Val Reward: {val_metrics['mean_reward']:.4f} | "
            f"Val Best: {val_metrics['best_reward']:.4f}"
        )

        # Checkpoint  # [SELF-IMPLEMENTED]
        if val_metrics['mean_reward'] > best_reward:
            best_reward = val_metrics['mean_reward']
            save_checkpoint(
                action_prior, optimizer, epoch, val_metrics,
                Path(args.checkpoint_dir) / "planner_best.pt"
            )

    print(f"\n  Planner training complete. Best val reward: {best_reward:.4f}")


@torch.no_grad()
def validate_planner(  # [SELF-IMPLEMENTED]
    world_model: WorldModelTransformer,
    tokenizer: VideoTokenizer,
    action_prior: nn.Module,
    planner: WorldModelPlanner,
    dataloader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, float]:
    """Validate planner: trajectory quality scores."""
    action_prior.eval()
    all_best_rewards = []
    all_mean_rewards = []

    for batch in dataloader:
        video = batch['video'].to(device)
        B, T, C, H, W = video.shape

        current_frame = video[:, 0]
        z = tokenizer.encoder(current_frame)
        _, _, indices = tokenizer.vq(z)
        h, w = indices.shape[1], indices.shape[2]
        tokens_per_frame = h * w
        current_tokens = indices.reshape(B, 1, tokens_per_frame)

        for b in range(B):
            # Use planner to find best action
            best_actions = planner.plan(
                current_frame[b:b+1], current_tokens[b:b+1]
            )
            # Score the best trajectory  # [SELF-IMPLEMENTED]
            best_reward = (
                best_actions[:, 1].mean() - 0.5 * (best_actions[:, 0] ** 2).mean()
            ).item()
            all_best_rewards.append(best_reward)

            # Also compute mean reward across random candidates
            random_actions = torch.randn(
                args.num_candidates, args.planning_horizon, 3, device=device
            ) * 0.3
            tokens_exp = current_tokens[b:b+1].expand(args.num_candidates, -1, -1)
            imagined = world_model.imagine(
                tokens_exp, random_actions, num_future_frames=args.planning_horizon
            )
            rewards = planner._score_imagined_futures(imagined, random_actions)
            all_mean_rewards.append(rewards.mean().item())

    action_prior.train()

    return {
        'best_reward': sum(all_best_rewards) / max(len(all_best_rewards), 1),
        'mean_reward': sum(all_mean_rewards) / max(len(all_mean_rewards), 1),
    }


# =============================================================================
# Checkpoint Management
# =============================================================================

def save_checkpoint(  # [SELF-IMPLEMENTED]
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict,
    path: Path,
) -> None:
    """Save model checkpoint with metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }
    torch.save(checkpoint, path)
    print(f"    Checkpoint saved: {path}")


def load_checkpoint(  # [SELF-IMPLEMENTED]
    model: nn.Module,
    path: Path,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: torch.device = torch.device('cpu'),
) -> Dict:
    """Load model checkpoint."""
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print(f"    Loaded checkpoint: {path} (epoch {checkpoint['epoch']})")
    return checkpoint


# =============================================================================
# Main Entry Point
# =============================================================================

def parse_args() -> argparse.Namespace:  # [SELF-IMPLEMENTED]
    parser = argparse.ArgumentParser(
        description="GAIA-1 World Model Training Script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Phase selection
    parser.add_argument(
        '--phase', type=str, required=True,
        choices=['tokenizer', 'world_model', 'planner', 'all'],
        help='Training phase: tokenizer (Phase 1), world_model (Phase 2), '
             'planner (Phase 3), or all (sequential)'
    )

    # Model architecture
    parser.add_argument('--latent_dim', type=int, default=32,
                        help='Latent dimension for VQ-VAE encoder')
    parser.add_argument('--num_codes', type=int, default=256,
                        help='Number of VQ codebook entries')
    parser.add_argument('--d_model', type=int, default=256,
                        help='Transformer hidden dimension')
    parser.add_argument('--n_heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=4,
                        help='Number of transformer layers')
    parser.add_argument('--tokens_per_frame', type=int, default=64,
                        help='Number of discrete tokens per frame')
    parser.add_argument('--max_frames', type=int, default=16,
                        help='Maximum sequence length in frames')
    parser.add_argument('--action_dim', type=int, default=3,
                        help='Action dimension [steer, gas, brake]')

    # Dataset
    parser.add_argument('--num_train_samples', type=int, default=800,
                        help='Number of training samples')
    parser.add_argument('--num_val_samples', type=int, default=200,
                        help='Number of validation samples')
    parser.add_argument('--seq_length', type=int, default=8,
                        help='Video sequence length (frames)')
    parser.add_argument('--image_size', type=int, default=64,
                        help='Image resolution (H=W)')

    # Training - Tokenizer
    parser.add_argument('--epochs_tokenizer', type=int, default=20,
                        help='Number of epochs for tokenizer training')
    parser.add_argument('--lr_tokenizer', type=float, default=3e-4,
                        help='Learning rate for tokenizer')
    parser.add_argument('--lambda_recon', type=float, default=1.0,
                        help='Weight for reconstruction loss')
    parser.add_argument('--lambda_perceptual', type=float, default=0.1,
                        help='Weight for perceptual loss')
    parser.add_argument('--lambda_vq', type=float, default=1.0,
                        help='Weight for VQ loss')

    # Training - World Model
    parser.add_argument('--epochs_world_model', type=int, default=30,
                        help='Number of epochs for world model training')
    parser.add_argument('--lr_world_model', type=float, default=1e-4,
                        help='Learning rate for world model')
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                        help='Label smoothing for cross-entropy loss')

    # Training - Planner
    parser.add_argument('--epochs_planner', type=int, default=10,
                        help='Number of epochs for planner training')
    parser.add_argument('--lr_planner', type=float, default=1e-4,
                        help='Learning rate for planner')
    parser.add_argument('--num_candidates', type=int, default=32,
                        help='Number of candidate action sequences for planning')
    parser.add_argument('--planning_horizon', type=int, default=5,
                        help='Planning horizon (future frames)')
    parser.add_argument('--planning_temperature', type=float, default=1.0,
                        help='Temperature for reward-weighted selection')

    # Training - General
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay for AdamW')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping max norm')
    parser.add_argument('--use_amp', action='store_true',
                        help='Use automatic mixed precision (AMP)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='DataLoader num_workers')

    # Checkpointing
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/gaia1',
                        help='Directory to save checkpoints')
    parser.add_argument('--resume_tokenizer', type=str, default=None,
                        help='Path to tokenizer checkpoint to resume from')
    parser.add_argument('--resume_world_model', type=str, default=None,
                        help='Path to world model checkpoint to resume from')

    # Misc
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device: auto, cuda, cpu')

    return parser.parse_args()


def main():
    args = parse_args()

    # --- Setup ---  # [SELF-IMPLEMENTED]
    # Device selection
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    # Reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print("=" * 70)
    print("GAIA-1 World Model Training")
    print("=" * 70)
    print(f"  Phase: {args.phase}")
    print(f"  Device: {device}")
    print(f"  AMP: {args.use_amp}")
    print(f"  Seed: {args.seed}")
    print(f"  Checkpoint dir: {args.checkpoint_dir}")

    # --- Create datasets ---  # [SELF-IMPLEMENTED]
    print("\n  Creating synthetic driving datasets...")
    train_dataset = GAIAVideoDataset(
        num_samples=args.num_train_samples,
        seq_length=args.seq_length,
        image_size=args.image_size,
        action_dim=args.action_dim,
        seed=args.seed,
    )
    val_dataset = GAIAVideoDataset(
        num_samples=args.num_val_samples,
        seq_length=args.seq_length,
        image_size=args.image_size,
        action_dim=args.action_dim,
        seed=args.seed + 1,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
    )

    print(f"  Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    print(f"  Sequence length: {args.seq_length} frames")
    print(f"  Image size: {args.image_size}x{args.image_size}")

    # --- Create models ---  # [SELF-IMPLEMENTED]
    # Compute tokens_per_frame from image_size and encoder stride
    # Encoder has 3 stride-2 conv layers: H/8 x W/8
    spatial_dim = args.image_size // 8
    tokens_per_frame = spatial_dim * spatial_dim

    tokenizer = VideoTokenizer(
        in_channels=3,
        latent_dim=args.latent_dim,
        num_codes=args.num_codes,
    ).to(device)

    world_model = WorldModelTransformer(
        num_codes=args.num_codes,
        action_dim=args.action_dim,
        d_model=args.d_model,
        n_heads=args.n_heads,
        num_layers=args.num_layers,
        tokens_per_frame=tokens_per_frame,
        max_frames=args.max_frames,
    ).to(device)

    tok_params = sum(p.numel() for p in tokenizer.parameters())
    wm_params = sum(p.numel() for p in world_model.parameters())
    print(f"\n  Tokenizer parameters: {tok_params:,}")
    print(f"  World model parameters: {wm_params:,}")
    print(f"  Tokens per frame: {tokens_per_frame} ({spatial_dim}x{spatial_dim})")

    # --- Load checkpoints if resuming ---  # [SELF-IMPLEMENTED]
    if args.resume_tokenizer:
        load_checkpoint(tokenizer, Path(args.resume_tokenizer), device=device)
    if args.resume_world_model:
        load_checkpoint(world_model, Path(args.resume_world_model), device=device)

    # --- Run training phase(s) ---
    start_time = time.time()

    if args.phase in ('tokenizer', 'all'):
        tokenizer = train_tokenizer(
            tokenizer, train_loader, val_loader, args, device
        )

    if args.phase in ('world_model', 'all'):
        # For world_model phase, ensure tokenizer is trained  # [FROM PAPER]
        if args.phase == 'world_model' and args.resume_tokenizer is None:
            # Try to load best tokenizer checkpoint
            tok_ckpt = Path(args.checkpoint_dir) / "tokenizer_best.pt"
            if tok_ckpt.exists():
                load_checkpoint(tokenizer, tok_ckpt, device=device)
                print("  Loaded pretrained tokenizer for world model training.")
            else:
                print("  WARNING: No pretrained tokenizer found. "
                      "World model training may not converge well.")

        world_model = train_world_model(
            world_model, tokenizer, train_loader, val_loader, args, device
        )

    if args.phase in ('planner', 'all'):
        # For planner phase, ensure both tokenizer and world model are trained  # [FROM PAPER]
        if args.phase == 'planner':
            tok_ckpt = Path(args.checkpoint_dir) / "tokenizer_best.pt"
            wm_ckpt = Path(args.checkpoint_dir) / "world_model_best.pt"
            if tok_ckpt.exists() and args.resume_tokenizer is None:
                load_checkpoint(tokenizer, tok_ckpt, device=device)
            if wm_ckpt.exists() and args.resume_world_model is None:
                load_checkpoint(world_model, wm_ckpt, device=device)

        train_planner(
            world_model, tokenizer, train_loader, val_loader, args, device
        )

    elapsed = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"Training complete! Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
