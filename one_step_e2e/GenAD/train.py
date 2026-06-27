"""
GenAD Training Script
=====================
ATTRIBUTION:
- Loss functions: Based on GenAD / diffusion planning papers
  - Diffusion loss: MSE between predicted and actual noise (DDPM, Ho et al. 2020)
  - Trajectory scorer loss: BCE + ranking loss (self-implemented)
  - Guidance: Classifier-free guidance during inference (Ho & Salimans 2022)
- Training strategy:
  - Train diffusion model to denoise expert trajectories
  - Optionally train a scorer to rank generated trajectories
- DDPM formulation: Forward process q(x_t|x_0), reverse p(x_{t-1}|x_t)
- Implementation: Self-implemented in PyTorch
- Synthetic dataset: Self-implemented for demonstration (real uses nuScenes/nuPlan)
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
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm is not installed  # [SELF-IMPLEMENTED]
    def tqdm(iterable, **kwargs):
        desc = kwargs.get("desc", "")
        total = kwargs.get("total", None)
        for i, item in enumerate(iterable):
            if total:
                print(f"\r{desc} [{i+1}/{total}]", end="", flush=True)
            yield item
        print()

from model import GenAD, TrajectoryDiffusionModel


# =============================================================================
# Synthetic Dataset
# =============================================================================


class GenADDataset(Dataset):  # [SELF-IMPLEMENTED]
    """
    Synthetic dataset for GenAD training demonstration.

    In a real setting, this would load nuScenes or nuPlan data with:
    - Bird's-eye-view features from a scene encoder
    - Expert (human-driven) trajectory waypoints
    - Map/lane information for context

    Here we generate synthetic driving-like trajectories:
    - Straight driving with slight curves
    - Lane changes (lateral shifts)
    - Turns (circular arcs)

    Each sample provides:
    - scene_features: (scene_dim,) encoded scene context vector
    - expert_trajectory: (num_waypoints, 2) ground truth expert path
    - negative_trajectories: (num_negatives, num_waypoints, 2) poor trajectories for scorer
    """

    def __init__(
        self,
        num_samples: int = 5000,
        num_waypoints: int = 12,
        scene_dim: int = 256,
        num_negatives: int = 4,
        split: str = "train",
        seed: int = 42,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.num_waypoints = num_waypoints
        self.scene_dim = scene_dim
        self.num_negatives = num_negatives

        # Generate synthetic data deterministically  # [SELF-IMPLEMENTED]
        rng = torch.Generator()
        rng.manual_seed(seed if split == "train" else seed + 1000)

        self.scene_features = torch.randn(
            num_samples, scene_dim, generator=rng
        )  # Simulated encoded scene features

        # Generate diverse expert trajectories  # [SELF-IMPLEMENTED]
        self.expert_trajectories = self._generate_expert_trajectories(rng)

        # Generate negative (poor quality) trajectories for scorer training  # [SELF-IMPLEMENTED]
        self.negative_trajectories = self._generate_negative_trajectories(rng)

    def _generate_expert_trajectories(
        self, rng: torch.Generator
    ) -> torch.Tensor:  # [SELF-IMPLEMENTED]
        """
        Generate synthetic expert trajectories mimicking real driving patterns.

        Produces a mix of:
        - Straight driving (40%): forward motion with minor noise
        - Lane changes (30%): lateral sigmoid transitions
        - Turns (30%): circular arc segments
        """
        trajectories = torch.zeros(self.num_samples, self.num_waypoints, 2)
        t = torch.linspace(0, 1, self.num_waypoints).unsqueeze(0)  # (1, T)

        for i in range(self.num_samples):
            pattern = torch.rand(1, generator=rng).item()

            if pattern < 0.4:
                # Straight driving with slight curve  # [SELF-IMPLEMENTED]
                speed = 5.0 + 10.0 * torch.rand(1, generator=rng).item()
                curvature = 0.5 * (torch.rand(1, generator=rng).item() - 0.5)
                x = speed * t.squeeze()
                y = curvature * t.squeeze() ** 2 * speed
            elif pattern < 0.7:
                # Lane change (sigmoid lateral movement)  # [SELF-IMPLEMENTED]
                speed = 5.0 + 8.0 * torch.rand(1, generator=rng).item()
                direction = 1.0 if torch.rand(1, generator=rng).item() > 0.5 else -1.0
                lane_width = 3.5  # standard lane width in meters
                x = speed * t.squeeze()
                # Sigmoid lane change profile
                y = direction * lane_width * torch.sigmoid(10.0 * (t.squeeze() - 0.5))
            else:
                # Turn (circular arc)  # [SELF-IMPLEMENTED]
                radius = 15.0 + 20.0 * torch.rand(1, generator=rng).item()
                angle_range = (
                    0.3 + 0.7 * torch.rand(1, generator=rng).item()
                )  # radians
                direction = 1.0 if torch.rand(1, generator=rng).item() > 0.5 else -1.0
                angles = direction * angle_range * t.squeeze()
                x = radius * torch.sin(angles)
                y = radius * (1 - torch.cos(angles))

            trajectories[i, :, 0] = x
            trajectories[i, :, 1] = y

        # Add small noise to make trajectories more realistic  # [SELF-IMPLEMENTED]
        noise = 0.1 * torch.randn_like(trajectories)
        trajectories = trajectories + noise

        return trajectories

    def _generate_negative_trajectories(
        self, rng: torch.Generator
    ) -> torch.Tensor:  # [SELF-IMPLEMENTED]
        """
        Generate poor-quality trajectories that violate driving constraints.

        These serve as negative examples for the trajectory scorer:
        - Large random deviations from expert path
        - Physically implausible trajectories (sharp turns, reversals)
        """
        negatives = torch.zeros(
            self.num_samples, self.num_negatives, self.num_waypoints, 2
        )

        for i in range(self.num_samples):
            for j in range(self.num_negatives):
                # Start from expert and add large perturbations  # [SELF-IMPLEMENTED]
                neg = self.expert_trajectories[i].clone()
                perturbation_type = torch.rand(1, generator=rng).item()

                if perturbation_type < 0.33:
                    # Large random offset
                    neg += 3.0 * torch.randn(self.num_waypoints, 2)
                elif perturbation_type < 0.66:
                    # Sudden direction reversal (physically implausible)
                    reverse_point = torch.randint(
                        2, self.num_waypoints - 2, (1,), generator=rng
                    ).item()
                    neg[reverse_point:] = neg[reverse_point - 1].unsqueeze(
                        0
                    ) - 0.5 * (neg[reverse_point:] - neg[reverse_point - 1].unsqueeze(0))
                else:
                    # Extreme curvature (uncomfortable/unsafe)
                    neg[:, 1] += 5.0 * torch.sin(
                        torch.linspace(0, 4 * math.pi, self.num_waypoints)
                    )

                negatives[i, j] = neg

        return negatives

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "scene_features": self.scene_features[idx],  # (scene_dim,)
            "expert_trajectory": self.expert_trajectories[idx],  # (T, 2)
            "negative_trajectories": self.negative_trajectories[
                idx
            ],  # (num_negatives, T, 2)
        }


# =============================================================================
# Loss Functions
# =============================================================================


class DiffusionLoss(nn.Module):  # [FROM PAPER]
    """
    DDPM Diffusion Training Loss (Ho et al. 2020, "Denoising Diffusion
    Probabilistic Models").

    The DDPM training objective is elegantly simple:
    -----------------------------------------------
    Given a clean trajectory x_0, the forward process adds Gaussian noise:

        q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)

    We train a neural network epsilon_theta to predict the noise:

        L_simple = E_{t, x_0, epsilon} [ || epsilon - epsilon_theta(x_t, t) ||^2 ]

    This is equivalent (up to weighting) to maximizing the variational lower
    bound on log p(x_0). The "simple" unweighted version works better in
    practice (Ho et al. 2020, Section 3.4).

    At inference, we reverse the process: starting from x_T ~ N(0, I),
    iteratively denoise using the trained model to recover x_0.
    """

    def __init__(self, loss_type: str = "mse"):
        super().__init__()
        self.loss_type = loss_type  # [FROM PAPER] "mse" is the standard DDPM objective

    def forward(
        self,
        predicted_noise: torch.Tensor,
        target_noise: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute the simple diffusion loss.  # [FROM PAPER]

        L = || epsilon - epsilon_theta(x_t, t, c) ||^2

        where:
            epsilon: the actual noise added during forward diffusion
            epsilon_theta: the model's noise prediction
            x_t: the noisy trajectory at timestep t
            c: conditioning (scene context)

        Args:
            predicted_noise: (B, T, 2) model output (predicted noise)
            target_noise: (B, T, 2) actual noise that was added
            timesteps: (B,) optional, for timestep-weighted loss variants
        Returns:
            scalar loss value
        """
        if self.loss_type == "mse":
            # Standard DDPM objective: simple MSE  # [FROM PAPER]
            loss = F.mse_loss(predicted_noise, target_noise)
        elif self.loss_type == "l1":
            # L1 variant (sometimes used for sharper predictions)  # [SELF-IMPLEMENTED]
            loss = F.l1_loss(predicted_noise, target_noise)
        elif self.loss_type == "huber":
            # Huber loss for robustness to outliers  # [SELF-IMPLEMENTED]
            loss = F.smooth_l1_loss(predicted_noise, target_noise)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        return loss


class ScorerLoss(nn.Module):  # [SELF-IMPLEMENTED]
    """
    Trajectory Scorer Loss.

    Trains the scorer network to distinguish good trajectories (expert) from
    poor ones (generated/negative). Combines:
    1. Binary Cross-Entropy: expert=1 (good), negative=0 (bad)
    2. Ranking Loss: ensure expert scores higher than any negative

    This is inspired by the GenAD paper's trajectory selection mechanism,
    where a learned scorer picks the best trajectory from multiple proposals.
    """

    def __init__(self, ranking_weight: float = 0.5, margin: float = 1.0):
        super().__init__()
        self.ranking_weight = ranking_weight  # [SELF-IMPLEMENTED]
        self.margin = margin  # [SELF-IMPLEMENTED]

    def forward(
        self,
        expert_scores: torch.Tensor,
        negative_scores: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined scorer loss.  # [SELF-IMPLEMENTED]

        Args:
            expert_scores: (B,) scores for expert trajectories (should be high)
            negative_scores: (B, K) scores for negative trajectories (should be low)
        Returns:
            dict with 'total', 'bce', 'ranking' losses
        """
        B = expert_scores.shape[0]
        K = negative_scores.shape[1]

        # BCE Loss: expert=1, negatives=0  # [SELF-IMPLEMENTED]
        expert_targets = torch.ones_like(expert_scores)
        negative_targets = torch.zeros_like(negative_scores)

        bce_expert = F.binary_cross_entropy_with_logits(expert_scores, expert_targets)
        bce_negative = F.binary_cross_entropy_with_logits(
            negative_scores, negative_targets
        )
        bce_loss = (bce_expert + bce_negative) / 2.0

        # Ranking Loss: expert should score higher than all negatives by margin  # [SELF-IMPLEMENTED]
        # For each (expert, negative) pair: max(0, margin - (expert - negative))
        expert_expanded = expert_scores.unsqueeze(1).expand(-1, K)  # (B, K)
        ranking_loss = F.relu(
            self.margin - (expert_expanded - negative_scores)
        ).mean()

        total = bce_loss + self.ranking_weight * ranking_loss

        return {
            "total": total,
            "bce": bce_loss,
            "ranking": ranking_loss,
        }


class GuidanceLoss(nn.Module):  # [FROM PAPER]
    """
    Classifier-Free Guidance Training Loss (Ho & Salimans 2022).

    During training, randomly drop the conditioning with probability p_uncond.
    This trains the model to work both conditionally and unconditionally,
    enabling classifier-free guidance at inference time:

        epsilon_guided = epsilon_uncond + w * (epsilon_cond - epsilon_uncond)

    where w > 1 amplifies the effect of conditioning (scene context).
    This makes generated trajectories more faithful to the scene.
    """

    def __init__(self, p_uncond: float = 0.1):
        """
        Args:
            p_uncond: probability of dropping condition during training  # [FROM PAPER]
        """
        super().__init__()
        self.p_uncond = p_uncond

    def get_conditioning_mask(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Generate a random mask for classifier-free guidance training.  # [FROM PAPER]

        Returns a boolean mask where True means "drop conditioning" (use zeros).
        With probability p_uncond, the condition is replaced with zeros,
        teaching the model to generate unconditionally as well.
        """
        mask = torch.rand(batch_size, device=device) < self.p_uncond
        return mask

    def apply_conditioning_dropout(
        self, scene_context: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Zero out scene context for masked samples.  # [FROM PAPER]

        This is the "dropout" in classifier-free guidance:
        randomly replacing the conditioning signal with null (zeros)
        so the model learns p(x) as well as p(x|c).
        """
        # mask shape: (B,), scene_context shape: (B, N, D) or (B, D)
        if scene_context.dim() == 3:
            mask_expanded = mask.unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)
        else:
            mask_expanded = mask.unsqueeze(-1)  # (B, 1)

        # Where mask is True, replace with zeros  # [FROM PAPER]
        context_dropped = scene_context * (~mask_expanded).float()
        return context_dropped


# =============================================================================
# Training Functions
# =============================================================================


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:  # [SELF-IMPLEMENTED]
    """
    Cosine annealing learning rate scheduler with linear warmup.

    - Warmup: linearly increase LR from 0 to base_lr over num_warmup_steps
    - Cosine decay: smoothly decrease LR from base_lr to min_lr_ratio * base_lr
    """

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            # Linear warmup  # [SELF-IMPLEMENTED]
            return float(current_step) / float(max(1, num_warmup_steps))
        # Cosine decay  # [SELF-IMPLEMENTED]
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


def train_diffusion(
    model: GenAD,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    diffusion_loss_fn: DiffusionLoss,
    guidance_loss_fn: GuidanceLoss,
    device: torch.device,
    epoch: int,
    use_amp: bool = True,
    grad_clip: float = 1.0,
) -> Dict[str, float]:  # [SELF-IMPLEMENTED]
    """
    Train the diffusion denoising model for one epoch.

    Training procedure (from DDPM, Ho et al. 2020):
    ------------------------------------------------
    1. Sample a batch of expert trajectories x_0
    2. Sample random timesteps t ~ Uniform(0, T)
    3. Sample noise epsilon ~ N(0, I)
    4. Compute noisy trajectory: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1-alpha_bar_t) * epsilon
    5. Predict noise: epsilon_hat = model(x_t, t, scene_context)
    6. Minimize: L = || epsilon - epsilon_hat ||^2

    With classifier-free guidance (Ho & Salimans 2022):
    - Randomly drop scene_context with probability p_uncond
    - This enables guided sampling at inference: amplify condition influence
    """
    model.train()
    scaler = GradScaler(enabled=use_amp)

    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"[Diffusion] Epoch {epoch}")
    for batch in pbar:
        scene_features = batch["scene_features"].to(device)  # (B, scene_dim)
        expert_traj = batch["expert_trajectory"].to(device)  # (B, T, 2)
        B = expert_traj.shape[0]

        # Expand scene features to simulate token sequence  # [SELF-IMPLEMENTED]
        # In real GenAD, scene_context comes from BEV encoder with spatial tokens
        scene_context = scene_features.unsqueeze(1).expand(
            -1, 4, -1
        )  # (B, 4, scene_dim) - 4 synthetic spatial tokens

        # Apply classifier-free guidance dropout  # [FROM PAPER]
        cfg_mask = guidance_loss_fn.get_conditioning_mask(B, device)
        scene_context_dropped = guidance_loss_fn.apply_conditioning_dropout(
            scene_context, cfg_mask
        )

        optimizer.zero_grad()

        with autocast(enabled=use_amp):
            # Sample random timesteps  # [FROM PAPER]
            t = torch.randint(0, model.num_steps, (B,), device=device)

            # Forward diffusion: add noise to expert trajectory  # [FROM PAPER]
            # q(x_t | x_0) = N(sqrt(alpha_bar_t) * x_0, (1-alpha_bar_t) * I)
            noisy_traj, noise = model.add_noise(expert_traj, t)

            # Predict noise with (possibly dropped) conditioning  # [FROM PAPER]
            predicted_noise = model.diffusion(
                noisy_traj, t, scene_context_dropped
            )

            # Compute loss: L = || epsilon - epsilon_theta(x_t, t, c) ||^2  # [FROM PAPER]
            loss = diffusion_loss_fn(predicted_noise, noise, t)

        # Backward pass with AMP scaling  # [SELF-IMPLEMENTED]
        scaler.scale(loss).backward()

        # Gradient clipping for training stability  # [SELF-IMPLEMENTED]
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()
        num_batches += 1

        pbar.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
            }
        )

    avg_loss = total_loss / max(num_batches, 1)
    return {"diffusion_loss": avg_loss}


def train_scorer(
    model: GenAD,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scorer_loss_fn: ScorerLoss,
    device: torch.device,
    epoch: int,
    use_amp: bool = True,
    grad_clip: float = 1.0,
) -> Dict[str, float]:  # [SELF-IMPLEMENTED]
    """
    Train the trajectory scorer network.

    The scorer learns to distinguish expert trajectories (positive) from
    poor-quality ones (negative). During inference, it selects the best
    trajectory from K diffusion-generated proposals.

    Training procedure:
    1. Encode scene context
    2. Compute score for expert trajectory (should be high)
    3. Compute scores for negative trajectories (should be low)
    4. Minimize BCE + ranking loss
    """
    model.train()
    scaler = GradScaler(enabled=use_amp)

    total_loss = 0.0
    total_bce = 0.0
    total_ranking = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"[Scorer] Epoch {epoch}")
    for batch in pbar:
        scene_features = batch["scene_features"].to(device)  # (B, scene_dim)
        expert_traj = batch["expert_trajectory"].to(device)  # (B, T, 2)
        negative_trajs = batch["negative_trajectories"].to(
            device
        )  # (B, K_neg, T, 2)

        B = expert_traj.shape[0]
        K_neg = negative_trajs.shape[1]
        T = expert_traj.shape[1]

        optimizer.zero_grad()

        with autocast(enabled=use_amp):
            # Flatten trajectories for scorer input  # [SELF-IMPLEMENTED]
            expert_flat = expert_traj.reshape(B, T * 2)  # (B, T*2)
            neg_flat = negative_trajs.reshape(B, K_neg, T * 2)  # (B, K_neg, T*2)

            # Scene context (global pooled)  # [SELF-IMPLEMENTED]
            scene_global = scene_features  # (B, scene_dim)

            # Score expert trajectory  # [SELF-IMPLEMENTED]
            expert_input = torch.cat(
                [expert_flat, scene_global], dim=-1
            )  # (B, T*2 + scene_dim)
            expert_scores = model.scorer(expert_input).squeeze(-1)  # (B,)

            # Score negative trajectories  # [SELF-IMPLEMENTED]
            scene_exp = scene_global.unsqueeze(1).expand(
                -1, K_neg, -1
            )  # (B, K_neg, scene_dim)
            neg_input = torch.cat(
                [neg_flat, scene_exp], dim=-1
            )  # (B, K_neg, T*2 + scene_dim)
            neg_scores = model.scorer(
                neg_input.reshape(B * K_neg, -1)
            ).squeeze(-1)
            neg_scores = neg_scores.reshape(B, K_neg)  # (B, K_neg)

            # Compute scorer loss  # [SELF-IMPLEMENTED]
            loss_dict = scorer_loss_fn(expert_scores, neg_scores)
            loss = loss_dict["total"]

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.scorer.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_bce += loss_dict["bce"].item()
        total_ranking += loss_dict["ranking"].item()
        num_batches += 1

        pbar.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
                "bce": f"{loss_dict['bce'].item():.4f}",
                "rank": f"{loss_dict['ranking'].item():.4f}",
            }
        )

    n = max(num_batches, 1)
    return {
        "scorer_loss": total_loss / n,
        "scorer_bce": total_bce / n,
        "scorer_ranking": total_ranking / n,
    }


def train_one_epoch(
    model: GenAD,
    train_loader: DataLoader,
    diffusion_optimizer: torch.optim.Optimizer,
    scorer_optimizer: torch.optim.Optimizer,
    diffusion_scheduler: LambdaLR,
    diffusion_loss_fn: DiffusionLoss,
    guidance_loss_fn: GuidanceLoss,
    scorer_loss_fn: ScorerLoss,
    device: torch.device,
    epoch: int,
    use_amp: bool = True,
    grad_clip: float = 1.0,
    train_scorer_flag: bool = True,
) -> Dict[str, float]:  # [SELF-IMPLEMENTED]
    """
    Train both diffusion model and scorer for one epoch.

    Two-phase training per epoch:
    Phase 1: Train diffusion denoising model (main objective)
    Phase 2: Train trajectory scorer (auxiliary objective)
    """
    # Phase 1: Train diffusion model  # [FROM PAPER]
    diffusion_metrics = train_diffusion(
        model=model,
        dataloader=train_loader,
        optimizer=diffusion_optimizer,
        scheduler=diffusion_scheduler,
        diffusion_loss_fn=diffusion_loss_fn,
        guidance_loss_fn=guidance_loss_fn,
        device=device,
        epoch=epoch,
        use_amp=use_amp,
        grad_clip=grad_clip,
    )

    metrics = {**diffusion_metrics}

    # Phase 2: Train scorer  # [SELF-IMPLEMENTED]
    if train_scorer_flag:
        scorer_metrics = train_scorer(
            model=model,
            dataloader=train_loader,
            optimizer=scorer_optimizer,
            scorer_loss_fn=scorer_loss_fn,
            device=device,
            epoch=epoch,
            use_amp=use_amp,
            grad_clip=grad_clip,
        )
        metrics.update(scorer_metrics)

    return metrics


# =============================================================================
# Inference / Sampling
# =============================================================================


@torch.no_grad()
def sample_trajectories(
    model: GenAD,
    scene_context: torch.Tensor,
    num_samples: int = 16,
    guidance_scale: float = 2.0,
    use_guidance: bool = True,
) -> torch.Tensor:  # [FROM PAPER]
    """
    Generate trajectories via DDPM reverse process with optional classifier-free guidance.

    DDPM Reverse Process (Ho et al. 2020):
    ----------------------------------------
    Starting from x_T ~ N(0, I), iteratively denoise:

        p(x_{t-1} | x_t) = N(x_{t-1}; mu_theta(x_t, t), sigma_t^2 * I)

    where:
        mu_theta = (1/sqrt(alpha_t)) * (x_t - (beta_t/sqrt(1-alpha_bar_t)) * epsilon_theta(x_t,t))

    Classifier-Free Guidance (Ho & Salimans 2022):
    -----------------------------------------------
    At inference, amplify the effect of conditioning:

        epsilon_guided = epsilon_uncond + w * (epsilon_cond - epsilon_uncond)

    where w > 1 steers generation toward the conditioned distribution.
    This makes trajectories more consistent with the observed scene.

    Args:
        model: trained GenAD model
        scene_context: (B, N, D) encoded scene features
        num_samples: K trajectories to generate per scene
        guidance_scale: w, the classifier-free guidance weight (w=1 means no guidance)
        use_guidance: whether to apply classifier-free guidance
    Returns:
        trajectories: (B, num_samples, T, 2) generated trajectory proposals
    """
    model.eval()
    B = scene_context.shape[0]
    device = scene_context.device

    # Expand scene for multiple samples  # [FROM PAPER]
    scene_exp = scene_context.unsqueeze(1).expand(-1, num_samples, -1, -1)
    scene_exp = scene_exp.reshape(B * num_samples, *scene_context.shape[1:])

    # Null conditioning for guidance (all zeros)  # [FROM PAPER]
    null_context = torch.zeros_like(scene_exp)

    # Start from pure Gaussian noise: x_T ~ N(0, I)  # [FROM PAPER]
    x = torch.randn(
        B * num_samples, model.num_waypoints, 2, device=device
    )

    # Reverse diffusion loop  # [FROM PAPER]
    for t_idx in reversed(range(model.num_steps)):
        t = torch.full(
            (B * num_samples,), t_idx, device=device, dtype=torch.long
        )

        # Conditional noise prediction  # [FROM PAPER]
        noise_cond = model.diffusion(x, t, scene_exp)

        if use_guidance and guidance_scale != 1.0:
            # Unconditional noise prediction  # [FROM PAPER]
            noise_uncond = model.diffusion(x, t, null_context)

            # Classifier-free guidance interpolation  # [FROM PAPER]
            # epsilon_guided = epsilon_uncond + w * (epsilon_cond - epsilon_uncond)
            predicted_noise = noise_uncond + guidance_scale * (
                noise_cond - noise_uncond
            )
        else:
            predicted_noise = noise_cond

        # DDPM update step  # [FROM PAPER]
        alpha = model.alphas_cumprod[t_idx]
        alpha_prev = (
            model.alphas_cumprod[t_idx - 1]
            if t_idx > 0
            else torch.tensor(1.0, device=device)
        )
        beta = model.betas[t_idx]

        # Predict x_0 from x_t and predicted noise  # [FROM PAPER]
        # x_0_hat = (x_t - sqrt(1-alpha_bar_t) * epsilon) / sqrt(alpha_bar_t)
        x0_pred = (x - (1 - alpha).sqrt() * predicted_noise) / alpha.sqrt()
        x0_pred = x0_pred.clamp(-10, 10)  # [SELF-IMPLEMENTED] stability clamp

        if t_idx > 0:
            # Sample x_{t-1} with noise  # [FROM PAPER]
            noise = torch.randn_like(x)
            sigma = ((1 - alpha_prev) / (1 - alpha) * beta).sqrt()
            x = (
                alpha_prev.sqrt() * x0_pred
                + (1 - alpha_prev - sigma**2).sqrt() * predicted_noise
                + sigma * noise
            )
        else:
            # Final step: no noise  # [FROM PAPER]
            x = x0_pred

    return x.reshape(B, num_samples, model.num_waypoints, 2)


@torch.no_grad()
def score_and_select(
    model: GenAD,
    trajectories: torch.Tensor,
    scene_features: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # [SELF-IMPLEMENTED]
    """
    Use the trained scorer to select the best trajectory from K proposals.

    The scorer evaluates each trajectory in the context of the scene and
    returns the highest-scoring one as the final planning output.

    Args:
        model: trained GenAD model with scorer
        trajectories: (B, K, T, 2) generated trajectory proposals
        scene_features: (B, scene_dim) scene context (global vector)
    Returns:
        best_trajectory: (B, T, 2) selected best trajectory
        scores: (B, K) all scores
        best_indices: (B,) index of best trajectory per batch
    """
    model.eval()
    B, K, T, D = trajectories.shape

    # Flatten trajectories  # [SELF-IMPLEMENTED]
    traj_flat = trajectories.reshape(B, K, T * D)  # (B, K, T*2)

    # Expand scene features  # [SELF-IMPLEMENTED]
    scene_exp = scene_features.unsqueeze(1).expand(-1, K, -1)  # (B, K, scene_dim)

    # Concatenate for scorer input  # [SELF-IMPLEMENTED]
    scorer_input = torch.cat([traj_flat, scene_exp], dim=-1)  # (B, K, T*2+scene_dim)

    # Score all trajectories  # [SELF-IMPLEMENTED]
    scores = model.scorer(scorer_input).squeeze(-1)  # (B, K)

    # Select best  # [SELF-IMPLEMENTED]
    best_indices = scores.argmax(dim=-1)  # (B,)
    best_trajectory = torch.stack(
        [trajectories[b, best_indices[b]] for b in range(B)]
    )  # (B, T, 2)

    return best_trajectory, scores, best_indices


# =============================================================================
# Validation Metrics
# =============================================================================


@torch.no_grad()
def compute_fde(
    predicted: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:  # [SELF-IMPLEMENTED]
    """
    Final Displacement Error: L2 distance at the last waypoint.

    FDE = || predicted[-1] - target[-1] ||_2

    This measures how accurately the trajectory reaches the correct endpoint.
    Standard metric in motion forecasting (Alahi et al. 2016).
    """
    # predicted: (B, T, 2) or (B, K, T, 2)
    # target: (B, T, 2)
    if predicted.dim() == 4:
        # Multiple samples: compute for each
        target_exp = target.unsqueeze(1).expand_as(predicted)
        fde = torch.norm(
            predicted[:, :, -1, :] - target_exp[:, :, -1, :], dim=-1
        )  # (B, K)
        return fde
    else:
        fde = torch.norm(predicted[:, -1, :] - target[:, -1, :], dim=-1)  # (B,)
        return fde


@torch.no_grad()
def compute_ade(
    predicted: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:  # [SELF-IMPLEMENTED]
    """
    Average Displacement Error: mean L2 distance across all waypoints.

    ADE = (1/T) * sum_t || predicted[t] - target[t] ||_2

    This measures the overall trajectory accuracy, averaging over all timesteps.
    Standard metric in motion forecasting (Alahi et al. 2016).
    """
    if predicted.dim() == 4:
        target_exp = target.unsqueeze(1).expand_as(predicted)
        ade = torch.norm(predicted - target_exp, dim=-1).mean(dim=-1)  # (B, K)
        return ade
    else:
        ade = torch.norm(predicted - target, dim=-1).mean(dim=-1)  # (B,)
        return ade


@torch.no_grad()
def compute_diversity(
    trajectories: torch.Tensor,
) -> torch.Tensor:  # [SELF-IMPLEMENTED]
    """
    Trajectory diversity: average pairwise L2 distance between generated samples.

    Diversity = (2 / K(K-1)) * sum_{i<j} || traj_i - traj_j ||_2

    Higher diversity indicates the model captures multi-modal behavior
    (e.g., both lane-change and lane-keep options). This is a key advantage
    of diffusion-based planners over single-trajectory regression.
    """
    # trajectories: (B, K, T, 2)
    B, K, T, D = trajectories.shape

    if K < 2:
        return torch.zeros(B, device=trajectories.device)

    # Compute pairwise distances  # [SELF-IMPLEMENTED]
    # Flatten spatial dims for distance computation
    traj_flat = trajectories.reshape(B, K, T * D)  # (B, K, T*2)

    # Pairwise L2 distance between all trajectory pairs
    dists = torch.cdist(traj_flat, traj_flat, p=2)  # (B, K, K)

    # Average upper triangle (exclude diagonal)  # [SELF-IMPLEMENTED]
    mask = torch.triu(torch.ones(K, K, device=trajectories.device), diagonal=1).bool()
    diversity = dists[:, mask].mean(dim=-1)  # (B,)

    return diversity


@torch.no_grad()
def compute_best_of_k_fde(
    trajectories: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:  # [SELF-IMPLEMENTED]
    """
    Best-of-K FDE: minimum FDE among K generated samples.

    minFDE_K = min_{k=1..K} || traj_k[-1] - target[-1] ||_2

    This is the oracle metric -- the best possible result if we had a perfect
    scorer. It measures the coverage of the trajectory distribution:
    does at least one sample land close to the ground truth?
    """
    # trajectories: (B, K, T, 2), target: (B, T, 2)
    fde_all = compute_fde(trajectories, target)  # (B, K)
    best_fde = fde_all.min(dim=-1).values  # (B,)
    return best_fde


@torch.no_grad()
def validate(
    model: GenAD,
    val_loader: DataLoader,
    device: torch.device,
    num_samples: int = 8,
    guidance_scale: float = 2.0,
    max_batches: int = 50,
) -> Dict[str, float]:  # [SELF-IMPLEMENTED]
    """
    Validate the model by generating trajectories and computing metrics.

    Metrics computed:
    - Diffusion loss (noise prediction MSE on validation set)
    - FDE: Final Displacement Error of best-scored trajectory
    - ADE: Average Displacement Error of best-scored trajectory
    - Diversity: average pairwise distance between K generated samples
    - Best-of-K FDE: oracle minimum FDE (measures distribution coverage)
    """
    model.eval()

    total_diff_loss = 0.0
    total_fde = 0.0
    total_ade = 0.0
    total_diversity = 0.0
    total_best_k_fde = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(
        tqdm(val_loader, desc="[Validate]", total=min(max_batches, len(val_loader)))
    ):
        if batch_idx >= max_batches:
            break

        scene_features = batch["scene_features"].to(device)  # (B, scene_dim)
        expert_traj = batch["expert_trajectory"].to(device)  # (B, T, 2)
        B = expert_traj.shape[0]

        # Compute validation diffusion loss  # [SELF-IMPLEMENTED]
        scene_context = scene_features.unsqueeze(1).expand(-1, 4, -1)
        t = torch.randint(0, model.num_steps, (B,), device=device)
        noisy_traj, noise = model.add_noise(expert_traj, t)
        predicted_noise = model.diffusion(noisy_traj, t, scene_context)
        diff_loss = F.mse_loss(predicted_noise, noise)
        total_diff_loss += diff_loss.item()

        # Generate trajectories via reverse diffusion  # [FROM PAPER]
        trajectories = sample_trajectories(
            model=model,
            scene_context=scene_context,
            num_samples=num_samples,
            guidance_scale=guidance_scale,
            use_guidance=True,
        )  # (B, K, T, 2)

        # Score and select best trajectory  # [SELF-IMPLEMENTED]
        best_traj, scores, best_idx = score_and_select(
            model, trajectories, scene_features
        )

        # Compute metrics  # [SELF-IMPLEMENTED]
        fde = compute_fde(best_traj, expert_traj).mean().item()
        ade = compute_ade(best_traj, expert_traj).mean().item()
        diversity = compute_diversity(trajectories).mean().item()
        best_k_fde = compute_best_of_k_fde(trajectories, expert_traj).mean().item()

        total_fde += fde
        total_ade += ade
        total_diversity += diversity
        total_best_k_fde += best_k_fde
        num_batches += 1

    n = max(num_batches, 1)
    return {
        "val_diffusion_loss": total_diff_loss / n,
        "val_fde": total_fde / n,
        "val_ade": total_ade / n,
        "val_diversity": total_diversity / n,
        "val_best_of_k_fde": total_best_k_fde / n,
    }


# =============================================================================
# Checkpoint Management
# =============================================================================


def save_checkpoint(
    model: GenAD,
    diffusion_optimizer: torch.optim.Optimizer,
    scorer_optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    epoch: int,
    metrics: Dict[str, float],
    save_dir: str,
    is_best: bool = False,
) -> str:  # [SELF-IMPLEMENTED]
    """
    Save training checkpoint with model weights, optimizer state, and metrics.

    Saves:
    - checkpoint_epoch_{N}.pt: periodic checkpoint
    - best_model.pt: best model by validation FDE (if is_best)
    """
    os.makedirs(save_dir, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "diffusion_optimizer_state_dict": diffusion_optimizer.state_dict(),
        "scorer_optimizer_state_dict": scorer_optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "metrics": metrics,
    }

    # Save periodic checkpoint  # [SELF-IMPLEMENTED]
    path = os.path.join(save_dir, f"checkpoint_epoch_{epoch:03d}.pt")
    torch.save(checkpoint, path)

    # Save best model  # [SELF-IMPLEMENTED]
    if is_best:
        best_path = os.path.join(save_dir, "best_model.pt")
        torch.save(checkpoint, best_path)
        print(f"  -> Saved best model (val_fde: {metrics.get('val_fde', 'N/A'):.4f})")

    return path


def load_checkpoint(
    path: str,
    model: GenAD,
    diffusion_optimizer: Optional[torch.optim.Optimizer] = None,
    scorer_optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[LambdaLR] = None,
) -> Dict:  # [SELF-IMPLEMENTED]
    """Load checkpoint and restore training state."""
    checkpoint = torch.load(path, map_location="cpu")

    model.load_state_dict(checkpoint["model_state_dict"])
    if diffusion_optimizer and "diffusion_optimizer_state_dict" in checkpoint:
        diffusion_optimizer.load_state_dict(
            checkpoint["diffusion_optimizer_state_dict"]
        )
    if scorer_optimizer and "scorer_optimizer_state_dict" in checkpoint:
        scorer_optimizer.load_state_dict(
            checkpoint["scorer_optimizer_state_dict"]
        )
    if scheduler and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    return checkpoint


# =============================================================================
# Main Training Entry Point
# =============================================================================


def parse_args() -> argparse.Namespace:  # [SELF-IMPLEMENTED]
    """Parse command-line arguments for GenAD training."""
    parser = argparse.ArgumentParser(
        description="GenAD: Generative End-to-End Autonomous Driving - Training Script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Training hyperparameters
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Training batch size"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Peak learning rate"
    )
    parser.add_argument(
        "--scorer_lr",
        type=float,
        default=3e-4,
        help="Learning rate for scorer network",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-2, help="AdamW weight decay"
    )
    parser.add_argument(
        "--grad_clip", type=float, default=1.0, help="Gradient clipping max norm"
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,
        help="Fraction of steps for LR warmup",
    )

    # Diffusion settings
    parser.add_argument(
        "--diffusion_steps",
        type=int,
        default=100,
        help="Number of diffusion timesteps T",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=8,
        help="Number of trajectory samples K to generate",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=2.0,
        help="Classifier-free guidance scale w (1.0 = no guidance)",
    )
    parser.add_argument(
        "--p_uncond",
        type=float,
        default=0.1,
        help="Probability of dropping conditioning during training",
    )

    # Model architecture
    parser.add_argument(
        "--scene_dim", type=int, default=256, help="Scene feature dimension"
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=512, help="Hidden dimension"
    )
    parser.add_argument(
        "--num_waypoints", type=int, default=12, help="Number of trajectory waypoints"
    )

    # Dataset
    parser.add_argument(
        "--num_train_samples",
        type=int,
        default=5000,
        help="Number of synthetic training samples",
    )
    parser.add_argument(
        "--num_val_samples",
        type=int,
        default=1000,
        help="Number of synthetic validation samples",
    )

    # Training settings
    parser.add_argument(
        "--use_amp",
        action="store_true",
        default=True,
        help="Use automatic mixed precision",
    )
    parser.add_argument(
        "--no_amp", action="store_true", help="Disable AMP (use FP32)"
    )
    parser.add_argument(
        "--num_workers", type=int, default=0, help="DataLoader workers"
    )
    parser.add_argument(
        "--train_scorer",
        action="store_true",
        default=True,
        help="Train scorer alongside diffusion",
    )
    parser.add_argument(
        "--no_scorer",
        action="store_true",
        help="Disable scorer training",
    )

    # Checkpointing
    parser.add_argument(
        "--save_dir",
        type=str,
        default="checkpoints/genad",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--save_every", type=int, default=10, help="Save checkpoint every N epochs"
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )

    # Validation
    parser.add_argument(
        "--val_every", type=int, default=5, help="Validate every N epochs"
    )
    parser.add_argument(
        "--val_max_batches",
        type=int,
        default=20,
        help="Max batches for validation (for speed)",
    )

    # Reproducibility
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Handle flag conflicts  # [SELF-IMPLEMENTED]
    if args.no_amp:
        args.use_amp = False
    if args.no_scorer:
        args.train_scorer = False

    return args


def main():  # [SELF-IMPLEMENTED]
    """
    Main training entry point for GenAD.

    Training overview:
    ------------------
    GenAD trains a conditional diffusion model to generate diverse trajectory
    proposals for autonomous driving. The key insight is that driving is
    inherently multi-modal: at any moment, multiple valid actions exist
    (lane change, slow down, speed up, etc.).

    Instead of regressing a single "average" trajectory, GenAD:
    1. Trains a denoising model to reverse a noise process on expert trajectories
    2. At inference, generates K diverse samples from learned distribution
    3. Uses a scorer to select the best trajectory for the current scene

    The diffusion objective is simple but powerful:
        L = E[|| noise - model(noisy_trajectory, t, scene) ||^2]

    This trains the model to predict what noise was added, enabling iterative
    denoising from pure Gaussian noise to valid trajectories.
    """
    args = parse_args()

    # Set random seeds for reproducibility  # [SELF-IMPLEMENTED]
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Device setup  # [SELF-IMPLEMENTED]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("GenAD: Generative End-to-End Autonomous Driving - Training")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Diffusion steps: {args.diffusion_steps}")
    print(f"Num trajectory samples (K): {args.num_samples}")
    print(f"Guidance scale: {args.guidance_scale}")
    print(f"CFG conditioning dropout: {args.p_uncond}")
    print(f"AMP: {args.use_amp}")
    print(f"Train scorer: {args.train_scorer}")
    print("=" * 70)

    # Create model  # [SELF-IMPLEMENTED]
    model = GenAD(
        scene_dim=args.scene_dim,
        hidden_dim=args.hidden_dim,
        num_waypoints=args.num_waypoints,
        num_diffusion_steps=args.diffusion_steps,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    num_diffusion_params = sum(p.numel() for p in model.diffusion.parameters())
    num_scorer_params = sum(p.numel() for p in model.scorer.parameters())
    print(f"\nModel parameters:")
    print(f"  Total:     {num_params:,}")
    print(f"  Diffusion: {num_diffusion_params:,}")
    print(f"  Scorer:    {num_scorer_params:,}")

    # Create datasets  # [SELF-IMPLEMENTED]
    print(f"\nCreating synthetic datasets...")
    train_dataset = GenADDataset(
        num_samples=args.num_train_samples,
        num_waypoints=args.num_waypoints,
        scene_dim=args.scene_dim,
        split="train",
        seed=args.seed,
    )
    val_dataset = GenADDataset(
        num_samples=args.num_val_samples,
        num_waypoints=args.num_waypoints,
        scene_dim=args.scene_dim,
        split="val",
        seed=args.seed,
    )
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val:   {len(val_dataset)} samples")

    # Create dataloaders  # [SELF-IMPLEMENTED]
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Optimizers  # [SELF-IMPLEMENTED]
    # Separate optimizer for diffusion model (main) and scorer (auxiliary)
    diffusion_params = list(model.diffusion.parameters()) + list(
        model.scene_encoder.parameters()
    )
    diffusion_optimizer = torch.optim.AdamW(
        diffusion_params, lr=args.lr, weight_decay=args.weight_decay
    )
    scorer_optimizer = torch.optim.AdamW(
        model.scorer.parameters(), lr=args.scorer_lr, weight_decay=args.weight_decay
    )

    # LR scheduler with warmup + cosine decay  # [SELF-IMPLEMENTED]
    num_training_steps = args.epochs * len(train_loader)
    num_warmup_steps = int(args.warmup_ratio * num_training_steps)
    diffusion_scheduler = get_cosine_schedule_with_warmup(
        diffusion_optimizer, num_warmup_steps, num_training_steps
    )

    # Loss functions  # [FROM PAPER] / [SELF-IMPLEMENTED]
    diffusion_loss_fn = DiffusionLoss(loss_type="mse")  # [FROM PAPER] standard DDPM
    scorer_loss_fn = ScorerLoss(ranking_weight=0.5, margin=1.0)  # [SELF-IMPLEMENTED]
    guidance_loss_fn = GuidanceLoss(p_uncond=args.p_uncond)  # [FROM PAPER]

    # Resume from checkpoint if specified  # [SELF-IMPLEMENTED]
    start_epoch = 0
    best_val_fde = float("inf")

    if args.resume:
        checkpoint = load_checkpoint(
            args.resume, model, diffusion_optimizer, scorer_optimizer, diffusion_scheduler
        )
        start_epoch = checkpoint["epoch"] + 1
        if "metrics" in checkpoint and "val_fde" in checkpoint["metrics"]:
            best_val_fde = checkpoint["metrics"]["val_fde"]
        print(f"Resuming from epoch {start_epoch}, best_val_fde={best_val_fde:.4f}")

    # Training loop  # [SELF-IMPLEMENTED]
    print(f"\n{'='*70}")
    print("Starting training...")
    print(f"{'='*70}\n")

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()

        # Train one epoch  # [SELF-IMPLEMENTED]
        train_metrics = train_one_epoch(
            model=model,
            train_loader=train_loader,
            diffusion_optimizer=diffusion_optimizer,
            scorer_optimizer=scorer_optimizer,
            diffusion_scheduler=diffusion_scheduler,
            diffusion_loss_fn=diffusion_loss_fn,
            guidance_loss_fn=guidance_loss_fn,
            scorer_loss_fn=scorer_loss_fn,
            device=device,
            epoch=epoch,
            use_amp=args.use_amp,
            grad_clip=args.grad_clip,
            train_scorer_flag=args.train_scorer,
        )

        epoch_time = time.time() - epoch_start

        # Print training metrics  # [SELF-IMPLEMENTED]
        print(f"\nEpoch {epoch}/{args.epochs-1} ({epoch_time:.1f}s):")
        print(f"  Diffusion loss: {train_metrics['diffusion_loss']:.4f}")
        if args.train_scorer and "scorer_loss" in train_metrics:
            print(
                f"  Scorer loss: {train_metrics['scorer_loss']:.4f} "
                f"(BCE: {train_metrics['scorer_bce']:.4f}, "
                f"Rank: {train_metrics['scorer_ranking']:.4f})"
            )

        # Validation  # [SELF-IMPLEMENTED]
        val_metrics = {}
        if (epoch + 1) % args.val_every == 0 or epoch == args.epochs - 1:
            print(f"\n  Running validation (K={args.num_samples}, guidance={args.guidance_scale})...")
            val_metrics = validate(
                model=model,
                val_loader=val_loader,
                device=device,
                num_samples=args.num_samples,
                guidance_scale=args.guidance_scale,
                max_batches=args.val_max_batches,
            )
            print(f"  Val diffusion loss: {val_metrics['val_diffusion_loss']:.4f}")
            print(f"  Val FDE (scored):   {val_metrics['val_fde']:.4f}")
            print(f"  Val ADE (scored):   {val_metrics['val_ade']:.4f}")
            print(f"  Val Diversity:      {val_metrics['val_diversity']:.4f}")
            print(f"  Val Best-of-{args.num_samples} FDE: {val_metrics['val_best_of_k_fde']:.4f}")

            # Track best model  # [SELF-IMPLEMENTED]
            is_best = val_metrics["val_fde"] < best_val_fde
            if is_best:
                best_val_fde = val_metrics["val_fde"]

        # Save checkpoint  # [SELF-IMPLEMENTED]
        all_metrics = {**train_metrics, **val_metrics}
        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            save_checkpoint(
                model=model,
                diffusion_optimizer=diffusion_optimizer,
                scorer_optimizer=scorer_optimizer,
                scheduler=diffusion_scheduler,
                epoch=epoch,
                metrics=all_metrics,
                save_dir=args.save_dir,
                is_best=val_metrics.get("val_fde", float("inf")) <= best_val_fde
                if val_metrics
                else False,
            )
            print(f"  Checkpoint saved to {args.save_dir}/")

        print()

    # Final summary  # [SELF-IMPLEMENTED]
    print("=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"  Best validation FDE: {best_val_fde:.4f}")
    print(f"  Checkpoints saved to: {args.save_dir}/")
    print(f"\nTo generate trajectories with the trained model:")
    print(f"  Load best_model.pt, call sample_trajectories() then score_and_select()")
    print("=" * 70)


if __name__ == "__main__":
    main()
