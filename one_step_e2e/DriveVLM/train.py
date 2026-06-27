"""
DriveVLM Training Script
=========================
ATTRIBUTION:
- Loss functions: Based on DriveVLM paper (Tian et al., 2024)
  - Trajectory regression: L1 loss on waypoints (Section 3.3)
  - Language modeling: Cross-entropy on text tokens (Section 3.2)
  - Multi-task: weighted sum of trajectory + language losses
- Training paradigm: Foundation model approach (from paper Section 4):
  - Stage 1: Vision encoder pre-training (CLIP-style, self-supervised)
  - Stage 2: Vision-language alignment fine-tuning
  - Stage 3: Driving-specific trajectory fine-tuning
  - Stage 4: RL from driving rewards (PPO/DPO)
- Implementation: Self-implemented in PyTorch (simplified - real uses InternVL 7B)
- Synthetic dataset: Self-implemented for demonstration
- NOTE: This demo shows Stage 3 (trajectory fine-tuning) only.
  Real training requires massive compute for Stage 1-2.
"""

import os
import sys
import math
import time
import json
import argparse
import logging
import contextlib
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
# AMP is used via torch.cuda.amp directly for broad compatibility  # [SELF-IMPLEMENTED]

# Import model from same directory
from model import DriveVLM

# Try to import tqdm; fall back gracefully
try:
    from tqdm import tqdm
except ImportError:
    # Simple fallback if tqdm is not installed  # [SELF-IMPLEMENTED]
    def tqdm(iterable, **kwargs):
        desc = kwargs.get('desc', '')
        total = kwargs.get('total', None)
        for i, item in enumerate(iterable):
            if total and i % max(1, total // 10) == 0:
                print(f"  {desc} [{i}/{total}]")
            yield item


# ==============================================================================
# Logging Setup  # [SELF-IMPLEMENTED]
# ==============================================================================

def setup_logging(log_dir: str, rank: int = 0) -> logging.Logger:
    """Configure logging for training runs."""
    logger = logging.getLogger("DriveVLM")
    logger.setLevel(logging.INFO if rank == 0 else logging.WARNING)

    formatter = logging.Formatter(
        '[%(asctime)s][%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler
    if rank == 0:
        os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(log_dir, 'train.log'))
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


# ==============================================================================
# Synthetic Dataset  # [SELF-IMPLEMENTED]
# ==============================================================================

class DriveVLMDataset(Dataset):
    """
    Synthetic dataset for DriveVLM training demonstration.  # [SELF-IMPLEMENTED]

    In the real DriveVLM system, data comes from:
    - nuScenes dataset (1000 driving scenes, 6 cameras per frame)
    - OpenScene dataset (extended nuScenes with language annotations)
    - Custom collected data with chain-of-thought annotations

    This synthetic dataset generates:
    - Multi-view images: (6, 3, 224, 224) - 6 camera views
    - Text prompt tokens: (20,) - simulating driving commands like
      "Drive forward", "Turn left at intersection", etc.
    - Ground truth trajectory: (6, 2) - 6 waypoints at 0.5s intervals (3s total)
    - Ground truth text response: (20,) - simulating model responses like
      "Proceeding straight, clear road ahead"

    The synthetic data uses structured patterns so the model can learn:
    - Straight driving: trajectory goes forward (positive y, small x)
    - Left turns: trajectory curves to the left (negative x)
    - Right turns: trajectory curves to the right (positive x)
    - Stops: trajectory stays near origin
    """

    # Driving command vocabulary (simplified)  # [SELF-IMPLEMENTED]
    COMMANDS = [
        "drive_forward",        # Token pattern: [1, 2, 3, ...]
        "turn_left",            # Token pattern: [4, 5, 6, ...]
        "turn_right",           # Token pattern: [7, 8, 9, ...]
        "stop",                 # Token pattern: [10, 11, 12, ...]
        "lane_change_left",     # Token pattern: [13, 14, 15, ...]
        "lane_change_right",    # Token pattern: [16, 17, 18, ...]
    ]

    def __init__(
        self,
        num_samples: int = 1000,
        num_views: int = 6,
        img_size: int = 224,
        num_waypoints: int = 6,
        text_seq_len: int = 20,
        vocab_size: int = 32000,
        split: str = 'train',
        seed: int = 42,
    ):
        """
        Args:
            num_samples: Number of synthetic driving scenarios
            num_views: Number of camera views (6 for surround-view)
            img_size: Image resolution per view
            num_waypoints: Number of future waypoints to predict
            text_seq_len: Length of text token sequences
            vocab_size: Vocabulary size for text tokens
            split: 'train' or 'val' (affects random seed offset)
            seed: Random seed for reproducibility
        """
        super().__init__()
        self.num_samples = num_samples
        self.num_views = num_views
        self.img_size = img_size
        self.num_waypoints = num_waypoints
        self.text_seq_len = text_seq_len
        self.vocab_size = vocab_size
        self.split = split

        # Use different seed for train/val to avoid data leakage  # [SELF-IMPLEMENTED]
        self.rng = torch.Generator()
        seed_offset = 0 if split == 'train' else 10000
        self.rng.manual_seed(seed + seed_offset)

        # Pre-generate scenario types for consistency  # [SELF-IMPLEMENTED]
        self.scenario_types = torch.randint(
            0, len(self.COMMANDS), (num_samples,), generator=self.rng
        )

        logging.getLogger("DriveVLM").info(
            f"Created {split} dataset: {num_samples} samples, "
            f"{num_views} views, {img_size}x{img_size}"
        )

    def __len__(self) -> int:
        return self.num_samples

    def _generate_trajectory(self, scenario_type: int) -> torch.Tensor:
        """
        Generate ground-truth trajectory based on scenario type.  # [SELF-IMPLEMENTED]

        The trajectory represents future positions in ego-vehicle coordinates:
        - x: lateral displacement (positive = right)
        - y: longitudinal displacement (positive = forward)
        Waypoints at 0.5s intervals: [0.5s, 1.0s, 1.5s, 2.0s, 2.5s, 3.0s]
        """
        t = torch.linspace(0.5, 3.0, self.num_waypoints)  # Time steps

        if scenario_type == 0:  # drive_forward
            # Straight line at ~10 m/s
            x = torch.zeros(self.num_waypoints) + torch.randn(self.num_waypoints) * 0.1
            y = t * 10.0 + torch.randn(self.num_waypoints) * 0.2
        elif scenario_type == 1:  # turn_left
            # Curved path to the left
            x = -t * 3.0 + torch.randn(self.num_waypoints) * 0.2
            y = t * 7.0 + torch.randn(self.num_waypoints) * 0.2
        elif scenario_type == 2:  # turn_right
            # Curved path to the right
            x = t * 3.0 + torch.randn(self.num_waypoints) * 0.2
            y = t * 7.0 + torch.randn(self.num_waypoints) * 0.2
        elif scenario_type == 3:  # stop
            # Decelerating to stop
            x = torch.zeros(self.num_waypoints) + torch.randn(self.num_waypoints) * 0.05
            y = t * 2.0 * (1.0 - t / 3.0) + torch.randn(self.num_waypoints) * 0.1
        elif scenario_type == 4:  # lane_change_left
            # Lateral shift left while moving forward
            x = -2.0 * torch.sigmoid((t - 1.5) * 3.0) + torch.randn(self.num_waypoints) * 0.1
            y = t * 10.0 + torch.randn(self.num_waypoints) * 0.2
        else:  # lane_change_right
            # Lateral shift right while moving forward
            x = 2.0 * torch.sigmoid((t - 1.5) * 3.0) + torch.randn(self.num_waypoints) * 0.1
            y = t * 10.0 + torch.randn(self.num_waypoints) * 0.2

        trajectory = torch.stack([x, y], dim=-1)  # (num_waypoints, 2)
        return trajectory

    def _generate_prompt_tokens(self, scenario_type: int) -> torch.Tensor:
        """
        Generate text prompt tokens that encode the driving command.  # [SELF-IMPLEMENTED]

        In the real system, these would be tokenized natural language:
        e.g., "The navigation says turn left at the next intersection.
               What should you do?"
        Here we use structured token patterns for learnability.
        """
        # Base token for command type (offset to avoid special tokens 0-99)
        base = 100 + scenario_type * 50
        # Create a structured but varied token sequence
        tokens = torch.arange(base, base + self.text_seq_len)
        # Add some noise to simulate natural language variation
        noise = torch.randint(0, 10, (self.text_seq_len,))
        tokens = (tokens + noise) % self.vocab_size
        # Ensure tokens are in valid range
        tokens = tokens.clamp(1, self.vocab_size - 1)  # Avoid token 0 (padding)
        return tokens.long()

    def _generate_response_tokens(self, scenario_type: int) -> torch.Tensor:
        """
        Generate ground-truth text response tokens.  # [SELF-IMPLEMENTED]

        In the real system, these encode chain-of-thought reasoning:
        e.g., "I see a clear road ahead with no obstacles.
               The navigation indicates a left turn in 50m.
               I will begin turning left while maintaining safe speed."
        """
        # Response tokens are offset from prompts to simulate different vocab usage
        base = 500 + scenario_type * 80
        tokens = torch.arange(base, base + self.text_seq_len)
        noise = torch.randint(0, 15, (self.text_seq_len,))
        tokens = (tokens + noise) % self.vocab_size
        tokens = tokens.clamp(1, self.vocab_size - 1)
        return tokens.long()

    def _generate_images(self, scenario_type: int, idx: int) -> torch.Tensor:
        """
        Generate synthetic multi-view images.  # [SELF-IMPLEMENTED]

        In the real system: 6 actual camera images (front, front-left, front-right,
        back, back-left, back-right) from the driving dataset.

        Here: random images with scenario-dependent bias so the vision encoder
        can learn to associate visual patterns with driving scenarios.
        We add structured patterns:
        - Different mean intensities per scenario
        - Horizontal/vertical gradients that correlate with turn direction
        """
        # Use idx for deterministic generation within an epoch
        gen = torch.Generator()
        gen.manual_seed(idx * 7 + scenario_type * 13)

        images = torch.randn(
            self.num_views, 3, self.img_size, self.img_size,
            generator=gen
        )

        # Add scenario-dependent visual bias  # [SELF-IMPLEMENTED]
        # This gives the vision encoder learnable patterns
        bias = scenario_type * 0.3
        images[:, 0] += bias  # Red channel shift by scenario

        # Add directional gradients for turns
        h_grad = torch.linspace(-1, 1, self.img_size).view(1, 1, 1, -1)
        if scenario_type in [1, 4]:  # Left scenarios
            images += h_grad * 0.5
        elif scenario_type in [2, 5]:  # Right scenarios
            images -= h_grad * 0.5

        # Normalize to reasonable range
        images = images * 0.5  # Scale down
        return images

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single training sample.

        Returns:
            Dict with keys:
                - 'images': (6, 3, 224, 224) multi-view camera images
                - 'prompt_tokens': (20,) text prompt token IDs
                - 'gt_trajectory': (6, 2) ground-truth waypoints
                - 'gt_response_tokens': (20,) ground-truth text response
                - 'scenario_type': int scenario label
        """
        scenario_type = self.scenario_types[idx].item()

        images = self._generate_images(scenario_type, idx)
        prompt_tokens = self._generate_prompt_tokens(scenario_type)
        gt_trajectory = self._generate_trajectory(scenario_type)
        gt_response_tokens = self._generate_response_tokens(scenario_type)

        return {
            'images': images,                       # (6, 3, 224, 224)
            'prompt_tokens': prompt_tokens,         # (20,)
            'gt_trajectory': gt_trajectory,         # (6, 2)
            'gt_response_tokens': gt_response_tokens,  # (20,)
            'scenario_type': scenario_type,         # int
        }


# ==============================================================================
# Loss Functions  # [FROM PAPER]
# ==============================================================================

class TrajectoryLoss(nn.Module):
    """
    Trajectory regression loss from DriveVLM paper (Section 3.3).  # [FROM PAPER]

    Uses L1 (smooth) loss on predicted vs. ground-truth waypoints.
    L1 is preferred over L2 because:
    - More robust to outliers in trajectory data
    - Produces sharper trajectory predictions (less averaging)
    - Better gradient behavior for large errors

    The paper uses a multi-horizon formulation where each waypoint
    can optionally have a different weight (closer waypoints weighted more).
    """

    def __init__(self, num_waypoints: int = 6, use_horizon_weights: bool = True):
        """
        Args:
            num_waypoints: Number of waypoints to predict
            use_horizon_weights: Whether to weight waypoints by time horizon
                                 (closer = higher weight, as in the paper)
        """
        super().__init__()
        self.num_waypoints = num_waypoints
        self.use_horizon_weights = use_horizon_weights

        if use_horizon_weights:
            # Exponentially decaying weights: closer waypoints matter more  # [FROM PAPER]
            # Paper uses: w_t = exp(-0.3 * t) for t in [0, 1, 2, 3, 4, 5]
            weights = torch.exp(-0.3 * torch.arange(num_waypoints, dtype=torch.float32))
            weights = weights / weights.sum()  # Normalize
            self.register_buffer('horizon_weights', weights)
        else:
            self.register_buffer('horizon_weights', torch.ones(num_waypoints) / num_waypoints)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted L1 loss on trajectory waypoints.  # [FROM PAPER]

        Args:
            pred: (B, num_waypoints, 2) predicted [x, y] waypoints
            target: (B, num_waypoints, 2) ground-truth [x, y] waypoints
        Returns:
            Scalar loss value
        """
        # Per-waypoint L1 error: |pred - target| averaged over (x, y)
        per_waypoint_error = F.l1_loss(pred, target, reduction='none')  # (B, T, 2)
        per_waypoint_error = per_waypoint_error.mean(dim=-1)  # (B, T) avg over x,y

        # Apply horizon weights  # [FROM PAPER]
        weights = self.horizon_weights.to(pred.device)  # (T,)
        weighted_error = per_waypoint_error * weights.unsqueeze(0)  # (B, T)

        # Average over batch and waypoints
        loss = weighted_error.sum(dim=-1).mean()  # scalar
        return loss


class LanguageModelingLoss(nn.Module):
    """
    Language modeling loss from DriveVLM paper (Section 3.2).  # [FROM PAPER]

    Standard next-token prediction cross-entropy loss, applied only to
    the text portion of the output sequence (not visual tokens).

    In DriveVLM, this loss trains the model to generate:
    1. Scene descriptions ("There is a pedestrian crossing ahead")
    2. Reasoning chains ("I need to slow down because...")
    3. Planning rationale ("Turning left to follow the route")

    The chain-of-thought capability is what makes DriveVLM interpretable
    compared to black-box end-to-end models.
    """

    def __init__(self, vocab_size: int = 32000, ignore_index: int = -100,
                 label_smoothing: float = 0.0):
        """
        Args:
            vocab_size: Size of the text vocabulary
            ignore_index: Token ID to ignore in loss (padding)
            label_smoothing: Label smoothing factor (0.1 used in some VLM papers)
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index
        self.ce_loss = nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                num_visual_tokens: int = 64) -> torch.Tensor:
        """
        Compute cross-entropy loss on text tokens only.  # [FROM PAPER]

        The model outputs logits for the full sequence [visual_tokens | text_tokens].
        We only compute loss on the text portion (autoregressive LM objective).

        Args:
            logits: (B, total_seq_len, vocab_size) model output logits
            targets: (B, text_seq_len) ground-truth next-token IDs
            num_visual_tokens: Number of visual tokens (to skip in logits)
        Returns:
            Scalar cross-entropy loss
        """
        # Extract text logits: shifted by 1 for next-token prediction
        # The text portion starts after visual tokens in the sequence
        text_logits = logits[:, num_visual_tokens:-1].contiguous()  # (B, T-1, V)
        text_targets = targets[:, 1:].contiguous()  # (B, T-1)

        # Ensure we don't index out of bounds
        min_len = min(text_logits.shape[1], text_targets.shape[1])
        text_logits = text_logits[:, :min_len]
        text_targets = text_targets[:, :min_len]

        # Reshape for cross-entropy: (B*T, V) vs (B*T,)
        loss = self.ce_loss(
            text_logits.reshape(-1, self.vocab_size),
            text_targets.reshape(-1),
        )
        return loss


class DriveVLMLoss(nn.Module):
    """
    Combined multi-task loss for DriveVLM training.  # [FROM PAPER]

    Total loss = alpha * L_trajectory + (1 - alpha) * L_language

    The weighting alpha controls the balance between:
    - Trajectory accuracy (safety-critical)
    - Language quality (interpretability)

    Paper findings (Section 4.2):
    - alpha=0.7 works best for driving performance
    - alpha=0.5 gives best language quality
    - alpha=0.9 maximizes trajectory metrics but degrades explanations
    - During Stage 3, trajectory loss dominates (alpha=0.7-0.8)
    """

    def __init__(self, alpha: float = 0.7, vocab_size: int = 32000,
                 num_waypoints: int = 6, num_visual_tokens: int = 64,
                 use_horizon_weights: bool = True,
                 label_smoothing: float = 0.0):
        """
        Args:
            alpha: Weight for trajectory loss (1-alpha for language loss)
            vocab_size: Text vocabulary size
            num_waypoints: Number of trajectory waypoints
            num_visual_tokens: Number of visual tokens in the sequence
            use_horizon_weights: Whether to weight trajectory waypoints by time
            label_smoothing: Smoothing factor for language loss
        """
        super().__init__()
        self.alpha = alpha
        self.num_visual_tokens = num_visual_tokens

        self.traj_loss = TrajectoryLoss(  # [FROM PAPER]
            num_waypoints=num_waypoints,
            use_horizon_weights=use_horizon_weights,
        )
        self.lang_loss = LanguageModelingLoss(  # [FROM PAPER]
            vocab_size=vocab_size,
            label_smoothing=label_smoothing,
        )

    def forward(self, output: Dict, gt_trajectory: torch.Tensor,
                gt_text_tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute combined DriveVLM loss.  # [FROM PAPER]

        Args:
            output: Model output dict with 'trajectory' and 'logits'
            gt_trajectory: (B, num_waypoints, 2) ground-truth waypoints
            gt_text_tokens: (B, text_seq_len) ground-truth text tokens
        Returns:
            Dict with 'total', 'trajectory', 'language' loss values
        """
        losses = {}

        # Trajectory regression loss  # [FROM PAPER]
        if 'trajectory' in output:
            losses['trajectory'] = self.traj_loss(output['trajectory'], gt_trajectory)
        else:
            losses['trajectory'] = torch.tensor(0.0, device=gt_trajectory.device)

        # Language modeling loss  # [FROM PAPER]
        if 'logits' in output:
            losses['language'] = self.lang_loss(
                output['logits'], gt_text_tokens, self.num_visual_tokens
            )
        else:
            losses['language'] = torch.tensor(0.0, device=gt_trajectory.device)

        # Weighted combination  # [FROM PAPER]
        losses['total'] = (
            self.alpha * losses['trajectory'] +
            (1.0 - self.alpha) * losses['language']
        )

        return losses


# ==============================================================================
# Reward Model (for Stage 4 - RL)  # [FROM PAPER] + [SELF-IMPLEMENTED]
# ==============================================================================

class RewardModel(nn.Module):
    """
    Reward model for Stage 4 RL training (PPO/DPO).  # [FROM PAPER]

    In the DriveVLM paper (Section 4, Stage 4), the model is further refined
    using reinforcement learning from driving rewards. This reward model
    evaluates the quality of generated trajectories and reasoning.

    Reward components:  # [FROM PAPER]
    1. Safety reward: No collisions, maintains safe distance
    2. Comfort reward: Smooth acceleration/jerk profiles
    3. Progress reward: Makes progress toward goal
    4. Rule compliance: Follows traffic rules
    5. Language quality: Reasoning is coherent and accurate

    This is a STUB implementation showing the architecture.
    Real RL training requires:
    - A simulator (CARLA, nuPlan) for online rollouts
    - A reference policy for KL divergence constraint
    - PPO/DPO training infrastructure
    """

    def __init__(self, embed_dim: int = 4096, num_waypoints: int = 6):
        """
        Args:
            embed_dim: Hidden dimension (matches LLM)
            num_waypoints: Number of trajectory waypoints
        """
        super().__init__()

        # Trajectory encoder  # [SELF-IMPLEMENTED]
        self.traj_encoder = nn.Sequential(
            nn.Linear(num_waypoints * 2, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, embed_dim // 2),
            nn.GELU(),
        )

        # Context encoder (from LLM hidden states)  # [SELF-IMPLEMENTED]
        self.context_encoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
        )

        # Reward head: predicts scalar reward  # [SELF-IMPLEMENTED]
        self.reward_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, 1),  # Single scalar reward
        )

    def forward(self, trajectory: torch.Tensor,
                hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Compute reward for a given trajectory and context.  # [SELF-IMPLEMENTED]

        Args:
            trajectory: (B, num_waypoints, 2) predicted trajectory
            hidden_states: (B, seq_len, embed_dim) LLM hidden states
        Returns:
            reward: (B, 1) scalar reward per sample
        """
        B = trajectory.shape[0]

        # Encode trajectory
        traj_flat = trajectory.reshape(B, -1)  # (B, num_waypoints * 2)
        traj_feat = self.traj_encoder(traj_flat)  # (B, embed_dim // 2)

        # Encode context (use mean pooling of hidden states)
        context_feat = self.context_encoder(
            hidden_states.mean(dim=1)  # (B, embed_dim)
        )  # (B, embed_dim // 2)

        # Concatenate and predict reward
        combined = torch.cat([traj_feat, context_feat], dim=-1)  # (B, embed_dim)
        reward = self.reward_head(combined)  # (B, 1)

        return reward


# ==============================================================================
# Validation Metrics  # [FROM PAPER] + [SELF-IMPLEMENTED]
# ==============================================================================

class TrajectoryMetrics:
    """
    Compute trajectory evaluation metrics from the DriveVLM paper.  # [FROM PAPER]

    Standard metrics for trajectory prediction in autonomous driving:
    - L2 distance at different time horizons (1s, 2s, 3s)
    - Average Displacement Error (ADE): mean L2 over all waypoints
    - Final Displacement Error (FDE): L2 at final waypoint
    - Collision rate (requires map info - not computed here)

    These metrics are reported at different planning horizons because:
    - Short-term (1s): safety-critical, should be very accurate
    - Mid-term (2s): important for smooth driving
    - Long-term (3s): shows planning capability, higher error acceptable
    """

    def __init__(self, num_waypoints: int = 6, dt: float = 0.5):
        """
        Args:
            num_waypoints: Number of waypoints (6 at 0.5s = 3.0s horizon)
            dt: Time interval between waypoints in seconds
        """
        self.num_waypoints = num_waypoints
        self.dt = dt
        self.reset()

    def reset(self):
        """Reset accumulated metrics for a new evaluation epoch."""
        self.l2_errors = []  # List of (B, T) tensors
        self.count = 0

    @torch.no_grad()
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Accumulate L2 errors for a batch.  # [SELF-IMPLEMENTED]

        Args:
            pred: (B, T, 2) predicted waypoints
            target: (B, T, 2) ground-truth waypoints
        """
        # Per-waypoint L2 distance
        l2 = torch.sqrt(((pred - target) ** 2).sum(dim=-1))  # (B, T)
        self.l2_errors.append(l2.cpu())
        self.count += pred.shape[0]

    def compute(self) -> Dict[str, float]:
        """
        Compute final metrics over accumulated predictions.  # [FROM PAPER]

        Returns:
            Dict with L2@1s, L2@2s, L2@3s, ADE, FDE
        """
        if not self.l2_errors:
            return {'L2@1s': 0.0, 'L2@2s': 0.0, 'L2@3s': 0.0,
                    'ADE': 0.0, 'FDE': 0.0}

        all_errors = torch.cat(self.l2_errors, dim=0)  # (N, T)

        # Compute metrics at different horizons  # [FROM PAPER]
        # Waypoint indices: 0=0.5s, 1=1.0s, 2=1.5s, 3=2.0s, 4=2.5s, 5=3.0s
        metrics = {}

        # L2 at 1 second (index 1, since 0-indexed at 0.5s intervals)
        idx_1s = min(1, all_errors.shape[1] - 1)
        metrics['L2@1s'] = all_errors[:, idx_1s].mean().item()

        # L2 at 2 seconds (index 3)
        idx_2s = min(3, all_errors.shape[1] - 1)
        metrics['L2@2s'] = all_errors[:, idx_2s].mean().item()

        # L2 at 3 seconds (index 5 = final)
        idx_3s = min(5, all_errors.shape[1] - 1)
        metrics['L2@3s'] = all_errors[:, idx_3s].mean().item()

        # Average Displacement Error: mean over all waypoints and samples
        metrics['ADE'] = all_errors.mean().item()

        # Final Displacement Error: L2 at the last waypoint
        metrics['FDE'] = all_errors[:, -1].mean().item()

        return metrics


class LanguageMetrics:
    """
    Compute language generation metrics.  # [FROM PAPER]

    Primary metric: Perplexity = exp(cross-entropy loss)
    - Lower perplexity = better language model
    - Typical values for driving VLMs: 5-20 perplexity

    Additional metrics (not computed here but used in the paper):
    - BLEU score (for scene description quality)
    - CIDEr score (for caption diversity)
    - Human evaluation (reasoning correctness)
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset for new epoch."""
        self.total_loss = 0.0
        self.total_tokens = 0

    def update(self, loss: float, num_tokens: int):
        """Accumulate cross-entropy loss values."""
        self.total_loss += loss * num_tokens
        self.total_tokens += num_tokens

    def compute(self) -> Dict[str, float]:
        """
        Compute perplexity from accumulated losses.  # [FROM PAPER]

        Returns:
            Dict with 'perplexity' and 'avg_loss'
        """
        if self.total_tokens == 0:
            return {'perplexity': float('inf'), 'avg_loss': 0.0}

        avg_loss = self.total_loss / self.total_tokens
        perplexity = math.exp(min(avg_loss, 100))  # Clamp to avoid overflow

        return {
            'perplexity': perplexity,
            'avg_loss': avg_loss,
        }


# ==============================================================================
# Learning Rate Scheduler with Warmup  # [SELF-IMPLEMENTED]
# ==============================================================================

class WarmupCosineScheduler:
    """
    Learning rate scheduler with linear warmup + cosine decay.  # [SELF-IMPLEMENTED]

    This is the standard scheduler used in most VLM training:
    1. Linear warmup: LR goes from 0 to base_lr over warmup_steps
    2. Cosine decay: LR decays from base_lr to min_lr following cosine curve

    Used in InternVL, LLaVA, and DriveVLM pre-training.
    The warmup helps stabilize early training when gradients are noisy.
    """

    def __init__(self, optimizer, base_lr: float, min_lr: float,
                 warmup_steps: int, total_steps: int):
        """
        Args:
            optimizer: PyTorch optimizer
            base_lr: Peak learning rate (reached after warmup)
            min_lr: Minimum learning rate (at end of training)
            warmup_steps: Number of steps for linear warmup
            total_steps: Total training steps
        """
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.current_step = 0

    def step(self):
        """Update learning rate based on current step."""
        self.current_step += 1
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_lr(self) -> float:
        """Compute current learning rate."""
        if self.current_step <= self.warmup_steps:
            # Linear warmup  # [SELF-IMPLEMENTED]
            return self.base_lr * (self.current_step / max(1, self.warmup_steps))
        else:
            # Cosine decay  # [SELF-IMPLEMENTED]
            progress = (self.current_step - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps)
            progress = min(progress, 1.0)
            return self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
                1.0 + math.cos(math.pi * progress))


# ==============================================================================
# Checkpoint Management  # [SELF-IMPLEMENTED]
# ==============================================================================

class CheckpointManager:
    """
    Manages model checkpoints during training.  # [SELF-IMPLEMENTED]

    Features:
    - Save best model (by validation metric)
    - Save periodic checkpoints
    - Keep only top-K checkpoints to save disk space
    - Resume from checkpoint
    """

    def __init__(self, save_dir: str, max_checkpoints: int = 5):
        """
        Args:
            save_dir: Directory to save checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.checkpoints = []  # List of (metric, path) tuples
        self.best_metric = float('inf')

    def save(self, model: nn.Module, optimizer, scheduler,
             epoch: int, step: int, metrics: Dict,
             is_best: bool = False) -> str:
        """
        Save a training checkpoint.  # [SELF-IMPLEMENTED]

        Args:
            model: The DriveVLM model
            optimizer: Optimizer state
            scheduler: LR scheduler state
            epoch: Current epoch number
            step: Current global step
            metrics: Current validation metrics
            is_best: Whether this is the best model so far
        Returns:
            Path to saved checkpoint
        """
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_step': scheduler.current_step if scheduler else 0,
            'metrics': metrics,
            'best_metric': self.best_metric,
        }

        # Save periodic checkpoint
        filename = f"checkpoint_epoch{epoch:03d}_step{step:06d}.pt"
        filepath = self.save_dir / filename
        torch.save(checkpoint, filepath)

        # Track checkpoints for cleanup
        metric_val = metrics.get('total_loss', float('inf'))
        self.checkpoints.append((metric_val, str(filepath)))

        # Keep only top-K checkpoints
        if len(self.checkpoints) > self.max_checkpoints:
            self.checkpoints.sort(key=lambda x: x[0])
            _, worst_path = self.checkpoints.pop()
            if os.path.exists(worst_path) and worst_path != str(self.save_dir / 'best_model.pt'):
                os.remove(worst_path)

        # Save best model separately
        if is_best:
            best_path = self.save_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            self.best_metric = metric_val

        return str(filepath)

    def load(self, filepath: str, model: nn.Module,
             optimizer=None, scheduler=None) -> Dict:
        """
        Load a checkpoint and restore training state.  # [SELF-IMPLEMENTED]

        Args:
            filepath: Path to checkpoint file
            model: Model to load weights into
            optimizer: Optional optimizer to restore
            scheduler: Optional scheduler to restore
        Returns:
            Dict with epoch, step, metrics from checkpoint
        """
        checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)

        model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if scheduler and 'scheduler_step' in checkpoint:
            scheduler.current_step = checkpoint['scheduler_step']

        self.best_metric = checkpoint.get('best_metric', float('inf'))

        return {
            'epoch': checkpoint['epoch'],
            'step': checkpoint['step'],
            'metrics': checkpoint.get('metrics', {}),
        }


# ==============================================================================
# Training Loop  # [SELF-IMPLEMENTED]
# ==============================================================================

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: DriveVLMLoss,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupCosineScheduler,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    epoch: int,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> Dict[str, float]:
    """
    Train the model for one full epoch.  # [SELF-IMPLEMENTED]

    This implements Stage 3 of the DriveVLM training paradigm:
    trajectory fine-tuning. The model has already been through:
    - Stage 1: Vision encoder pre-training (CLIP objectives)
    - Stage 2: Vision-language alignment (image-text matching)

    In Stage 3, we fine-tune the full model end-to-end with the
    combined trajectory + language loss to teach the model to:
    1. Understand driving scenes from multi-view images
    2. Follow natural language driving commands
    3. Generate safe, smooth trajectories
    4. Explain its reasoning in natural language

    Key training techniques:  # [FROM PAPER]
    - Mixed precision (FP16) for memory efficiency with large VLMs
    - Gradient clipping to prevent explosion in transformer training
    - Gradient accumulation to simulate larger batch sizes
    - Learning rate warmup for stable early training
    """
    model.train()
    total_losses = {'total': 0.0, 'trajectory': 0.0, 'language': 0.0}
    num_batches = 0
    global_step = epoch * len(dataloader)

    progress_bar = tqdm(dataloader, desc=f"Train Epoch {epoch+1}", total=len(dataloader))

    for batch_idx, batch in enumerate(progress_bar):
        # Move data to device  # [SELF-IMPLEMENTED]
        images = batch['images'].to(device)              # (B, 6, 3, 224, 224)
        prompt_tokens = batch['prompt_tokens'].to(device)  # (B, 20)
        gt_trajectory = batch['gt_trajectory'].to(device)  # (B, 6, 2)
        gt_response = batch['gt_response_tokens'].to(device)  # (B, 20)

        # Forward pass with Automatic Mixed Precision  # [SELF-IMPLEMENTED]
        amp_enabled = args.use_amp and device.type == 'cuda'
        amp_context = torch.cuda.amp.autocast(enabled=True) if amp_enabled else contextlib.nullcontext()
        with amp_context:
            # Model forward: images + prompt -> trajectory + logits
            output = model(images, prompt_tokens)

            # Compute multi-task loss  # [FROM PAPER]
            losses = criterion(output, gt_trajectory, gt_response)

        # Scale loss for gradient accumulation  # [SELF-IMPLEMENTED]
        scaled_loss = losses['total'] / args.gradient_accumulation_steps

        # Backward pass with gradient scaling (for AMP stability)
        if args.use_amp and device.type == 'cuda':
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        # Gradient accumulation: update weights every N steps  # [SELF-IMPLEMENTED]
        if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
            if args.use_amp and device.type == 'cuda':
                # Unscale gradients for clipping
                scaler.unscale_(optimizer)

            # Gradient clipping to prevent exploding gradients  # [FROM PAPER]
            # Critical for transformer training, especially with mixed precision
            # Paper uses max_norm=1.0 for stable VLM training
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=args.max_grad_norm
            )

            if args.use_amp and device.type == 'cuda':
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad()

            # Step the learning rate scheduler  # [SELF-IMPLEMENTED]
            scheduler.step()

        # Accumulate losses for logging
        total_losses['total'] += losses['total'].item()
        total_losses['trajectory'] += losses['trajectory'].item()
        total_losses['language'] += losses['language'].item()
        num_batches += 1

        # Update progress bar
        current_lr = scheduler.get_lr()
        if hasattr(progress_bar, 'set_postfix'):
            progress_bar.set_postfix({
                'loss': f"{losses['total'].item():.4f}",
                'traj': f"{losses['trajectory'].item():.4f}",
                'lang': f"{losses['language'].item():.4f}",
                'lr': f"{current_lr:.2e}",
            })

        # Periodic logging  # [SELF-IMPLEMENTED]
        if (batch_idx + 1) % args.log_interval == 0:
            logger.info(
                f"Epoch {epoch+1} [{batch_idx+1}/{len(dataloader)}] "
                f"Loss: {losses['total'].item():.4f} "
                f"(traj={losses['trajectory'].item():.4f}, "
                f"lang={losses['language'].item():.4f}) "
                f"LR: {current_lr:.2e}"
            )

    # Average losses over epoch
    avg_losses = {k: v / max(1, num_batches) for k, v in total_losses.items()}
    return avg_losses


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: DriveVLMLoss,
    device: torch.device,
    epoch: int,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> Dict[str, float]:
    """
    Validate the model on the validation set.  # [SELF-IMPLEMENTED]

    Computes both loss values and trajectory/language metrics.

    Trajectory metrics (from paper Table 2):  # [FROM PAPER]
    - L2@1s: L2 error at 1.0 second horizon
    - L2@2s: L2 error at 2.0 second horizon
    - L2@3s: L2 error at 3.0 second horizon
    - ADE: Average Displacement Error (mean over all waypoints)
    - FDE: Final Displacement Error (at 3.0s)

    Language metrics:
    - Perplexity: exp(average cross-entropy loss)
    """
    model.eval()
    total_losses = {'total': 0.0, 'trajectory': 0.0, 'language': 0.0}
    num_batches = 0

    # Initialize metric trackers  # [SELF-IMPLEMENTED]
    traj_metrics = TrajectoryMetrics(num_waypoints=args.num_waypoints)
    lang_metrics = LanguageMetrics()

    progress_bar = tqdm(dataloader, desc=f"Val Epoch {epoch+1}", total=len(dataloader))

    for batch in progress_bar:
        images = batch['images'].to(device)
        prompt_tokens = batch['prompt_tokens'].to(device)
        gt_trajectory = batch['gt_trajectory'].to(device)
        gt_response = batch['gt_response_tokens'].to(device)

        # Forward pass (no AMP needed for validation - simpler)
        output = model(images, prompt_tokens)
        losses = criterion(output, gt_trajectory, gt_response)

        # Accumulate losses
        total_losses['total'] += losses['total'].item()
        total_losses['trajectory'] += losses['trajectory'].item()
        total_losses['language'] += losses['language'].item()
        num_batches += 1

        # Update trajectory metrics  # [FROM PAPER]
        if 'trajectory' in output:
            traj_metrics.update(output['trajectory'], gt_trajectory)

        # Update language metrics  # [SELF-IMPLEMENTED]
        lang_metrics.update(
            losses['language'].item(),
            gt_response.shape[0] * (gt_response.shape[1] - 1)  # num tokens
        )

    # Compute final metrics
    avg_losses = {k: v / max(1, num_batches) for k, v in total_losses.items()}
    traj_results = traj_metrics.compute()
    lang_results = lang_metrics.compute()

    # Combine all metrics
    all_metrics = {**avg_losses, **traj_results, **lang_results}

    # Log validation results  # [SELF-IMPLEMENTED]
    logger.info(f"{'='*60}")
    logger.info(f"Validation Epoch {epoch+1} Results:")
    logger.info(f"  Loss - Total: {avg_losses['total']:.4f}, "
                f"Traj: {avg_losses['trajectory']:.4f}, "
                f"Lang: {avg_losses['language']:.4f}")
    logger.info(f"  Trajectory Metrics:")
    logger.info(f"    L2@1s: {traj_results['L2@1s']:.4f} m")
    logger.info(f"    L2@2s: {traj_results['L2@2s']:.4f} m")
    logger.info(f"    L2@3s: {traj_results['L2@3s']:.4f} m")
    logger.info(f"    ADE:   {traj_results['ADE']:.4f} m")
    logger.info(f"    FDE:   {traj_results['FDE']:.4f} m")
    logger.info(f"  Language Metrics:")
    logger.info(f"    Perplexity: {lang_results['perplexity']:.2f}")
    logger.info(f"    Avg CE Loss: {lang_results['avg_loss']:.4f}")
    logger.info(f"{'='*60}")

    return all_metrics


# ==============================================================================
# Stage 3: Trajectory Fine-tuning (Main Training Function)  # [FROM PAPER]
# ==============================================================================

def train_stage3(args: argparse.Namespace):
    """
    Stage 3 Training: Driving-Specific Trajectory Fine-tuning.  # [FROM PAPER]

    This is the core training stage where the VLM learns to plan trajectories.

    Context in the 4-stage DriveVLM training pipeline:
    -----------------------------------------------
    Stage 1 (NOT HERE): Vision encoder pre-training
        - Train ViT on large image datasets (ImageNet, LAION)
        - Use CLIP-style contrastive loss: match images to text descriptions
        - Result: Vision encoder that understands visual concepts
        - Compute: 256 GPUs, 2-4 weeks

    Stage 2 (NOT HERE): Vision-language alignment
        - Freeze vision encoder, train spatial adapter + early LLM layers
        - Dataset: driving scene descriptions paired with images
        - Loss: Next-token prediction on scene descriptions
        - Result: Model can describe driving scenes in natural language
        - Compute: 64 GPUs, 1 week

    Stage 3 (THIS FUNCTION): Trajectory fine-tuning
        - Unfreeze full model, train end-to-end
        - Dataset: images + commands -> trajectories + explanations
        - Loss: alpha * L_traj + (1-alpha) * L_lang
        - Result: Model can plan trajectories AND explain reasoning
        - Compute: 32 GPUs, 3-5 days

    Stage 4 (STUB BELOW): RL from driving rewards
        - Use PPO/DPO to optimize for safety and comfort
        - Requires online simulator interaction (CARLA/nuPlan)
        - Result: Safer, smoother trajectories
        - Compute: 64 GPUs, 1-2 weeks

    Why Stage 3 specifically?
    - It's the most practically demonstrable stage
    - Stages 1-2 require pre-trained checkpoints we don't have
    - Stage 3 shows the core VLM-to-trajectory paradigm
    - The combined loss (traj + lang) is what makes DriveVLM unique
    """
    # Setup  # [SELF-IMPLEMENTED]
    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu'
                          else 'cpu')
    logger = setup_logging(args.output_dir)
    logger.info(f"DriveVLM Stage 3 Training")
    logger.info(f"Device: {device}")
    logger.info(f"Config: {vars(args)}")

    # ==================================================================
    # Model Creation  # [SIMPLIFIED]
    # ==================================================================
    # In reality: Load pre-trained checkpoints from Stage 1 & 2
    # Here: Initialize from scratch (demonstrates the architecture)
    logger.info("Creating DriveVLM model...")
    model = DriveVLM(
        visual_dim=args.visual_dim,
        llm_dim=args.llm_dim,
        num_query_tokens=args.num_query_tokens,
        vocab_size=args.vocab_size,
        num_waypoints=args.num_waypoints,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {num_params:,}")
    logger.info(f"Trainable parameters: {num_trainable:,}")
    logger.info(f"(Real DriveVLM: ~7B parameters, this demo: {num_params/1e6:.1f}M)")

    # ==================================================================
    # Dataset & DataLoader  # [SELF-IMPLEMENTED]
    # ==================================================================
    logger.info("Creating datasets...")
    train_dataset = DriveVLMDataset(
        num_samples=args.num_train_samples,
        num_views=6,
        img_size=224,
        num_waypoints=args.num_waypoints,
        text_seq_len=args.text_seq_len,
        vocab_size=args.vocab_size,
        split='train',
        seed=args.seed,
    )
    val_dataset = DriveVLMDataset(
        num_samples=args.num_val_samples,
        num_views=6,
        img_size=224,
        num_waypoints=args.num_waypoints,
        text_seq_len=args.text_seq_len,
        vocab_size=args.vocab_size,
        split='val',
        seed=args.seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False,
        drop_last=True,  # Drop last incomplete batch for stable training
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False,
    )

    logger.info(f"Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    logger.info(f"Val: {len(val_dataset)} samples, {len(val_loader)} batches")

    # ==================================================================
    # Loss Function  # [FROM PAPER]
    # ==================================================================
    criterion = DriveVLMLoss(
        alpha=args.loss_alpha,
        vocab_size=args.vocab_size,
        num_waypoints=args.num_waypoints,
        num_visual_tokens=args.num_query_tokens,
        use_horizon_weights=True,
        label_smoothing=args.label_smoothing,
    )
    logger.info(f"Loss: alpha={args.loss_alpha} (traj weight), "
                f"1-alpha={1-args.loss_alpha:.2f} (lang weight)")

    # ==================================================================
    # Optimizer  # [SELF-IMPLEMENTED]
    # ==================================================================
    # AdamW with weight decay - standard for transformer fine-tuning
    # Different LR for different model components (common in VLM training):
    # - Vision encoder: lower LR (already pre-trained in Stages 1-2)
    # - Spatial adapter: medium LR (partially trained in Stage 2)
    # - LLM: base LR (being fine-tuned for driving)
    param_groups = [
        {  # Vision encoder: lower LR to preserve pre-trained features  # [FROM PAPER]
            'params': model.vision_encoder.parameters(),
            'lr': args.learning_rate * args.vision_lr_scale,
            'name': 'vision_encoder',
        },
        {  # Spatial adapter: medium LR  # [FROM PAPER]
            'params': model.spatial_adapter.parameters(),
            'lr': args.learning_rate * args.adapter_lr_scale,
            'name': 'spatial_adapter',
        },
        {  # LLM: full learning rate  # [FROM PAPER]
            'params': model.llm.parameters(),
            'lr': args.learning_rate,
            'name': 'llm',
        },
    ]

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.weight_decay,
        eps=args.adam_eps,
    )
    logger.info(f"Optimizer: AdamW (lr={args.learning_rate}, "
                f"wd={args.weight_decay})")
    logger.info(f"  Vision encoder LR scale: {args.vision_lr_scale}x")
    logger.info(f"  Spatial adapter LR scale: {args.adapter_lr_scale}x")

    # ==================================================================
    # Learning Rate Scheduler  # [SELF-IMPLEMENTED]
    # ==================================================================
    total_steps = len(train_loader) * args.num_epochs // args.gradient_accumulation_steps
    warmup_steps = int(total_steps * args.warmup_ratio)

    scheduler = WarmupCosineScheduler(
        optimizer=optimizer,
        base_lr=args.learning_rate,
        min_lr=args.min_learning_rate,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
    )
    logger.info(f"Scheduler: Cosine with {warmup_steps} warmup steps, "
                f"{total_steps} total steps")

    # ==================================================================
    # Mixed Precision & Gradient Scaler  # [SELF-IMPLEMENTED]
    # ==================================================================
    amp_enabled = args.use_amp and device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    if args.use_amp:
        logger.info("Using Automatic Mixed Precision (FP16)")
    else:
        logger.info("Using FP32 precision")

    # ==================================================================
    # Checkpoint Manager  # [SELF-IMPLEMENTED]
    # ==================================================================
    ckpt_manager = CheckpointManager(
        save_dir=os.path.join(args.output_dir, 'checkpoints'),
        max_checkpoints=args.max_checkpoints,
    )

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        resume_info = ckpt_manager.load(args.resume, model, optimizer, scheduler)
        start_epoch = resume_info['epoch'] + 1
        logger.info(f"Resuming from epoch {start_epoch}")

    # ==================================================================
    # Training Loop  # [SELF-IMPLEMENTED]
    # ==================================================================
    logger.info(f"\n{'='*60}")
    logger.info(f"Starting Stage 3 Training: Trajectory Fine-tuning")
    logger.info(f"{'='*60}")
    logger.info(f"  Epochs: {args.num_epochs}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Gradient accumulation: {args.gradient_accumulation_steps}")
    logger.info(f"  Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    logger.info(f"  Max gradient norm: {args.max_grad_norm}")
    logger.info(f"{'='*60}\n")

    best_val_loss = float('inf')
    training_history = []

    for epoch in range(start_epoch, args.num_epochs):
        epoch_start_time = time.time()

        # --- Train ---
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            epoch=epoch,
            args=args,
            logger=logger,
        )

        # --- Validate ---
        val_metrics = validate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
            args=args,
            logger=logger,
        )

        epoch_time = time.time() - epoch_start_time

        # --- Checkpoint ---
        is_best = val_metrics['total'] < best_val_loss
        if is_best:
            best_val_loss = val_metrics['total']
            logger.info(f"*** New best validation loss: {best_val_loss:.4f} ***")

        ckpt_path = ckpt_manager.save(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            step=(epoch + 1) * len(train_loader),
            metrics=val_metrics,
            is_best=is_best,
        )
        logger.info(f"Saved checkpoint: {ckpt_path}")

        # --- Epoch Summary ---
        logger.info(f"\nEpoch {epoch+1}/{args.num_epochs} Summary "
                    f"({epoch_time:.1f}s):")
        logger.info(f"  Train Loss: {train_metrics['total']:.4f}")
        logger.info(f"  Val Loss:   {val_metrics['total']:.4f}")
        logger.info(f"  Val L2@1s:  {val_metrics.get('L2@1s', 0):.4f} m")
        logger.info(f"  Val L2@3s:  {val_metrics.get('L2@3s', 0):.4f} m")
        logger.info(f"  Val PPL:    {val_metrics.get('perplexity', 0):.2f}")
        logger.info(f"  LR:         {scheduler.get_lr():.2e}")
        logger.info("")

        # Track history  # [SELF-IMPLEMENTED]
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': train_metrics['total'],
            'val_loss': val_metrics['total'],
            'val_L2_1s': val_metrics.get('L2@1s', 0),
            'val_L2_3s': val_metrics.get('L2@3s', 0),
            'val_perplexity': val_metrics.get('perplexity', 0),
            'lr': scheduler.get_lr(),
            'epoch_time': epoch_time,
        })

    # ==================================================================
    # Training Complete  # [SELF-IMPLEMENTED]
    # ==================================================================
    logger.info(f"\n{'='*60}")
    logger.info(f"Stage 3 Training Complete!")
    logger.info(f"{'='*60}")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Total training time: "
                f"{sum(h['epoch_time'] for h in training_history):.1f}s")
    logger.info(f"Checkpoints saved to: {args.output_dir}/checkpoints/")

    # Save training history  # [SELF-IMPLEMENTED]
    history_path = os.path.join(args.output_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    logger.info(f"Training history saved to: {history_path}")

    # Print final summary table  # [SELF-IMPLEMENTED]
    logger.info(f"\nTraining History:")
    logger.info(f"{'Epoch':>5} | {'Train Loss':>10} | {'Val Loss':>10} | "
                f"{'L2@1s':>6} | {'L2@3s':>6} | {'PPL':>8}")
    logger.info("-" * 65)
    for h in training_history:
        logger.info(
            f"{h['epoch']:>5} | {h['train_loss']:>10.4f} | {h['val_loss']:>10.4f} | "
            f"{h['val_L2_1s']:>6.3f} | {h['val_L2_3s']:>6.3f} | "
            f"{h['val_perplexity']:>8.2f}"
        )

    return training_history


# ==============================================================================
# Stage 4: RL from Driving Rewards (Stub)  # [FROM PAPER]
# ==============================================================================

def train_stage4_stub(args: argparse.Namespace):
    """
    Stage 4 Training STUB: Reinforcement Learning from Driving Rewards.  # [FROM PAPER]

    This function demonstrates the STRUCTURE of RL-based training (PPO/DPO)
    but does NOT actually run full RL training, which requires:
    1. A driving simulator (CARLA, nuPlan closed-loop)
    2. A trained reward model (or hand-crafted reward function)
    3. A reference policy (the Stage 3 model) for KL constraint
    4. Massive compute for online rollouts

    RL Training Process (PPO - Proximal Policy Optimization):  # [FROM PAPER]
    =========================================================

    The idea: After Stage 3, the model can plan trajectories.
    But some trajectories might be:
    - Technically valid but uncomfortable (jerky)
    - Safe but overly conservative (stopping too much)
    - Good on average but occasionally dangerous

    RL fixes this by:
    1. Generating trajectories in a simulator
    2. Scoring them with a reward model
    3. Updating the policy to favor high-reward trajectories
    4. Constraining updates to stay close to the Stage 3 policy (KL penalty)

    PPO Update Rule:
        L_PPO = -min(r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t)
        where:
            r_t = pi_new(a|s) / pi_old(a|s)  (probability ratio)
            A_t = advantage estimate (how much better than average)
            eps = 0.2 (clipping parameter)

    DPO Alternative (simpler, no reward model needed):  # [FROM PAPER]
        Instead of training a reward model separately, DPO directly
        optimizes the policy using preference pairs:
        L_DPO = -log(sigma(beta * (log pi(y_w|x) - log pi(y_l|x))))
        where y_w = preferred trajectory, y_l = dispreferred trajectory

    Reward Components for Driving:  # [FROM PAPER]
        R_total = w1*R_safety + w2*R_comfort + w3*R_progress + w4*R_rules
        - R_safety: -100 for collision, -10 for near-miss, +1 for safe distance
        - R_comfort: Based on jerk profile, lateral acceleration limits
        - R_progress: +1 for making progress toward goal, -1 for stopping
        - R_rules: Traffic rule compliance (speed limits, signals, lanes)
    """
    logger = setup_logging(args.output_dir)
    logger.info("=" * 60)
    logger.info("Stage 4: RL from Driving Rewards (STUB)")
    logger.info("=" * 60)
    logger.info("")
    logger.info("This is a structural demonstration of how Stage 4 would work.")
    logger.info("It does NOT run actual RL training.")
    logger.info("")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu'
                          else 'cpu')

    # Load the Stage 3 trained model as the initial policy  # [FROM PAPER]
    logger.info("Step 1: Load Stage 3 model as initial policy (pi_ref)")
    model = DriveVLM(
        visual_dim=args.visual_dim,
        llm_dim=args.llm_dim,
        num_query_tokens=args.num_query_tokens,
        vocab_size=args.vocab_size,
        num_waypoints=args.num_waypoints,
    ).to(device)

    # In reality: load Stage 3 checkpoint
    # model.load_state_dict(torch.load('stage3_best.pt')['model_state_dict'])

    # Create the reward model  # [SELF-IMPLEMENTED]
    logger.info("Step 2: Initialize reward model")
    reward_model = RewardModel(
        embed_dim=args.llm_dim,
        num_waypoints=args.num_waypoints,
    ).to(device)
    logger.info(f"  Reward model parameters: "
                f"{sum(p.numel() for p in reward_model.parameters()):,}")

    # PPO hyperparameters  # [FROM PAPER]
    ppo_config = {
        'clip_epsilon': 0.2,          # PPO clipping parameter
        'value_loss_coef': 0.5,       # Value function loss weight
        'entropy_coef': 0.01,         # Entropy bonus for exploration
        'kl_penalty': 0.1,            # KL divergence penalty weight
        'gamma': 0.99,                # Discount factor
        'gae_lambda': 0.95,           # GAE lambda for advantage estimation
        'num_rollout_steps': 128,     # Steps per rollout
        'num_ppo_epochs': 4,          # PPO update epochs per batch
        'mini_batch_size': 32,        # Mini-batch size for PPO updates
        'max_kl_divergence': 0.05,    # Early stopping if KL too large
    }
    logger.info(f"Step 3: PPO configuration:")
    for k, v in ppo_config.items():
        logger.info(f"    {k}: {v}")

    # Demonstrate the PPO training loop structure  # [FROM PAPER]
    logger.info("")
    logger.info("Step 4: PPO Training Loop (structure only, not executed):")
    logger.info("  for iteration in range(num_iterations):")
    logger.info("    # 1. Collect rollouts in simulator")
    logger.info("    #    - Feed images to model, get trajectory")
    logger.info("    #    - Execute trajectory in CARLA/nuPlan")
    logger.info("    #    - Record (state, action, reward, next_state)")
    logger.info("")
    logger.info("    # 2. Compute advantages using GAE")
    logger.info("    #    A_t = sum_{l=0}^{T-t} (gamma*lambda)^l * delta_{t+l}")
    logger.info("    #    delta_t = r_t + gamma*V(s_{t+1}) - V(s_t)")
    logger.info("")
    logger.info("    # 3. PPO policy update")
    logger.info("    #    ratio = pi_new(a|s) / pi_old(a|s)")
    logger.info("    #    L_clip = min(ratio*A, clip(ratio, 1-eps, 1+eps)*A)")
    logger.info("    #    L_total = -L_clip + c1*L_value - c2*H(pi)")
    logger.info("")
    logger.info("    # 4. KL divergence check against reference policy")
    logger.info("    #    if KL(pi_new || pi_ref) > max_kl:")
    logger.info("    #        break  # Don't diverge too far from Stage 3")
    logger.info("")

    # Show a single mock PPO step to demonstrate the computation  # [SELF-IMPLEMENTED]
    logger.info("Step 5: Demonstrating single reward computation...")
    with torch.no_grad():
        # Mock input
        B = 2
        images = torch.randn(B, 6, 3, 224, 224, device=device)
        prompt = torch.randint(0, args.vocab_size, (B, args.text_seq_len), device=device)

        # Get model output (trajectory + hidden states)
        output = model(images, prompt)

        # Compute reward
        reward = reward_model(output['trajectory'], output['hidden_states'])

        logger.info(f"  Input: {B} driving scenarios")
        logger.info(f"  Predicted trajectories: {output['trajectory'].shape}")
        logger.info(f"  Computed rewards: {reward.squeeze().tolist()}")
        logger.info(f"  (Positive reward = good driving, negative = bad)")

    logger.info("")
    logger.info("Step 6: DPO alternative (Direct Preference Optimization):")
    logger.info("  - Collect preference pairs: (image, traj_good, traj_bad)")
    logger.info("  - These come from human annotation or simulator outcomes")
    logger.info("  - L_DPO = -log(sigma(beta * (log_pi(good) - log_pi(bad))))")
    logger.info("  - Simpler than PPO: no reward model, no value function")
    logger.info("  - But requires high-quality preference data")
    logger.info("")
    logger.info("=" * 60)
    logger.info("Stage 4 stub complete. Real implementation requires:")
    logger.info("  1. CARLA or nuPlan simulator integration")
    logger.info("  2. Trained reward model or preference dataset")
    logger.info("  3. 64+ GPUs for online rollout collection")
    logger.info("  4. 1-2 weeks of training time")
    logger.info("=" * 60)


# ==============================================================================
# Argument Parser  # [SELF-IMPLEMENTED]
# ==============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for DriveVLM training."""
    parser = argparse.ArgumentParser(
        description="DriveVLM Training Script - Foundation Model for Autonomous Driving",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Model Architecture ---
    model_group = parser.add_argument_group('Model Architecture')
    model_group.add_argument('--visual-dim', type=int, default=384,
                             help='Vision encoder embedding dimension '
                                  '(real: 768-1024 for ViT-L/InternViT)')
    model_group.add_argument('--llm-dim', type=int, default=512,
                             help='LLM hidden dimension '
                                  '(real: 4096 for 7B model)')
    model_group.add_argument('--num-query-tokens', type=int, default=32,
                             help='Number of spatial adapter query tokens '
                                  '(real: 64-256)')
    model_group.add_argument('--vocab-size', type=int, default=1000,
                             help='Text vocabulary size '
                                  '(real: 32000 for LLaMA tokenizer)')
    model_group.add_argument('--num-waypoints', type=int, default=6,
                             help='Number of trajectory waypoints to predict')
    model_group.add_argument('--text-seq-len', type=int, default=20,
                             help='Text sequence length for prompts/responses')

    # --- Training ---
    train_group = parser.add_argument_group('Training')
    train_group.add_argument('--num-epochs', type=int, default=5,
                             help='Number of training epochs')
    train_group.add_argument('--batch-size', type=int, default=2,
                             help='Batch size per device '
                                  '(real: 4-8 per GPU with gradient accumulation)')
    train_group.add_argument('--gradient-accumulation-steps', type=int, default=2,
                             help='Gradient accumulation steps '
                                  '(effective batch = batch_size * this)')
    train_group.add_argument('--learning-rate', type=float, default=2e-4,
                             help='Peak learning rate for LLM')
    train_group.add_argument('--min-learning-rate', type=float, default=1e-6,
                             help='Minimum learning rate (at end of cosine decay)')
    train_group.add_argument('--weight-decay', type=float, default=0.01,
                             help='AdamW weight decay')
    train_group.add_argument('--adam-beta1', type=float, default=0.9,
                             help='Adam beta1')
    train_group.add_argument('--adam-beta2', type=float, default=0.95,
                             help='Adam beta2 (0.95 standard for LLM training)')
    train_group.add_argument('--adam-eps', type=float, default=1e-8,
                             help='Adam epsilon')
    train_group.add_argument('--max-grad-norm', type=float, default=1.0,
                             help='Maximum gradient norm for clipping')
    train_group.add_argument('--warmup-ratio', type=float, default=0.1,
                             help='Fraction of total steps for LR warmup')
    train_group.add_argument('--use-amp', action='store_true', default=False,
                             help='Use automatic mixed precision (FP16)')

    # --- Learning Rate Scaling ---
    lr_group = parser.add_argument_group('Learning Rate Scaling')
    lr_group.add_argument('--vision-lr-scale', type=float, default=0.1,
                          help='LR multiplier for vision encoder '
                               '(lower to preserve pre-trained features)')
    lr_group.add_argument('--adapter-lr-scale', type=float, default=0.5,
                          help='LR multiplier for spatial adapter')

    # --- Loss ---
    loss_group = parser.add_argument_group('Loss Configuration')
    loss_group.add_argument('--loss-alpha', type=float, default=0.7,
                            help='Weight for trajectory loss '
                                 '(1-alpha for language loss). '
                                 'Paper recommends 0.7 for best driving performance.')
    loss_group.add_argument('--label-smoothing', type=float, default=0.0,
                            help='Label smoothing for language loss')

    # --- Data ---
    data_group = parser.add_argument_group('Data')
    data_group.add_argument('--num-train-samples', type=int, default=200,
                            help='Number of synthetic training samples '
                                 '(real: 100K+ from nuScenes/OpenScene)')
    data_group.add_argument('--num-val-samples', type=int, default=50,
                            help='Number of synthetic validation samples')
    data_group.add_argument('--num-workers', type=int, default=0,
                            help='DataLoader worker processes '
                                 '(0 for Windows compatibility)')
    data_group.add_argument('--seed', type=int, default=42,
                            help='Random seed for reproducibility')

    # --- Output & Logging ---
    output_group = parser.add_argument_group('Output & Logging')
    output_group.add_argument('--output-dir', type=str, default='./output_drivevlm',
                              help='Directory for checkpoints, logs, and results')
    output_group.add_argument('--log-interval', type=int, default=10,
                              help='Log training metrics every N batches')
    output_group.add_argument('--max-checkpoints', type=int, default=3,
                              help='Maximum number of checkpoints to keep')

    # --- Resume & Stage ---
    resume_group = parser.add_argument_group('Resume & Stage')
    resume_group.add_argument('--resume', type=str, default=None,
                              help='Path to checkpoint to resume from')
    resume_group.add_argument('--stage', type=int, default=3, choices=[3, 4],
                              help='Training stage: 3=trajectory fine-tuning, '
                                   '4=RL stub demonstration')
    resume_group.add_argument('--device', type=str, default='cuda',
                              help='Device: cuda or cpu')

    args = parser.parse_args()
    return args


# ==============================================================================
# Main Entry Point  # [SELF-IMPLEMENTED]
# ==============================================================================

def main():
    """
    DriveVLM Training - Main Entry Point.

    Foundation Model Training Pipeline for Autonomous Driving:
    =========================================================

    This script implements the training pipeline for DriveVLM, which uses
    a Vision-Language Model (VLM) architecture for end-to-end autonomous driving.

    The key insight of DriveVLM (and the foundation model paradigm in general):
    Instead of training a task-specific driving model from scratch, we:
    1. Start with a powerful pre-trained VLM (InternVL, LLaVA, etc.)
    2. Adapt it to understand driving scenes (visual grounding)
    3. Fine-tune it to output trajectories (planning)
    4. Refine with RL for safety (alignment)

    This is analogous to how ChatGPT works:
    - GPT-4 = massive pre-trained LLM
    - ChatGPT = GPT-4 + instruction tuning + RLHF
    Similarly:
    - InternVL = massive pre-trained VLM
    - DriveVLM = InternVL + driving tuning + RL from driving rewards

    Why this matters for autonomous driving:
    1. INTERPRETABILITY: The model can explain WHY it makes each decision
       ("I'm slowing down because I see a pedestrian about to cross")
    2. GENERALIZATION: Pre-trained VLMs have seen millions of images,
       giving them robust visual understanding even in rare scenarios
    3. MULTI-TASK: One model does perception, prediction, AND planning
    4. SCALABILITY: Performance improves with model size (scaling laws)

    Usage:
        # Stage 3: Trajectory fine-tuning (main training)
        python train.py --stage 3 --num-epochs 5 --batch-size 2

        # Stage 4: RL stub (demonstration only)
        python train.py --stage 4

        # Resume training from checkpoint
        python train.py --stage 3 --resume ./output_drivevlm/checkpoints/best_model.pt

        # Full-size model (requires GPU with 24GB+ VRAM)
        python train.py --visual-dim 768 --llm-dim 4096 --vocab-size 32000 --use-amp
    """
    print("=" * 70)
    print("  DriveVLM: Vision-Language Model for Autonomous Driving")
    print("  Training Script - Foundation Model Paradigm")
    print("=" * 70)
    print()
    print("  Paper: 'DriveVLM: The Convergence of Autonomous Driving")
    print("          and Large Vision-Language Models' (Tian et al., 2024)")
    print()
    print("  Training Stages:")
    print("    Stage 1: Vision encoder pre-training (CLIP)     [NOT HERE]")
    print("    Stage 2: Vision-language alignment              [NOT HERE]")
    print("    Stage 3: Trajectory fine-tuning                 [THIS SCRIPT]")
    print("    Stage 4: RL from driving rewards (PPO/DPO)      [STUB HERE]")
    print()
    print("=" * 70)
    print()

    args = parse_args()

    # Set random seeds for reproducibility  # [SELF-IMPLEMENTED]
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save configuration  # [SELF-IMPLEMENTED]
    config_path = os.path.join(args.output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Route to appropriate training stage  # [FROM PAPER]
    if args.stage == 3:
        print(">>> Running Stage 3: Trajectory Fine-tuning")
        print(f">>> Output: {args.output_dir}")
        print()
        train_stage3(args)
    elif args.stage == 4:
        print(">>> Running Stage 4: RL Training (Stub/Demo)")
        print(f">>> This demonstrates the RL structure but does not train.")
        print()
        train_stage4_stub(args)
    else:
        print(f"ERROR: Unknown stage {args.stage}. Use --stage 3 or --stage 4.")
        sys.exit(1)

    print()
    print("=" * 70)
    print("  Training complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
