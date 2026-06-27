"""
InterFuser Training Script
===========================
ATTRIBUTION:
- Loss functions: Based on InterFuser paper (Shao et al., CoRL 2022)
  - Waypoint regression: L1 loss (Section 3.3)
  - Density map prediction: Focal loss (Section 3.2)
  - Safety score: Binary cross-entropy (Section 3.4)
  - Traffic light: Cross-entropy classification
- Training strategy: Multi-task imitation learning (CARLA expert data)
- Safety score labeling: From paper - labeled based on TTC and collision proximity
- Implementation: Self-implemented in PyTorch (simplified from official InterFuser)
- Synthetic dataset: Self-implemented for demonstration (real uses CARLA data)
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm not installed  # [SELF-IMPLEMENTED]
    def tqdm(iterable, **kwargs):
        return iterable

from model import InterFuser


# =============================================================================
# Synthetic Dataset
# =============================================================================

class InterFuserDataset(Dataset):  # [SELF-IMPLEMENTED]
    """
    Synthetic dataset for InterFuser training demonstration.

    In real usage, this would load CARLA expert driving data including:
    - Multi-view camera images (front, left, right)
    - LiDAR BEV representation
    - Expert waypoints from privileged planner
    - Density maps from object annotations
    - Safety scores from TTC computation
    - Traffic light state labels

    Generates:
        multi_view_images: (3, 3, 224, 224) - front, left, right RGB images
        lidar_bev: (2, 224, 224) - LiDAR bird's eye view (height + intensity)
        waypoints: (4, 2) - future waypoint positions (x, y)
        density_map: (1, 32, 32) - traffic density BEV map
        safety_score: (1,) - binary safety label
        traffic_light: (4,) - one-hot traffic light state [red, yellow, green, none]
    """

    def __init__(self, num_samples: int = 1000, num_waypoints: int = 4,
                 bev_size: int = 32, img_size: int = 224, split: str = "train"):
        super().__init__()
        self.num_samples = num_samples
        self.num_waypoints = num_waypoints
        self.bev_size = bev_size
        self.img_size = img_size
        self.split = split

        # Pre-generate random seed offsets for reproducibility  # [SELF-IMPLEMENTED]
        base_seed = 42 if split == "train" else 1234
        self.rng = np.random.RandomState(base_seed)
        self.seeds = self.rng.randint(0, 2**31, size=num_samples)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        rng = np.random.RandomState(self.seeds[idx])  # [SELF-IMPLEMENTED]

        # Multi-view images: front, left, right (3 views x 3 channels x H x W)
        # Simulates camera inputs from three viewpoints  # [SELF-IMPLEMENTED]
        multi_view_images = rng.randn(3, 3, self.img_size, self.img_size).astype(np.float32)

        # LiDAR BEV: 2 channels (height map + intensity)  # [SELF-IMPLEMENTED]
        lidar_bev = rng.randn(2, self.img_size, self.img_size).astype(np.float32)

        # Waypoints: sequential future positions along a trajectory  # [SIMPLIFIED]
        # Real data: from privileged CARLA planner with access to ground truth
        waypoints = np.zeros((self.num_waypoints, 2), dtype=np.float32)
        for i in range(self.num_waypoints):
            # Simulate forward-moving trajectory with slight curvature
            waypoints[i, 0] = (i + 1) * 2.0 + rng.randn() * 0.5  # x (forward)
            waypoints[i, 1] = rng.randn() * 0.3  # y (lateral)

        # Density map: sparse occupancy in BEV space  # [SIMPLIFIED]
        # Real data: projected from 3D bounding box annotations
        density_map = np.zeros((1, self.bev_size, self.bev_size), dtype=np.float32)
        num_objects = rng.randint(2, 8)
        for _ in range(num_objects):
            cx, cy = rng.randint(2, self.bev_size - 2, size=2)
            w, h = rng.randint(1, 4, size=2)
            x1, x2 = max(0, cx - w), min(self.bev_size, cx + w)
            y1, y2 = max(0, cy - h), min(self.bev_size, cy + h)
            density_map[0, y1:y2, x1:x2] = 1.0

        # Safety score: binary label  # [FROM PAPER]
        # Paper: labeled based on time-to-collision (TTC) and collision proximity
        # TTC < threshold or proximity < threshold -> unsafe (0), else safe (1)
        safety_score = np.array([rng.choice([0.0, 1.0], p=[0.3, 0.7])],
                                dtype=np.float32)

        # Traffic light state: one-hot [red, yellow, green, none]  # [FROM PAPER]
        tl_state = rng.randint(0, 4)
        traffic_light = np.zeros(4, dtype=np.float32)
        traffic_light[tl_state] = 1.0

        return {
            "multi_view_images": torch.from_numpy(multi_view_images),
            "lidar_bev": torch.from_numpy(lidar_bev),
            "waypoints": torch.from_numpy(waypoints),
            "density_map": torch.from_numpy(density_map),
            "safety_score": torch.from_numpy(safety_score),
            "traffic_light": torch.from_numpy(traffic_light),
        }


# =============================================================================
# Loss Functions
# =============================================================================

class WaypointLoss(nn.Module):  # [FROM PAPER]
    """
    L1 regression loss for waypoint prediction (Section 3.3).

    The paper uses L1 loss between predicted and ground truth waypoints
    from the privileged expert planner.
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred_waypoints: torch.Tensor,
                gt_waypoints: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_waypoints: (B, num_wp, 2) predicted waypoints
            gt_waypoints: (B, num_wp, 2) ground truth waypoints
        Returns:
            Scalar L1 loss
        """
        return F.l1_loss(pred_waypoints, gt_waypoints)  # [FROM PAPER]


class DensityMapLoss(nn.Module):  # [FROM PAPER]
    """
    Focal loss for traffic density map prediction (Section 3.2).

    The paper uses focal loss to handle class imbalance in the density map,
    where most cells are empty (background) and few contain objects.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):  # [FROM PAPER]
        super().__init__()
        self.alpha = alpha  # Balance factor for positive/negative samples
        self.gamma = gamma  # Focusing parameter to down-weight easy examples

    def forward(self, pred_density: torch.Tensor,
                gt_density: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_density: (B, 1, H, W) predicted density map (after sigmoid)
            gt_density: (B, 1, H, W) ground truth binary density map
        Returns:
            Scalar focal loss
        """
        # Clamp predictions to avoid log(0)  # [SELF-IMPLEMENTED]
        pred = pred_density.clamp(min=1e-6, max=1.0 - 1e-6)

        # Focal loss computation  # [FROM PAPER]
        # FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
        bce = F.binary_cross_entropy(pred, gt_density, reduction='none')

        p_t = pred * gt_density + (1 - pred) * (1 - gt_density)
        alpha_t = self.alpha * gt_density + (1 - self.alpha) * (1 - gt_density)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma  # [FROM PAPER]

        loss = (focal_weight * bce).mean()
        return loss


class SafetyLoss(nn.Module):  # [FROM PAPER]
    """
    Binary cross-entropy loss for safety score prediction (Section 3.4).

    The paper trains the safety head to predict whether the current driving
    scenario is safe, based on TTC and collision proximity labels.
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred_safety: torch.Tensor,
                gt_safety: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_safety: (B, 1) predicted safety score (after sigmoid in model)
            gt_safety: (B, 1) ground truth safety label (0=unsafe, 1=safe)
        Returns:
            Scalar BCE loss
        """
        return F.binary_cross_entropy(pred_safety, gt_safety)  # [FROM PAPER]


class TrafficLightLoss(nn.Module):  # [FROM PAPER]
    """
    Cross-entropy classification loss for traffic light state prediction.

    The paper classifies traffic light into 4 states: red, yellow, green, none.
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred_tl: torch.Tensor,
                gt_tl: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_tl: (B, 4) predicted traffic light logits
            gt_tl: (B, 4) ground truth one-hot traffic light state
        Returns:
            Scalar cross-entropy loss
        """
        # Convert one-hot to class indices for cross-entropy  # [SELF-IMPLEMENTED]
        gt_labels = gt_tl.argmax(dim=1)
        return F.cross_entropy(pred_tl, gt_labels)  # [FROM PAPER]


class InterFuserLoss(nn.Module):  # [FROM PAPER]
    """
    Combined multi-task loss for InterFuser training.

    Total loss = w_wp * L_waypoint + w_dm * L_density + w_sf * L_safety + w_tl * L_traffic

    Loss weights are from the paper's multi-task training formulation.
    """

    def __init__(self, w_waypoint: float = 1.0, w_density: float = 1.0,
                 w_safety: float = 0.5, w_traffic: float = 0.5):  # [FROM PAPER]
        super().__init__()
        self.waypoint_loss = WaypointLoss()
        self.density_loss = DensityMapLoss()
        self.safety_loss = SafetyLoss()
        self.traffic_loss = TrafficLightLoss()

        # Multi-task loss weights from paper  # [FROM PAPER]
        self.w_waypoint = w_waypoint
        self.w_density = w_density
        self.w_safety = w_safety
        self.w_traffic = w_traffic

    def forward(self, predictions: dict, targets: dict) -> dict:
        """
        Args:
            predictions: dict from model forward pass
            targets: dict with ground truth tensors
        Returns:
            dict with individual losses and total loss
        """
        l_wp = self.waypoint_loss(predictions["waypoints"], targets["waypoints"])
        l_dm = self.density_loss(predictions["density_map"], targets["density_map"])
        l_sf = self.safety_loss(predictions["safety_score"], targets["safety_score"])
        l_tl = self.traffic_loss(predictions["traffic_light"], targets["traffic_light"])

        # Weighted sum  # [FROM PAPER]
        total = (self.w_waypoint * l_wp + self.w_density * l_dm +
                 self.w_safety * l_sf + self.w_traffic * l_tl)

        return {
            "total": total,
            "waypoint": l_wp,
            "density": l_dm,
            "safety": l_sf,
            "traffic": l_tl,
        }


# =============================================================================
# InterFuser Model Wrapper (with traffic light head)
# =============================================================================

class InterFuserWithTraffic(nn.Module):  # [SELF-IMPLEMENTED]
    """
    Wraps InterFuser model with an additional traffic light classification head.

    The original model.py does not include a traffic light head, but the paper
    describes multi-task outputs including traffic light state prediction.
    This wrapper adds that capability for training.
    """

    def __init__(self, d_model: int = 256, n_heads: int = 8,
                 num_layers: int = 6, num_waypoints: int = 4,
                 bev_size: int = 32, num_tl_classes: int = 4):
        super().__init__()
        self.base_model = InterFuser(
            d_model=d_model, n_heads=n_heads,
            num_layers=num_layers, num_waypoints=num_waypoints,
            bev_size=bev_size
        )

        # Traffic light classification head  # [FROM PAPER]
        # Predicts traffic light state: red, yellow, green, none
        self.traffic_light_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, num_tl_classes),
        )

        self.d_model = d_model

    def forward(self, front_img: torch.Tensor, left_img: torch.Tensor,
                right_img: torch.Tensor, lidar_bev: torch.Tensor) -> dict:
        """Forward pass with traffic light prediction added."""
        # Get base model outputs  # [SELF-IMPLEMENTED]
        outputs = self.base_model(front_img, left_img, right_img, lidar_bev)

        # Extract global feature for traffic light prediction  # [SELF-IMPLEMENTED]
        # Re-run encoder to get fused features (shares computation with base model)
        B = front_img.shape[0]
        images = [front_img, left_img, right_img]
        all_tokens = []
        for i, encoder in enumerate(self.base_model.view_encoders):
            feat = encoder(images[i])
            tokens = feat.flatten(2).permute(0, 2, 1)
            all_tokens.append(tokens)

        lidar_feat = self.base_model.lidar_encoder(lidar_bev)
        lidar_tokens = lidar_feat.flatten(2).permute(0, 2, 1)
        all_tokens.append(lidar_tokens)

        multi_modal_tokens = torch.cat(all_tokens, dim=1)
        fused = self.base_model.transformer(multi_modal_tokens)
        global_feat = fused.mean(dim=1)

        # Traffic light classification  # [FROM PAPER]
        tl_logits = self.traffic_light_head(global_feat)
        outputs["traffic_light"] = tl_logits

        return outputs


# =============================================================================
# Validation Metrics
# =============================================================================

def compute_waypoint_error(pred: torch.Tensor, gt: torch.Tensor) -> float:  # [SELF-IMPLEMENTED]
    """
    Average displacement error (ADE) for waypoint predictions.

    Args:
        pred: (B, num_wp, 2) predicted waypoints
        gt: (B, num_wp, 2) ground truth waypoints
    Returns:
        Mean L2 distance across all waypoints
    """
    # L2 distance per waypoint, averaged over batch and waypoints
    displacement = torch.norm(pred - gt, dim=-1)  # (B, num_wp)
    return displacement.mean().item()


def compute_density_iou(pred: torch.Tensor, gt: torch.Tensor,
                        threshold: float = 0.5) -> float:  # [SELF-IMPLEMENTED]
    """
    Intersection over Union for density map prediction.

    Args:
        pred: (B, 1, H, W) predicted density map (sigmoid output)
        gt: (B, 1, H, W) ground truth binary density map
        threshold: binarization threshold for predictions
    Returns:
        Mean IoU score
    """
    pred_binary = (pred > threshold).float()
    gt_binary = (gt > threshold).float()

    intersection = (pred_binary * gt_binary).sum(dim=(1, 2, 3))
    union = ((pred_binary + gt_binary) > 0).float().sum(dim=(1, 2, 3))

    # Avoid division by zero for empty maps  # [SELF-IMPLEMENTED]
    iou = intersection / (union + 1e-6)
    return iou.mean().item()


def compute_safety_auc(pred: torch.Tensor, gt: torch.Tensor) -> float:  # [SELF-IMPLEMENTED]
    """
    Approximate AUC for safety score prediction.

    Uses a simple threshold-based approximation since sklearn may not be available.
    For proper evaluation, use sklearn.metrics.roc_auc_score.

    Args:
        pred: (N, 1) predicted safety scores
        gt: (N, 1) ground truth safety labels
    Returns:
        Approximate AUC score
    """
    pred_np = pred.detach().cpu().numpy().flatten()
    gt_np = gt.detach().cpu().numpy().flatten()

    # Handle edge case: all same label  # [SELF-IMPLEMENTED]
    if len(np.unique(gt_np)) < 2:
        return 0.5

    # Simple AUC approximation via threshold sweep  # [SELF-IMPLEMENTED]
    thresholds = np.linspace(0, 1, 50)
    tpr_list = []
    fpr_list = []

    for thresh in thresholds:
        pred_pos = pred_np >= thresh
        tp = np.sum(pred_pos & (gt_np == 1))
        fp = np.sum(pred_pos & (gt_np == 0))
        fn = np.sum(~pred_pos & (gt_np == 1))
        tn = np.sum(~pred_pos & (gt_np == 0))

        tpr = tp / (tp + fn + 1e-8)
        fpr = fp / (fp + tn + 1e-8)
        tpr_list.append(tpr)
        fpr_list.append(fpr)

    # Compute area under ROC curve via trapezoidal rule  # [SELF-IMPLEMENTED]
    sorted_pairs = sorted(zip(fpr_list, tpr_list))
    fpr_sorted = [p[0] for p in sorted_pairs]
    tpr_sorted = [p[1] for p in sorted_pairs]

    auc = 0.0
    for i in range(1, len(fpr_sorted)):
        auc += (fpr_sorted[i] - fpr_sorted[i - 1]) * (tpr_sorted[i] + tpr_sorted[i - 1]) / 2

    return float(np.clip(auc, 0.0, 1.0))


def compute_traffic_accuracy(pred: torch.Tensor, gt: torch.Tensor) -> float:  # [SELF-IMPLEMENTED]
    """
    Classification accuracy for traffic light state prediction.

    Args:
        pred: (B, 4) predicted logits
        gt: (B, 4) ground truth one-hot labels
    Returns:
        Accuracy as float
    """
    pred_classes = pred.argmax(dim=1)
    gt_classes = gt.argmax(dim=1)
    correct = (pred_classes == gt_classes).float().sum()
    return (correct / len(gt_classes)).item()


# =============================================================================
# Training Loop
# =============================================================================

def train_one_epoch(model: nn.Module, dataloader: DataLoader,
                    criterion: InterFuserLoss, optimizer: torch.optim.Optimizer,
                    scaler: GradScaler, device: torch.device,
                    grad_clip: float = 1.0, use_amp: bool = True,
                    amp_device: str = "cuda") -> dict:  # [SELF-IMPLEMENTED]
    """
    Train for one epoch with AMP and gradient clipping.

    Args:
        model: InterFuserWithTraffic model
        dataloader: training data loader
        criterion: InterFuserLoss instance
        optimizer: optimizer
        scaler: GradScaler for AMP
        device: torch device
        grad_clip: maximum gradient norm
        use_amp: whether to use automatic mixed precision
    Returns:
        dict with average losses for the epoch
    """
    model.train()
    total_losses = {"total": 0.0, "waypoint": 0.0, "density": 0.0,
                    "safety": 0.0, "traffic": 0.0}
    num_batches = 0

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for batch in pbar:
        # Move data to device  # [SELF-IMPLEMENTED]
        multi_view = batch["multi_view_images"].to(device)  # (B, 3, 3, H, W)
        lidar = batch["lidar_bev"].to(device)  # (B, 2, H, W)
        gt_waypoints = batch["waypoints"].to(device)
        gt_density = batch["density_map"].to(device)
        gt_safety = batch["safety_score"].to(device)
        gt_traffic = batch["traffic_light"].to(device)

        # Extract individual views  # [SELF-IMPLEMENTED]
        front_img = multi_view[:, 0]  # (B, 3, H, W)
        left_img = multi_view[:, 1]
        right_img = multi_view[:, 2]

        # Forward pass with AMP  # [SELF-IMPLEMENTED]
        optimizer.zero_grad()
        with autocast(device_type=amp_device, enabled=use_amp):
            predictions = model(front_img, left_img, right_img, lidar)

            targets = {
                "waypoints": gt_waypoints,
                "density_map": gt_density,
                "safety_score": gt_safety,
                "traffic_light": gt_traffic,
            }
            losses = criterion(predictions, targets)

        # Backward pass with gradient scaling  # [SELF-IMPLEMENTED]
        if use_amp:
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)  # [FROM PAPER]
            scaler.step(optimizer)
            scaler.update()
        else:
            losses["total"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)  # [FROM PAPER]
            optimizer.step()

        # Accumulate losses  # [SELF-IMPLEMENTED]
        for key in total_losses:
            total_losses[key] += losses[key].item()
        num_batches += 1

        # Update progress bar
        pbar.set_postfix({
            "loss": f"{losses['total'].item():.4f}",
            "wp": f"{losses['waypoint'].item():.4f}",
            "dm": f"{losses['density'].item():.4f}",
        })

    # Average losses  # [SELF-IMPLEMENTED]
    for key in total_losses:
        total_losses[key] /= max(num_batches, 1)

    return total_losses


@torch.no_grad()
def validate(model: nn.Module, dataloader: DataLoader,
             criterion: InterFuserLoss, device: torch.device,
             use_amp: bool = True, amp_device: str = "cuda") -> dict:  # [SELF-IMPLEMENTED]
    """
    Validate model and compute metrics.

    Args:
        model: InterFuserWithTraffic model
        dataloader: validation data loader
        criterion: InterFuserLoss instance
        device: torch device
        use_amp: whether to use automatic mixed precision
    Returns:
        dict with losses and metrics
    """
    model.eval()
    total_losses = {"total": 0.0, "waypoint": 0.0, "density": 0.0,
                    "safety": 0.0, "traffic": 0.0}
    all_wp_errors = []
    all_density_ious = []
    all_safety_preds = []
    all_safety_gts = []
    all_tl_preds = []
    all_tl_gts = []
    num_batches = 0

    pbar = tqdm(dataloader, desc="Validating", leave=False)
    for batch in pbar:
        # Move data to device  # [SELF-IMPLEMENTED]
        multi_view = batch["multi_view_images"].to(device)
        lidar = batch["lidar_bev"].to(device)
        gt_waypoints = batch["waypoints"].to(device)
        gt_density = batch["density_map"].to(device)
        gt_safety = batch["safety_score"].to(device)
        gt_traffic = batch["traffic_light"].to(device)

        front_img = multi_view[:, 0]
        left_img = multi_view[:, 1]
        right_img = multi_view[:, 2]

        # Forward pass  # [SELF-IMPLEMENTED]
        with autocast(device_type=amp_device, enabled=use_amp):
            predictions = model(front_img, left_img, right_img, lidar)

            targets = {
                "waypoints": gt_waypoints,
                "density_map": gt_density,
                "safety_score": gt_safety,
                "traffic_light": gt_traffic,
            }
            losses = criterion(predictions, targets)

        # Accumulate losses  # [SELF-IMPLEMENTED]
        for key in total_losses:
            total_losses[key] += losses[key].item()
        num_batches += 1

        # Compute metrics  # [SELF-IMPLEMENTED]
        all_wp_errors.append(
            compute_waypoint_error(predictions["waypoints"], gt_waypoints))
        all_density_ious.append(
            compute_density_iou(predictions["density_map"], gt_density))
        all_safety_preds.append(predictions["safety_score"].cpu())
        all_safety_gts.append(gt_safety.cpu())
        all_tl_preds.append(predictions["traffic_light"].cpu())
        all_tl_gts.append(gt_traffic.cpu())

    # Average losses  # [SELF-IMPLEMENTED]
    for key in total_losses:
        total_losses[key] /= max(num_batches, 1)

    # Aggregate metrics  # [SELF-IMPLEMENTED]
    metrics = {
        "waypoint_error": np.mean(all_wp_errors) if all_wp_errors else 0.0,
        "density_iou": np.mean(all_density_ious) if all_density_ious else 0.0,
        "safety_auc": compute_safety_auc(
            torch.cat(all_safety_preds, dim=0),
            torch.cat(all_safety_gts, dim=0)
        ) if all_safety_preds else 0.0,
        "traffic_accuracy": compute_traffic_accuracy(
            torch.cat(all_tl_preds, dim=0),
            torch.cat(all_tl_gts, dim=0)
        ) if all_tl_preds else 0.0,
    }

    return {**total_losses, **metrics}


# =============================================================================
# Checkpoint Management
# =============================================================================

def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer,
                    scheduler, scaler: GradScaler, epoch: int,
                    metrics: dict, save_path: str) -> None:  # [SELF-IMPLEMENTED]
    """Save training checkpoint."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "scaler_state_dict": scaler.state_dict(),
        "metrics": metrics,
    }
    torch.save(checkpoint, save_path)
    print(f"  Checkpoint saved: {save_path}")


def load_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer,
                    scheduler, scaler: GradScaler,
                    load_path: str, device: torch.device) -> int:  # [SELF-IMPLEMENTED]
    """Load training checkpoint. Returns the epoch to resume from."""
    if not os.path.exists(load_path):
        print(f"  No checkpoint found at {load_path}, starting from scratch.")
        return 0

    checkpoint = torch.load(load_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler and checkpoint["scheduler_state_dict"]:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    scaler.load_state_dict(checkpoint["scaler_state_dict"])

    epoch = checkpoint["epoch"]
    print(f"  Resumed from checkpoint: epoch {epoch}")
    if "metrics" in checkpoint:
        print(f"  Previous metrics: {checkpoint['metrics']}")
    return epoch + 1


# =============================================================================
# Argument Parser
# =============================================================================

def parse_args() -> argparse.Namespace:  # [SELF-IMPLEMENTED]
    parser = argparse.ArgumentParser(
        description="InterFuser Training Script (CoRL 2022)")

    # Model architecture
    parser.add_argument("--d-model", type=int, default=256,
                        help="Transformer hidden dimension (default: 256)")
    parser.add_argument("--n-heads", type=int, default=8,
                        help="Number of attention heads (default: 8)")
    parser.add_argument("--num-layers", type=int, default=6,
                        help="Number of transformer encoder layers (default: 6)")
    parser.add_argument("--num-waypoints", type=int, default=4,
                        help="Number of future waypoints to predict (default: 4)")
    parser.add_argument("--bev-size", type=int, default=32,
                        help="BEV density map spatial size (default: 32)")
    parser.add_argument("--img-size", type=int, default=224,
                        help="Input image size (default: 224)")

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs (default: 20)")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Training batch size (default: 4)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate (default: 1e-4)")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="Weight decay for AdamW (default: 1e-4)")
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="Gradient clipping max norm (default: 1.0)")
    parser.add_argument("--warmup-epochs", type=int, default=2,
                        help="Number of warmup epochs (default: 2)")

    # Loss weights  # [FROM PAPER]
    parser.add_argument("--w-waypoint", type=float, default=1.0,
                        help="Waypoint loss weight (default: 1.0)")
    parser.add_argument("--w-density", type=float, default=1.0,
                        help="Density map loss weight (default: 1.0)")
    parser.add_argument("--w-safety", type=float, default=0.5,
                        help="Safety score loss weight (default: 0.5)")
    parser.add_argument("--w-traffic", type=float, default=0.5,
                        help="Traffic light loss weight (default: 0.5)")

    # Dataset
    parser.add_argument("--train-samples", type=int, default=200,
                        help="Number of synthetic training samples (default: 200)")
    parser.add_argument("--val-samples", type=int, default=50,
                        help="Number of synthetic validation samples (default: 50)")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader num_workers (default: 0)")

    # Training options
    parser.add_argument("--amp", action="store_true", default=True,
                        help="Use automatic mixed precision (default: True)")
    parser.add_argument("--no-amp", action="store_true",
                        help="Disable automatic mixed precision")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (default: auto-detect)")

    # Checkpoint
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                        help="Directory to save checkpoints (default: checkpoints)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--save-every", type=int, default=5,
                        help="Save checkpoint every N epochs (default: 5)")

    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    # Device setup  # [SELF-IMPLEMENTED]
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    use_amp = args.amp and not args.no_amp and device.type == "cuda"

    print("=" * 70)
    print("InterFuser Training (CoRL 2022)")
    print("Safety-Enhanced Autonomous Driving Using Interpretable Sensor Fusion")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"AMP: {use_amp}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Loss weights: wp={args.w_waypoint}, dm={args.w_density}, "
          f"sf={args.w_safety}, tl={args.w_traffic}")
    print("=" * 70)

    # Create datasets  # [SELF-IMPLEMENTED]
    print("\n[1/5] Creating synthetic datasets...")
    train_dataset = InterFuserDataset(
        num_samples=args.train_samples, num_waypoints=args.num_waypoints,
        bev_size=args.bev_size, img_size=args.img_size, split="train")
    val_dataset = InterFuserDataset(
        num_samples=args.val_samples, num_waypoints=args.num_waypoints,
        bev_size=args.bev_size, img_size=args.img_size, split="val")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        drop_last=True)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    print(f"  Train samples: {len(train_dataset)}, "
          f"Val samples: {len(val_dataset)}")
    print(f"  Train batches: {len(train_loader)}, "
          f"Val batches: {len(val_loader)}")

    # Create model  # [SELF-IMPLEMENTED]
    print("\n[2/5] Building InterFuser model...")
    model = InterFuserWithTraffic(
        d_model=args.d_model, n_heads=args.n_heads,
        num_layers=args.num_layers, num_waypoints=args.num_waypoints,
        bev_size=args.bev_size
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {num_params:,}")
    print(f"  Trainable parameters: {num_trainable:,}")

    # Create loss, optimizer, scheduler  # [SELF-IMPLEMENTED]
    print("\n[3/5] Setting up loss, optimizer, scheduler...")
    criterion = InterFuserLoss(
        w_waypoint=args.w_waypoint, w_density=args.w_density,
        w_safety=args.w_safety, w_traffic=args.w_traffic)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)  # [FROM PAPER]

    scheduler = CosineAnnealingLR(
        optimizer, T_max=args.epochs - args.warmup_epochs, eta_min=1e-6)  # [SELF-IMPLEMENTED]

    scaler = GradScaler(device.type, enabled=use_amp)

    # Resume from checkpoint  # [SELF-IMPLEMENTED]
    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(
            model, optimizer, scheduler, scaler, args.resume, device)

    # Training loop  # [SELF-IMPLEMENTED]
    print("\n[4/5] Starting training...")
    print("-" * 70)

    best_val_loss = float("inf")
    best_metrics = {}

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()

        # Learning rate warmup  # [SELF-IMPLEMENTED]
        if epoch < args.warmup_epochs:
            warmup_factor = (epoch + 1) / args.warmup_epochs
            for param_group in optimizer.param_groups:
                param_group["lr"] = args.lr * warmup_factor

        # Train  # [SELF-IMPLEMENTED]
        train_losses = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler,
            device, grad_clip=args.grad_clip, use_amp=use_amp,
            amp_device=device.type)

        # Validate  # [SELF-IMPLEMENTED]
        val_results = validate(model, val_loader, criterion, device,
                               use_amp=use_amp, amp_device=device.type)

        # Step scheduler after warmup  # [SELF-IMPLEMENTED]
        if epoch >= args.warmup_epochs:
            scheduler.step()

        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        # Print epoch summary
        print(f"\nEpoch [{epoch + 1}/{args.epochs}] "
              f"({epoch_time:.1f}s, lr={current_lr:.2e})")
        print(f"  Train Loss: {train_losses['total']:.4f} "
              f"(wp={train_losses['waypoint']:.4f}, "
              f"dm={train_losses['density']:.4f}, "
              f"sf={train_losses['safety']:.4f}, "
              f"tl={train_losses['traffic']:.4f})")
        print(f"  Val   Loss: {val_results['total']:.4f} "
              f"(wp={val_results['waypoint']:.4f}, "
              f"dm={val_results['density']:.4f}, "
              f"sf={val_results['safety']:.4f}, "
              f"tl={val_results['traffic']:.4f})")
        print(f"  Metrics: WP_err={val_results['waypoint_error']:.4f}, "
              f"DM_IoU={val_results['density_iou']:.4f}, "
              f"Safety_AUC={val_results['safety_auc']:.4f}, "
              f"TL_acc={val_results['traffic_accuracy']:.4f}")

        # Checkpoint management  # [SELF-IMPLEMENTED]
        is_best = val_results["total"] < best_val_loss
        if is_best:
            best_val_loss = val_results["total"]
            best_metrics = {
                "epoch": epoch + 1,
                "val_loss": val_results["total"],
                "waypoint_error": val_results["waypoint_error"],
                "density_iou": val_results["density_iou"],
                "safety_auc": val_results["safety_auc"],
                "traffic_accuracy": val_results["traffic_accuracy"],
            }
            save_checkpoint(
                model, optimizer, scheduler, scaler, epoch, best_metrics,
                os.path.join(args.checkpoint_dir, "best_model.pth"))

        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(
                model, optimizer, scheduler, scaler, epoch, val_results,
                os.path.join(args.checkpoint_dir, f"epoch_{epoch + 1}.pth"))

    # Final summary  # [SELF-IMPLEMENTED]
    print("\n" + "=" * 70)
    print("[5/5] Training Complete!")
    print("=" * 70)
    print(f"Best validation loss: {best_val_loss:.4f}")
    if best_metrics:
        print(f"  Best epoch: {best_metrics['epoch']}")
        print(f"  Waypoint error: {best_metrics['waypoint_error']:.4f}")
        print(f"  Density IoU: {best_metrics['density_iou']:.4f}")
        print(f"  Safety AUC: {best_metrics['safety_auc']:.4f}")
        print(f"  Traffic accuracy: {best_metrics['traffic_accuracy']:.4f}")
    print(f"\nCheckpoints saved to: {os.path.abspath(args.checkpoint_dir)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
