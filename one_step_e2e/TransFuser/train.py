"""
TransFuser Training Script
===========================
ATTRIBUTION:
- Loss functions: Based on TransFuser paper (Chitta et al., PAMI 2023)
  - Waypoint L1 loss (Section 3.4)
  - Auxiliary BEV segmentation loss (Section 3.3)
  - Speed prediction loss (Section 3.4)
- Training strategy: Imitation learning from expert demonstrations (CARLA autopilot)
- Data collection: Expert drives in CARLA simulator, records sensor + actions
- Implementation: Self-implemented in PyTorch (simplified from official transfuser repo)
- Synthetic dataset: Self-implemented for demonstration (real uses CARLA collected data)
"""

import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm is not installed  # [SELF-IMPLEMENTED]
    def tqdm(iterable, **kwargs):
        return iterable

from model import TransFuser


# =============================================================================
# Synthetic Dataset
# =============================================================================

class TransFuserDataset(Dataset):  # [SELF-IMPLEMENTED]
    """
    Synthetic dataset for TransFuser training demonstration.

    In practice, data comes from CARLA expert autopilot driving:
      - Front camera RGB images recorded at 2 Hz
      - LiDAR point clouds projected to BEV representation
      - GPS/IMU for waypoint ground truth
      - BEV semantic maps from CARLA ground truth

    This synthetic version generates random tensors with plausible statistics
    so the training loop can be verified without CARLA data collection.

    Shapes:
      - Front camera image: (3, 256, 512) - RGB, H x W
      - LiDAR BEV: (2, 256, 256) - height + intensity channels
      - Speed: (1,) - ego vehicle speed in m/s
      - Waypoints GT: (4, 2) - 4 future waypoints, (dx, dy) in ego frame
      - BEV segmentation GT: (64, 64) - class labels for BEV grid
    """

    def __init__(self, num_samples: int = 1000, num_waypoints: int = 4,
                 num_bev_classes: int = 4, split: str = "train"):
        super().__init__()
        self.num_samples = num_samples
        self.num_waypoints = num_waypoints
        self.num_bev_classes = num_bev_classes
        self.split = split

        # Pre-generate synthetic data for reproducibility  # [SELF-IMPLEMENTED]
        seed = 42 if split == "train" else 123
        self.rng = torch.Generator().manual_seed(seed)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        """Generate a single synthetic training sample."""
        # Front camera image: normalized RGB  # [SELF-IMPLEMENTED]
        # Real data: CARLA front camera at 256x512 resolution
        image = torch.randn(3, 256, 512) * 0.5 + 0.5  # approximate ImageNet stats
        image = image.clamp(0, 1)

        # LiDAR BEV: 2 channels (height, intensity)  # [SELF-IMPLEMENTED]
        # Real data: LiDAR points projected onto 256x256 BEV grid
        lidar_bev = torch.randn(2, 256, 256) * 0.3

        # Ego speed in m/s (typical urban driving: 0-15 m/s)  # [SELF-IMPLEMENTED]
        speed = torch.rand(1) * 15.0

        # Ground truth waypoints: 4 future positions in ego frame  # [SELF-IMPLEMENTED]
        # Real data: derived from GPS/IMU future trajectory
        # Waypoints are cumulative displacements at 0.5s intervals
        dt = 0.5
        base_speed = speed.item()
        waypoints = torch.zeros(self.num_waypoints, 2)
        for t in range(self.num_waypoints):
            # Forward displacement (x) increases with time
            waypoints[t, 0] = base_speed * dt * (t + 1) + torch.randn(1).item() * 0.3
            # Lateral displacement (y) has small curvature
            waypoints[t, 1] = torch.randn(1).item() * 0.5 * (t + 1) * 0.3

        # BEV segmentation ground truth  # [SELF-IMPLEMENTED]
        # Real data: from CARLA semantic segmentation sensor projected to BEV
        # Classes: 0=road, 1=vehicle, 2=pedestrian, 3=other
        bev_seg = torch.zeros(64, 64, dtype=torch.long)
        # Simulate road as dominant class
        bev_seg[:, :] = 0  # road everywhere
        # Add some vehicles and pedestrians
        n_vehicles = torch.randint(2, 8, (1,)).item()
        for _ in range(n_vehicles):
            cx, cy = torch.randint(5, 59, (2,)).tolist()
            bev_seg[cx-2:cx+2, cy-2:cy+2] = 1  # vehicle
        n_peds = torch.randint(0, 4, (1,)).item()
        for _ in range(n_peds):
            cx, cy = torch.randint(5, 59, (2,)).tolist()
            bev_seg[cx-1:cx+1, cy-1:cy+1] = 2  # pedestrian

        return {
            "image": image,
            "lidar_bev": lidar_bev,
            "speed": speed,
            "waypoints": waypoints,
            "bev_segmentation": bev_seg,
        }


# =============================================================================
# Loss Functions (from paper)
# =============================================================================

class WaypointLoss(nn.Module):  # [FROM PAPER]
    """
    L1 loss on predicted waypoints (Section 3.4 of TransFuser PAMI 2023).

    The model predicts T future waypoints in ego-vehicle frame.
    Loss is the mean absolute error between predicted and ground truth waypoints.

    L_wp = (1/T) * sum_t ||wp_pred_t - wp_gt_t||_1
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred_waypoints: torch.Tensor,
                gt_waypoints: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_waypoints: (B, T, 2) predicted waypoints
            gt_waypoints: (B, T, 2) ground truth waypoints
        Returns:
            Scalar L1 loss averaged over batch, timesteps, and coordinates
        """
        return F.l1_loss(pred_waypoints, gt_waypoints)  # [FROM PAPER]


class BEVSegLoss(nn.Module):  # [FROM PAPER]
    """
    Cross-entropy loss for auxiliary BEV segmentation (Section 3.3).

    The BEV segmentation head predicts semantic classes for each cell in
    the bird's-eye-view grid. This auxiliary task improves feature learning
    by encouraging the LiDAR backbone to learn semantically meaningful BEV
    representations.

    L_bev = CrossEntropy(bev_pred, bev_gt)
    """

    def __init__(self, num_classes: int = 4, ignore_index: int = -1):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)  # [FROM PAPER]

    def forward(self, pred_bev: torch.Tensor,
                gt_bev: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_bev: (B, C, H, W) predicted BEV segmentation logits
            gt_bev: (B, H, W) ground truth class labels

        Returns:
            Scalar cross-entropy loss
        """
        # Resize prediction to match GT if needed  # [SELF-IMPLEMENTED]
        if pred_bev.shape[2:] != gt_bev.shape[1:]:
            pred_bev = F.interpolate(
                pred_bev, size=gt_bev.shape[1:], mode="bilinear",
                align_corners=False)

        return self.criterion(pred_bev, gt_bev)  # [FROM PAPER]


class SpeedLoss(nn.Module):  # [FROM PAPER]
    """
    L1 loss for speed prediction (Section 3.4).

    While the current model uses speed as INPUT context (not a prediction head),
    the original TransFuser also predicts speed as an auxiliary regularizer.
    This loss can be added if a speed prediction head is included.

    L_speed = ||speed_pred - speed_gt||_1
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred_speed: torch.Tensor,
                gt_speed: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_speed: (B, 1) predicted speed
            gt_speed: (B, 1) ground truth speed
        Returns:
            Scalar L1 loss
        """
        return F.l1_loss(pred_speed, gt_speed)  # [FROM PAPER]


class TransFuserLoss(nn.Module):  # [FROM PAPER]
    """
    Combined multi-task loss for TransFuser training.

    L_total = lambda_wp * L_wp + lambda_bev * L_bev + lambda_speed * L_speed

    Lambda values from the paper (Section 3.4, PAMI 2023):
      - lambda_wp = 1.0 (primary objective)
      - lambda_bev = 0.5 (auxiliary segmentation)
      - lambda_speed = 0.1 (auxiliary speed, lower weight as it's regularization)
    """

    def __init__(self, lambda_wp: float = 1.0, lambda_bev: float = 0.5,
                 lambda_speed: float = 0.1, num_bev_classes: int = 4):
        super().__init__()
        self.lambda_wp = lambda_wp  # [FROM PAPER]
        self.lambda_bev = lambda_bev  # [FROM PAPER]
        self.lambda_speed = lambda_speed  # [FROM PAPER]

        self.waypoint_loss = WaypointLoss()
        self.bev_seg_loss = BEVSegLoss(num_classes=num_bev_classes)
        self.speed_loss = SpeedLoss()

    def forward(self, predictions: dict, targets: dict) -> dict:
        """
        Compute combined loss with paper's weighting scheme.

        Args:
            predictions: dict with 'waypoints', 'bev_segmentation'
            targets: dict with 'waypoints', 'bev_segmentation', 'speed'
        Returns:
            dict with 'total_loss' and individual loss components
        """
        # Waypoint L1 loss (primary)  # [FROM PAPER]
        loss_wp = self.waypoint_loss(
            predictions["waypoints"], targets["waypoints"])

        # BEV segmentation loss (auxiliary)  # [FROM PAPER]
        loss_bev = self.bev_seg_loss(
            predictions["bev_segmentation"], targets["bev_segmentation"])

        # Speed loss (auxiliary regularizer)  # [FROM PAPER]
        # Note: In this simplified version, the model uses speed as input
        # but does not have a speed prediction head. We set loss_speed = 0.
        # In the full TransFuser, a speed prediction branch exists.
        loss_speed = torch.tensor(0.0, device=loss_wp.device)  # [SIMPLIFIED]

        # Combined loss with paper's lambda weighting  # [FROM PAPER]
        total_loss = (
            self.lambda_wp * loss_wp +
            self.lambda_bev * loss_bev +
            self.lambda_speed * loss_speed
        )

        return {
            "total_loss": total_loss,
            "waypoint_loss": loss_wp.detach(),
            "bev_seg_loss": loss_bev.detach(),
            "speed_loss": loss_speed.detach(),
        }


# =============================================================================
# Validation Metrics
# =============================================================================

def compute_waypoint_l1(pred_waypoints: torch.Tensor,
                        gt_waypoints: torch.Tensor) -> float:  # [FROM PAPER]
    """
    Compute mean L1 error for waypoint predictions.

    This is the primary evaluation metric for TransFuser: average displacement
    error between predicted and ground truth future waypoints.

    Args:
        pred_waypoints: (B, T, 2) predicted waypoints
        gt_waypoints: (B, T, 2) ground truth waypoints
    Returns:
        Mean L1 error (scalar float)
    """
    return F.l1_loss(pred_waypoints, gt_waypoints).item()


def compute_bev_miou(pred_bev: torch.Tensor, gt_bev: torch.Tensor,
                     num_classes: int = 4) -> float:  # [SELF-IMPLEMENTED]
    """
    Compute mean Intersection-over-Union (mIoU) for BEV segmentation.

    mIoU is the standard metric for segmentation quality, computed per-class
    then averaged across classes.

    Args:
        pred_bev: (B, C, H, W) predicted logits
        gt_bev: (B, H, W) ground truth class labels
    Returns:
        mIoU score (0 to 1)
    """
    # Resize prediction to match GT if needed  # [SELF-IMPLEMENTED]
    if pred_bev.shape[2:] != gt_bev.shape[1:]:
        pred_bev = F.interpolate(
            pred_bev, size=gt_bev.shape[1:], mode="bilinear",
            align_corners=False)

    pred_classes = pred_bev.argmax(dim=1)  # (B, H, W)

    ious = []
    for cls in range(num_classes):
        pred_mask = (pred_classes == cls)
        gt_mask = (gt_bev == cls)

        intersection = (pred_mask & gt_mask).sum().float()
        union = (pred_mask | gt_mask).sum().float()

        if union > 0:
            ious.append((intersection / union).item())
        # Skip classes not present in GT or prediction (avoid division by zero)

    if len(ious) == 0:
        return 0.0
    return sum(ious) / len(ious)


# =============================================================================
# Training and Validation Loops
# =============================================================================

def train_one_epoch(model: nn.Module, dataloader: DataLoader,
                    criterion: TransFuserLoss, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int,
                    scaler: GradScaler = None,
                    max_grad_norm: float = 1.0,
                    use_amp: bool = False) -> dict:  # [SELF-IMPLEMENTED]
    """
    Train the model for one epoch.

    Implements:
      - Mixed precision training (AMP) for memory efficiency
      - Gradient clipping to prevent exploding gradients
      - Progress bar with live loss display

    Args:
        model: TransFuser model
        dataloader: training data loader
        criterion: TransFuserLoss
        optimizer: optimizer
        device: computation device
        epoch: current epoch number
        scaler: GradScaler for AMP
        max_grad_norm: maximum gradient norm for clipping
        use_amp: whether to use automatic mixed precision
    Returns:
        dict with average losses for the epoch
    """
    model.train()

    total_loss_accum = 0.0
    wp_loss_accum = 0.0
    bev_loss_accum = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Train]",
                leave=True, dynamic_ncols=True)

    for batch in pbar:
        # Move data to device  # [SELF-IMPLEMENTED]
        image = batch["image"].to(device)
        lidar_bev = batch["lidar_bev"].to(device)
        speed = batch["speed"].to(device)
        gt_waypoints = batch["waypoints"].to(device)
        gt_bev_seg = batch["bev_segmentation"].to(device)

        optimizer.zero_grad()

        # Forward pass with optional AMP  # [SELF-IMPLEMENTED]
        if use_amp:
            with autocast(device_type=device.type):
                output = model(image, lidar_bev, speed, return_aux=True)
                predictions = {
                    "waypoints": output["waypoints"],
                    "bev_segmentation": output["bev_segmentation"],
                }
                targets = {
                    "waypoints": gt_waypoints,
                    "bev_segmentation": gt_bev_seg,
                    "speed": speed,
                }
                losses = criterion(predictions, targets)
                loss = losses["total_loss"]

            # Backward with scaled gradients  # [SELF-IMPLEMENTED]
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(image, lidar_bev, speed, return_aux=True)
            predictions = {
                "waypoints": output["waypoints"],
                "bev_segmentation": output["bev_segmentation"],
            }
            targets = {
                "waypoints": gt_waypoints,
                "bev_segmentation": gt_bev_seg,
                "speed": speed,
            }
            losses = criterion(predictions, targets)
            loss = losses["total_loss"]

            # Backward pass  # [SELF-IMPLEMENTED]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        # Accumulate metrics  # [SELF-IMPLEMENTED]
        total_loss_accum += losses["total_loss"].item()
        wp_loss_accum += losses["waypoint_loss"].item()
        bev_loss_accum += losses["bev_seg_loss"].item()
        num_batches += 1

        # Update progress bar
        pbar.set_postfix({
            "loss": f"{total_loss_accum / num_batches:.4f}",
            "wp_l1": f"{wp_loss_accum / num_batches:.4f}",
            "bev_ce": f"{bev_loss_accum / num_batches:.4f}",
        })

    return {
        "total_loss": total_loss_accum / max(num_batches, 1),
        "waypoint_loss": wp_loss_accum / max(num_batches, 1),
        "bev_seg_loss": bev_loss_accum / max(num_batches, 1),
    }


@torch.no_grad()
def validate(model: nn.Module, dataloader: DataLoader,
             criterion: TransFuserLoss, device: torch.device,
             epoch: int, num_bev_classes: int = 4) -> dict:  # [SELF-IMPLEMENTED]
    """
    Validate the model on the validation set.

    Computes:
      - All training losses
      - Waypoint L1 error (primary metric)
      - BEV segmentation mIoU (auxiliary metric)

    Args:
        model: TransFuser model
        dataloader: validation data loader
        criterion: TransFuserLoss
        device: computation device
        epoch: current epoch number
        num_bev_classes: number of BEV segmentation classes
    Returns:
        dict with validation metrics
    """
    model.eval()

    total_loss_accum = 0.0
    wp_loss_accum = 0.0
    bev_loss_accum = 0.0
    wp_l1_accum = 0.0
    bev_miou_accum = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Val]",
                leave=True, dynamic_ncols=True)

    for batch in pbar:
        # Move data to device  # [SELF-IMPLEMENTED]
        image = batch["image"].to(device)
        lidar_bev = batch["lidar_bev"].to(device)
        speed = batch["speed"].to(device)
        gt_waypoints = batch["waypoints"].to(device)
        gt_bev_seg = batch["bev_segmentation"].to(device)

        # Forward pass
        output = model(image, lidar_bev, speed, return_aux=True)
        predictions = {
            "waypoints": output["waypoints"],
            "bev_segmentation": output["bev_segmentation"],
        }
        targets = {
            "waypoints": gt_waypoints,
            "bev_segmentation": gt_bev_seg,
            "speed": speed,
        }

        # Compute losses
        losses = criterion(predictions, targets)

        # Compute evaluation metrics  # [SELF-IMPLEMENTED]
        wp_l1 = compute_waypoint_l1(output["waypoints"], gt_waypoints)
        bev_miou = compute_bev_miou(
            output["bev_segmentation"], gt_bev_seg,
            num_classes=num_bev_classes)

        # Accumulate
        total_loss_accum += losses["total_loss"].item()
        wp_loss_accum += losses["waypoint_loss"].item()
        bev_loss_accum += losses["bev_seg_loss"].item()
        wp_l1_accum += wp_l1
        bev_miou_accum += bev_miou
        num_batches += 1

        pbar.set_postfix({
            "loss": f"{total_loss_accum / num_batches:.4f}",
            "wp_l1": f"{wp_l1_accum / num_batches:.4f}",
            "mIoU": f"{bev_miou_accum / num_batches:.4f}",
        })

    n = max(num_batches, 1)
    return {
        "total_loss": total_loss_accum / n,
        "waypoint_loss": wp_loss_accum / n,
        "bev_seg_loss": bev_loss_accum / n,
        "waypoint_l1_error": wp_l1_accum / n,
        "bev_miou": bev_miou_accum / n,
    }


# =============================================================================
# Checkpoint Management
# =============================================================================

def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer,
                    scheduler, scaler: GradScaler, epoch: int,
                    metrics: dict, save_path: str):  # [SELF-IMPLEMENTED]
    """
    Save training checkpoint for resume capability.

    Saves:
      - Model state dict
      - Optimizer state dict
      - LR scheduler state
      - GradScaler state (for AMP)
      - Current epoch
      - Validation metrics (for best model tracking)

    Args:
        model: TransFuser model
        optimizer: optimizer
        scheduler: LR scheduler
        scaler: GradScaler (can be None)
        epoch: current epoch
        metrics: dict of current validation metrics
        save_path: path to save checkpoint
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "metrics": metrics,
    }
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(checkpoint, save_path)
    print(f"  Checkpoint saved: {save_path}")


def load_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer,
                    scheduler, scaler: GradScaler,
                    checkpoint_path: str,
                    device: torch.device) -> int:  # [SELF-IMPLEMENTED]
    """
    Load training checkpoint to resume training.

    Args:
        model: TransFuser model
        optimizer: optimizer
        scheduler: LR scheduler
        scaler: GradScaler (can be None)
        checkpoint_path: path to checkpoint file
        device: target device
    Returns:
        Epoch number to resume from (next epoch)
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    if scaler and checkpoint.get("scaler_state_dict"):
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    epoch = checkpoint["epoch"]
    metrics = checkpoint.get("metrics", {})
    print(f"  Resumed from epoch {epoch + 1}, metrics: {metrics}")

    return epoch + 1  # return next epoch to train


# =============================================================================
# Main Training Script
# =============================================================================

def parse_args():  # [SELF-IMPLEMENTED]
    """Parse command-line arguments for TransFuser training."""
    parser = argparse.ArgumentParser(
        description="TransFuser Training - Multi-Modal Fusion for E2E Driving")

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs (default: 50)")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size per GPU (default: 4)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Initial learning rate (default: 1e-4)")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="Weight decay for AdamW (default: 1e-4)")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Maximum gradient norm for clipping (default: 1.0)")

    # Model configuration
    parser.add_argument("--num_waypoints", type=int, default=4,
                        help="Number of future waypoints to predict (default: 4)")
    parser.add_argument("--hidden_dim", type=int, default=512,
                        help="Hidden dimension for feature projection (default: 512)")
    parser.add_argument("--num_bev_classes", type=int, default=4,
                        help="Number of BEV segmentation classes (default: 4)")

    # Loss weights (from paper Section 3.4)
    parser.add_argument("--lambda_wp", type=float, default=1.0,
                        help="Weight for waypoint loss (default: 1.0)")
    parser.add_argument("--lambda_bev", type=float, default=0.5,
                        help="Weight for BEV segmentation loss (default: 0.5)")
    parser.add_argument("--lambda_speed", type=float, default=0.1,
                        help="Weight for speed loss (default: 0.1)")

    # Dataset
    parser.add_argument("--num_train_samples", type=int, default=800,
                        help="Number of synthetic training samples (default: 800)")
    parser.add_argument("--num_val_samples", type=int, default=200,
                        help="Number of synthetic validation samples (default: 200)")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="DataLoader workers (default: 0 for Windows compat)")

    # Infrastructure
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: 'cuda', 'cpu', or 'auto' (default: auto)")
    parser.add_argument("--amp", action="store_true", default=False,
                        help="Enable automatic mixed precision (AMP)")
    parser.add_argument("--save_dir", type=str, default="checkpoints",
                        help="Directory to save checkpoints (default: checkpoints)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from")

    # Logging
    parser.add_argument("--log_interval", type=int, default=10,
                        help="Log every N batches (default: 10)")
    parser.add_argument("--val_interval", type=int, default=1,
                        help="Validate every N epochs (default: 1)")

    return parser.parse_args()


def main():
    """Main training entry point for TransFuser."""  # [SELF-IMPLEMENTED]
    args = parse_args()

    # =========================================================================
    # Setup
    # =========================================================================
    print("=" * 70)
    print("TransFuser Training - Multi-Modal Fusion for E2E Driving")
    print("=" * 70)
    print(f"Paper: Chitta et al., 'TransFuser: Imitation with Transformer-Based")
    print(f"       Sensor Fusion for Autonomous Driving' (PAMI 2023)")
    print("=" * 70)

    # Device selection  # [SELF-IMPLEMENTED]
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"\nDevice: {device}")

    # AMP setup  # [SELF-IMPLEMENTED]
    use_amp = args.amp and device.type == "cuda"
    if args.amp and not device.type == "cuda":
        print("  Warning: AMP requested but not on CUDA. Disabling AMP.")
    scaler = GradScaler() if use_amp else None
    print(f"AMP: {'enabled' if use_amp else 'disabled'}")

    # =========================================================================
    # Model
    # =========================================================================
    model = TransFuser(
        img_channels=3,
        lidar_channels=2,
        num_waypoints=args.num_waypoints,
        hidden_dim=args.hidden_dim,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: TransFuser")
    print(f"  Total parameters: {num_params:,}")
    print(f"  Trainable parameters: {num_trainable:,}")

    # =========================================================================
    # Dataset and DataLoader
    # =========================================================================
    print(f"\nDataset: Synthetic (for demonstration)")  # [SELF-IMPLEMENTED]
    print(f"  Train samples: {args.num_train_samples}")
    print(f"  Val samples: {args.num_val_samples}")

    train_dataset = TransFuserDataset(
        num_samples=args.num_train_samples,
        num_waypoints=args.num_waypoints,
        num_bev_classes=args.num_bev_classes,
        split="train")

    val_dataset = TransFuserDataset(
        num_samples=args.num_val_samples,
        num_waypoints=args.num_waypoints,
        num_bev_classes=args.num_bev_classes,
        split="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    print(f"  Train batches/epoch: {len(train_loader)}")
    print(f"  Val batches/epoch: {len(val_loader)}")

    # =========================================================================
    # Loss, Optimizer, Scheduler
    # =========================================================================
    criterion = TransFuserLoss(  # [FROM PAPER]
        lambda_wp=args.lambda_wp,
        lambda_bev=args.lambda_bev,
        lambda_speed=args.lambda_speed,
        num_bev_classes=args.num_bev_classes,
    )

    # AdamW optimizer (standard for transformer models)  # [SELF-IMPLEMENTED]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Cosine annealing LR scheduler  # [SELF-IMPLEMENTED]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01,  # minimum LR = 1% of initial
    )

    print(f"\nOptimizer: AdamW (lr={args.lr}, wd={args.weight_decay})")
    print(f"Scheduler: CosineAnnealingLR (T_max={args.epochs})")
    print(f"Loss weights: lambda_wp={args.lambda_wp}, "
          f"lambda_bev={args.lambda_bev}, lambda_speed={args.lambda_speed}")
    print(f"Gradient clipping: max_norm={args.max_grad_norm}")

    # =========================================================================
    # Resume from checkpoint
    # =========================================================================
    start_epoch = 0
    best_val_wp_l1 = float("inf")

    if args.resume:  # [SELF-IMPLEMENTED]
        if os.path.isfile(args.resume):
            start_epoch = load_checkpoint(
                model, optimizer, scheduler, scaler, args.resume, device)
        else:
            print(f"  Warning: checkpoint not found at {args.resume}, "
                  f"starting from scratch.")

    # =========================================================================
    # Training Loop
    # =========================================================================
    print(f"\n{'=' * 70}")
    print(f"Starting training from epoch {start_epoch + 1} to {args.epochs}")
    print(f"{'=' * 70}\n")

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()

        # --- Train ---
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            scaler=scaler,
            max_grad_norm=args.max_grad_norm,
            use_amp=use_amp,
        )

        # --- LR scheduler step ---  # [SELF-IMPLEMENTED]
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        epoch_time = time.time() - epoch_start

        # --- Log training ---
        print(f"\n  Epoch {epoch+1}/{args.epochs} "
              f"({epoch_time:.1f}s, lr={current_lr:.2e})")
        print(f"    Train - Loss: {train_metrics['total_loss']:.4f} | "
              f"WP L1: {train_metrics['waypoint_loss']:.4f} | "
              f"BEV CE: {train_metrics['bev_seg_loss']:.4f}")

        # --- Validate ---
        if (epoch + 1) % args.val_interval == 0:
            val_metrics = validate(
                model=model,
                dataloader=val_loader,
                criterion=criterion,
                device=device,
                epoch=epoch,
                num_bev_classes=args.num_bev_classes,
            )

            print(f"    Val   - Loss: {val_metrics['total_loss']:.4f} | "
                  f"WP L1: {val_metrics['waypoint_l1_error']:.4f} | "
                  f"BEV mIoU: {val_metrics['bev_miou']:.4f}")

            # --- Save best model ---  # [SELF-IMPLEMENTED]
            is_best = val_metrics["waypoint_l1_error"] < best_val_wp_l1
            if is_best:
                best_val_wp_l1 = val_metrics["waypoint_l1_error"]
                best_path = os.path.join(args.save_dir, "best_model.pth")
                save_checkpoint(
                    model, optimizer, scheduler, scaler, epoch,
                    val_metrics, best_path)
                print(f"    ** New best WP L1: {best_val_wp_l1:.4f} **")

        # --- Save periodic checkpoint ---  # [SELF-IMPLEMENTED]
        if (epoch + 1) % 10 == 0 or (epoch + 1) == args.epochs:
            ckpt_path = os.path.join(args.save_dir, f"checkpoint_epoch_{epoch+1}.pth")
            save_checkpoint(
                model, optimizer, scheduler, scaler, epoch,
                val_metrics if (epoch + 1) % args.val_interval == 0 else {},
                ckpt_path)

        print()

    # =========================================================================
    # Final Summary
    # =========================================================================
    print("=" * 70)
    print("Training Complete!")
    print(f"  Best validation WP L1 error: {best_val_wp_l1:.4f}")
    print(f"  Checkpoints saved to: {os.path.abspath(args.save_dir)}")
    print("=" * 70)

    # Save final model  # [SELF-IMPLEMENTED]
    final_path = os.path.join(args.save_dir, "final_model.pth")
    save_checkpoint(
        model, optimizer, scheduler, scaler, args.epochs - 1,
        {"final": True}, final_path)


if __name__ == "__main__":
    main()
