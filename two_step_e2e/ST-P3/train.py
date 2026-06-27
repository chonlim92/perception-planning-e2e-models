"""
ST-P3 Training Script
=====================
ATTRIBUTION:
- Loss functions: Based on ST-P3 paper (Hu et al., ECCV 2022)
  - BEV segmentation: cross-entropy loss (Section 3.3)
  - Future occupancy prediction: binary cross-entropy (Section 3.4)
  - Planning: L1 + collision loss (Section 3.5)
  - Multi-task weighting: equal weights as in paper
- Training strategy: Joint multi-task training from paper
- Implementation: Self-implemented in PyTorch (simplified from original)
- Synthetic dataset: Self-implemented for demonstration (real uses nuScenes)
"""

import os
import argparse
import time
from typing import Dict, Tuple

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

from model import STP3


# =============================================================================
# Synthetic Dataset
# =============================================================================

class STP3Dataset(Dataset):  # [SELF-IMPLEMENTED]
    """
    Synthetic dataset for ST-P3 training demonstration.

    In the real implementation, this would load nuScenes data with:
    - Multi-view camera images (6 cameras)
    - BEV segmentation labels (from map + 3D annotations)
    - Future occupancy grids (from future lidar sweeps)
    - Expert trajectory (from ego vehicle motion)

    Here we generate random tensors to demonstrate the training pipeline.
    """

    def __init__(self, num_samples: int = 100, temporal_frames: int = 4,
                 num_cameras: int = 6, img_h: int = 224, img_w: int = 400,
                 bev_h: int = 200, bev_w: int = 200,
                 num_seg_classes: int = 4, num_future_steps: int = 5,
                 num_waypoints: int = 6):
        super().__init__()
        self.num_samples = num_samples
        self.temporal_frames = temporal_frames
        self.num_cameras = num_cameras
        self.img_h = img_h
        self.img_w = img_w
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.num_seg_classes = num_seg_classes
        self.num_future_steps = num_future_steps
        self.num_waypoints = num_waypoints

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Multi-view images for T timesteps: (T, num_cameras, 3, H, W)  # [SELF-IMPLEMENTED]
        images = torch.randn(
            self.temporal_frames, self.num_cameras, 3, self.img_h, self.img_w
        )

        # BEV segmentation ground truth: (num_classes, bev_h, bev_w)  # [SELF-IMPLEMENTED]
        # Simulates class labels (road, vehicle, lane, background)
        seg_gt = torch.randint(
            0, self.num_seg_classes, (self.bev_h, self.bev_w)
        ).long()

        # Future occupancy ground truth: (num_future_steps, bev_h, bev_w)  # [SELF-IMPLEMENTED]
        # Binary occupancy grids for future timesteps
        occ_gt = torch.zeros(self.num_future_steps, self.bev_h, self.bev_w)
        # Create some random occupied regions to simulate vehicles
        for t in range(self.num_future_steps):
            num_objects = torch.randint(3, 10, (1,)).item()
            for _ in range(num_objects):
                cx = torch.randint(10, self.bev_w - 10, (1,)).item()
                cy = torch.randint(10, self.bev_h - 10, (1,)).item()
                w = torch.randint(3, 8, (1,)).item()
                h = torch.randint(3, 8, (1,)).item()
                occ_gt[t, max(0, cy-h):cy+h, max(0, cx-w):cx+w] = 1.0

        # Trajectory ground truth: (num_waypoints, 2)  # [SELF-IMPLEMENTED]
        # Simulates smooth forward driving trajectory (x, y) in BEV coords
        trajectory_gt = torch.zeros(self.num_waypoints, 2)
        for i in range(self.num_waypoints):
            trajectory_gt[i, 0] = (i + 1) * 2.0 + torch.randn(1).item() * 0.3  # forward
            trajectory_gt[i, 1] = torch.randn(1).item() * 0.5  # lateral

        return {
            'images': images,                # (T, 6, 3, H, W)
            'seg_gt': seg_gt,                # (bev_h, bev_w)
            'occ_gt': occ_gt,                # (5, bev_h, bev_w)
            'trajectory_gt': trajectory_gt,  # (6, 2)
        }


# =============================================================================
# Loss Functions
# =============================================================================

class BEVSegmentationLoss(nn.Module):  # [FROM PAPER]
    """
    Cross-entropy loss for BEV semantic segmentation.

    From ST-P3 Section 3.3: The perception module outputs per-cell class
    probabilities in the BEV grid, supervised with standard cross-entropy.
    """

    def __init__(self, num_classes: int = 4, class_weights: torch.Tensor = None):
        super().__init__()
        self.num_classes = num_classes
        # Class weights for handling imbalanced classes  # [SELF-IMPLEMENTED]
        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, num_classes, H, W) logits from segmentation head
            target: (B, H, W) class indices
        Returns:
            loss: scalar cross-entropy loss
        """
        # Standard cross-entropy as described in paper  # [FROM PAPER]
        loss = F.cross_entropy(pred, target, weight=self.class_weights)
        return loss


class OccupancyLoss(nn.Module):  # [FROM PAPER]
    """
    Binary cross-entropy loss for future occupancy prediction.

    From ST-P3 Section 3.4: The prediction module forecasts future occupancy
    grids, supervised with binary cross-entropy per cell.
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, T_future, H, W) raw logits for occupancy
            target: (B, T_future, H, W) binary occupancy ground truth
        Returns:
            loss: scalar binary cross-entropy loss
        """
        # Binary cross-entropy with logits for numerical stability  # [FROM PAPER]
        loss = F.binary_cross_entropy_with_logits(pred, target)
        return loss


class PlanningLoss(nn.Module):  # [FROM PAPER]
    """
    Planning loss: L1 regression + collision avoidance.

    From ST-P3 Section 3.5:
    - L1 loss between predicted and ground truth waypoints
    - Collision loss: penalizes trajectories that overlap with predicted
      future occupancy (occupied cells)
    """

    def __init__(self, l1_weight: float = 1.0, collision_weight: float = 1.0):
        super().__init__()
        self.l1_weight = l1_weight  # [FROM PAPER]
        self.collision_weight = collision_weight  # [FROM PAPER]

    def forward(self, pred_trajectory: torch.Tensor,
                gt_trajectory: torch.Tensor,
                occupancy_pred: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            pred_trajectory: (B, T_plan, 2) predicted waypoints
            gt_trajectory: (B, T_plan, 2) ground truth waypoints
            occupancy_pred: (B, T_future, H, W) predicted future occupancy (optional)
        Returns:
            dict with 'l1_loss', 'collision_loss', 'total'
        """
        # L1 regression loss for trajectory  # [FROM PAPER]
        l1_loss = F.l1_loss(pred_trajectory, gt_trajectory)

        # Collision loss  # [FROM PAPER]
        # Penalizes planned waypoints that fall in occupied regions
        if occupancy_pred is not None:
            collision_loss = self._compute_collision_loss(
                pred_trajectory, occupancy_pred
            )
        else:
            collision_loss = torch.tensor(0.0, device=pred_trajectory.device)

        total = self.l1_weight * l1_loss + self.collision_weight * collision_loss

        return {
            'l1_loss': l1_loss,
            'collision_loss': collision_loss,
            'total': total,
        }

    def _compute_collision_loss(self, trajectory: torch.Tensor,
                                occupancy: torch.Tensor) -> torch.Tensor:
        """
        Compute collision loss by checking if waypoints fall in occupied cells.

        This is a simplified version - the real implementation uses bilinear
        sampling from the occupancy grid at waypoint locations.
        """  # [SIMPLIFIED]
        B, T_plan, _ = trajectory.shape
        _, T_occ, H, W = occupancy.shape

        # Normalize trajectory coordinates to [-1, 1] for grid sampling  # [SIMPLIFIED]
        # Assume trajectory is in meters, map to BEV grid
        norm_traj = trajectory.clone()
        norm_traj[..., 0] = norm_traj[..., 0] / (H / 2.0)  # x -> [-1, 1]
        norm_traj[..., 1] = norm_traj[..., 1] / (W / 2.0)  # y -> [-1, 1]
        norm_traj = norm_traj.clamp(-1, 1)

        # Use minimum of T_plan and T_occ timesteps
        T = min(T_plan, T_occ)

        collision_loss = torch.tensor(0.0, device=trajectory.device)
        for t in range(T):
            # Get occupancy probability at waypoint location  # [SIMPLIFIED]
            occ_t = torch.sigmoid(occupancy[:, t:t+1, :, :])  # (B, 1, H, W)
            # Grid sample at waypoint position
            grid = norm_traj[:, t:t+1, :].unsqueeze(1)  # (B, 1, 1, 2)
            sampled = F.grid_sample(
                occ_t, grid, mode='bilinear', padding_mode='zeros',
                align_corners=True
            )  # (B, 1, 1, 1)
            collision_loss = collision_loss + sampled.mean()

        collision_loss = collision_loss / max(T, 1)
        return collision_loss


class STP3Loss(nn.Module):  # [FROM PAPER]
    """
    Combined multi-task loss for ST-P3.

    From paper: Joint training with equal weights for all task losses.
    L_total = L_seg + L_occ + L_plan
    """

    def __init__(self, seg_weight: float = 1.0, occ_weight: float = 1.0,
                 plan_weight: float = 1.0, num_seg_classes: int = 4):
        super().__init__()
        # Equal task weights as described in paper  # [FROM PAPER]
        self.seg_weight = seg_weight
        self.occ_weight = occ_weight
        self.plan_weight = plan_weight

        self.seg_loss = BEVSegmentationLoss(num_classes=num_seg_classes)
        self.occ_loss = OccupancyLoss()
        self.plan_loss = PlanningLoss(l1_weight=1.0, collision_weight=1.0)

    def forward(self, predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            predictions: model output dict with 'bev_segmentation', 'future_occupancy', 'trajectory'
            targets: ground truth dict with 'seg_gt', 'occ_gt', 'trajectory_gt'
        Returns:
            dict with individual losses and total loss
        """
        # BEV segmentation loss  # [FROM PAPER]
        seg_loss = self.seg_loss(
            predictions['bev_segmentation'], targets['seg_gt']
        )

        # Future occupancy loss  # [FROM PAPER]
        occ_loss = self.occ_loss(
            predictions['future_occupancy'], targets['occ_gt']
        )

        # Planning loss with collision avoidance  # [FROM PAPER]
        plan_losses = self.plan_loss(
            predictions['trajectory'], targets['trajectory_gt'],
            occupancy_pred=predictions['future_occupancy']
        )

        # Combined multi-task loss (equal weighting as in paper)  # [FROM PAPER]
        total_loss = (
            self.seg_weight * seg_loss +
            self.occ_weight * occ_loss +
            self.plan_weight * plan_losses['total']
        )

        return {
            'total_loss': total_loss,
            'seg_loss': seg_loss,
            'occ_loss': occ_loss,
            'plan_l1_loss': plan_losses['l1_loss'],
            'plan_collision_loss': plan_losses['collision_loss'],
            'plan_total_loss': plan_losses['total'],
        }


# =============================================================================
# Validation Metrics
# =============================================================================

class ValidationMetrics:  # [SELF-IMPLEMENTED]
    """Compute validation metrics for ST-P3 multi-task outputs."""

    def __init__(self, num_seg_classes: int = 4):
        self.num_seg_classes = num_seg_classes
        self.reset()

    def reset(self):
        """Reset accumulated metrics."""
        self.seg_intersection = torch.zeros(self.num_seg_classes)
        self.seg_union = torch.zeros(self.num_seg_classes)
        self.occ_intersection = 0.0
        self.occ_union = 0.0
        self.plan_l2_sum = 0.0
        self.collision_count = 0
        self.total_samples = 0
        self.total_waypoints = 0

    @torch.no_grad()
    def update(self, predictions: Dict[str, torch.Tensor],
               targets: Dict[str, torch.Tensor]):
        """
        Update metrics with a batch of predictions.

        Metrics computed (as in paper evaluation):
        - mIoU for BEV segmentation
        - IoU for future occupancy
        - L2 error for planning
        - Collision rate
        """
        B = predictions['bev_segmentation'].shape[0]
        self.total_samples += B

        # --- mIoU for BEV segmentation ---  # [FROM PAPER]
        seg_pred = predictions['bev_segmentation'].argmax(dim=1)  # (B, H, W)
        seg_gt = targets['seg_gt']  # (B, H, W)
        for cls in range(self.num_seg_classes):
            pred_mask = (seg_pred == cls)
            gt_mask = (seg_gt == cls)
            intersection = (pred_mask & gt_mask).sum().float().cpu()
            union = (pred_mask | gt_mask).sum().float().cpu()
            self.seg_intersection[cls] += intersection
            self.seg_union[cls] += union

        # --- IoU for future occupancy ---  # [FROM PAPER]
        occ_pred = (torch.sigmoid(predictions['future_occupancy']) > 0.5)
        occ_gt = (targets['occ_gt'] > 0.5)
        occ_inter = (occ_pred & occ_gt).sum().float().cpu().item()
        occ_uni = (occ_pred | occ_gt).sum().float().cpu().item()
        self.occ_intersection += occ_inter
        self.occ_union += occ_uni

        # --- Planning L2 error ---  # [FROM PAPER]
        traj_pred = predictions['trajectory']  # (B, T, 2)
        traj_gt = targets['trajectory_gt']     # (B, T, 2)
        l2_per_waypoint = torch.norm(traj_pred - traj_gt, dim=-1)  # (B, T)
        self.plan_l2_sum += l2_per_waypoint.sum().cpu().item()
        self.total_waypoints += l2_per_waypoint.numel()

        # --- Collision rate ---  # [FROM PAPER]
        # Check if any predicted waypoint falls in an occupied cell
        if predictions['future_occupancy'] is not None:
            occ_prob = torch.sigmoid(predictions['future_occupancy'])
            T_plan = traj_pred.shape[1]
            T_occ = occ_prob.shape[1]
            T = min(T_plan, T_occ)

            for b in range(B):
                has_collision = False
                for t in range(T):
                    # Convert trajectory point to grid coordinates  # [SIMPLIFIED]
                    H, W = occ_prob.shape[2], occ_prob.shape[3]
                    x = int(traj_pred[b, t, 0].item() + H // 2)
                    y = int(traj_pred[b, t, 1].item() + W // 2)
                    x = max(0, min(x, H - 1))
                    y = max(0, min(y, W - 1))
                    if occ_prob[b, t, x, y] > 0.5:
                        has_collision = True
                        break
                if has_collision:
                    self.collision_count += 1

    def compute(self) -> Dict[str, float]:
        """Compute final metrics."""
        # mIoU  # [FROM PAPER]
        iou_per_class = self.seg_intersection / (self.seg_union + 1e-6)
        miou = iou_per_class.mean().item()

        # Occupancy IoU  # [FROM PAPER]
        occ_iou = self.occ_intersection / (self.occ_union + 1e-6)

        # Planning L2  # [FROM PAPER]
        plan_l2 = self.plan_l2_sum / max(self.total_waypoints, 1)

        # Collision rate  # [FROM PAPER]
        collision_rate = self.collision_count / max(self.total_samples, 1)

        return {
            'miou': miou,
            'occ_iou': occ_iou,
            'plan_l2': plan_l2,
            'collision_rate': collision_rate,
            'iou_per_class': iou_per_class.tolist(),
        }


# =============================================================================
# Training and Validation Functions
# =============================================================================

def train_one_epoch(model: nn.Module, dataloader: DataLoader,
                    criterion: STP3Loss, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int,
                    scaler: GradScaler = None,
                    grad_clip: float = 5.0) -> Dict[str, float]:  # [SELF-IMPLEMENTED]
    """
    Train for one epoch.

    Features:
    - Mixed precision training (AMP) for memory efficiency
    - Gradient clipping to prevent exploding gradients
    - Progress bar with tqdm
    """
    model.train()
    total_losses = {
        'total_loss': 0.0,
        'seg_loss': 0.0,
        'occ_loss': 0.0,
        'plan_l1_loss': 0.0,
        'plan_collision_loss': 0.0,
    }
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Train]")
    for batch in pbar:
        # Move data to device  # [SELF-IMPLEMENTED]
        images = batch['images'].to(device)          # (B, T, 6, 3, H, W)
        seg_gt = batch['seg_gt'].to(device)          # (B, H, W)
        occ_gt = batch['occ_gt'].to(device)          # (B, T_occ, H, W)
        trajectory_gt = batch['trajectory_gt'].to(device)  # (B, T_plan, 2)

        targets = {
            'seg_gt': seg_gt,
            'occ_gt': occ_gt,
            'trajectory_gt': trajectory_gt,
        }

        optimizer.zero_grad()

        # Mixed precision forward pass  # [SELF-IMPLEMENTED]
        if scaler is not None:
            with autocast(device_type='cuda'):
                predictions = model(images)
                losses = criterion(predictions, targets)
                loss = losses['total_loss']

            # Backward with gradient scaling  # [SELF-IMPLEMENTED]
            scaler.scale(loss).backward()
            # Gradient clipping  # [SELF-IMPLEMENTED]
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            predictions = model(images)
            losses = criterion(predictions, targets)
            loss = losses['total_loss']

            loss.backward()
            # Gradient clipping  # [SELF-IMPLEMENTED]
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        # Accumulate losses  # [SELF-IMPLEMENTED]
        num_batches += 1
        for key in total_losses:
            if key in losses:
                total_losses[key] += losses[key].item()

        # Update progress bar
        pbar.set_postfix({
            'loss': f"{losses['total_loss'].item():.4f}",
            'seg': f"{losses['seg_loss'].item():.4f}",
            'plan': f"{losses['plan_total_loss'].item():.4f}",
        })

    # Average losses  # [SELF-IMPLEMENTED]
    avg_losses = {k: v / max(num_batches, 1) for k, v in total_losses.items()}
    return avg_losses


@torch.no_grad()
def validate(model: nn.Module, dataloader: DataLoader,
             criterion: STP3Loss, device: torch.device,
             epoch: int, num_seg_classes: int = 4) -> Dict[str, float]:  # [SELF-IMPLEMENTED]
    """
    Validate the model and compute metrics.

    Computes:
    - mIoU for BEV segmentation (paper metric)
    - IoU for future occupancy (paper metric)
    - L2 error for planning (paper metric)
    - Collision rate (paper metric)
    """
    model.eval()
    metrics = ValidationMetrics(num_seg_classes=num_seg_classes)
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Val]")
    for batch in pbar:
        images = batch['images'].to(device)
        seg_gt = batch['seg_gt'].to(device)
        occ_gt = batch['occ_gt'].to(device)
        trajectory_gt = batch['trajectory_gt'].to(device)

        targets = {
            'seg_gt': seg_gt,
            'occ_gt': occ_gt,
            'trajectory_gt': trajectory_gt,
        }

        predictions = model(images)
        losses = criterion(predictions, targets)

        total_loss += losses['total_loss'].item()
        num_batches += 1

        # Update validation metrics  # [SELF-IMPLEMENTED]
        metrics.update(predictions, targets)

        pbar.set_postfix({'val_loss': f"{losses['total_loss'].item():.4f}"})

    results = metrics.compute()
    results['val_loss'] = total_loss / max(num_batches, 1)
    return results


# =============================================================================
# Checkpoint Management
# =============================================================================

def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer,
                    scheduler, epoch: int, metrics: Dict,
                    save_dir: str, is_best: bool = False):  # [SELF-IMPLEMENTED]
    """
    Save training checkpoint.

    Saves:
    - Model state dict
    - Optimizer state dict
    - Scheduler state dict
    - Current epoch
    - Metrics at time of saving
    """
    os.makedirs(save_dir, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
    }

    # Save periodic checkpoint  # [SELF-IMPLEMENTED]
    path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth')
    torch.save(checkpoint, path)
    print(f"  Checkpoint saved: {path}")

    # Save latest (always overwritten)  # [SELF-IMPLEMENTED]
    latest_path = os.path.join(save_dir, 'checkpoint_latest.pth')
    torch.save(checkpoint, latest_path)

    # Save best model  # [SELF-IMPLEMENTED]
    if is_best:
        best_path = os.path.join(save_dir, 'checkpoint_best.pth')
        torch.save(checkpoint, best_path)
        print(f"  Best model saved: {best_path}")


def load_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer,
                    scheduler, checkpoint_path: str,
                    device: torch.device) -> int:  # [SELF-IMPLEMENTED]
    """
    Resume training from a checkpoint.

    Returns:
        start_epoch: the epoch to resume from
    """
    if not os.path.isfile(checkpoint_path):
        print(f"No checkpoint found at: {checkpoint_path}")
        return 0

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    epoch = checkpoint['epoch']
    metrics = checkpoint.get('metrics', {})
    print(f"  Resumed from epoch {epoch + 1}, metrics: {metrics}")
    return epoch + 1


# =============================================================================
# Main Training Script
# =============================================================================

def parse_args():  # [SELF-IMPLEMENTED]
    parser = argparse.ArgumentParser(
        description='ST-P3: End-to-End Driving Training Script'
    )
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs (default: 20)')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size (default: 2, keep small due to memory)')
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='Learning rate (default: 2e-4)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay (default: 1e-4)')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device: cuda, cpu, or auto (default: auto)')
    parser.add_argument('--save_dir', type=str, default='./checkpoints/stp3',
                        help='Directory to save checkpoints')

    # Model config
    parser.add_argument('--bev_h', type=int, default=100,
                        help='BEV grid height (default: 100, paper uses 200)')
    parser.add_argument('--bev_w', type=int, default=100,
                        help='BEV grid width (default: 100, paper uses 200)')
    parser.add_argument('--bev_channels', type=int, default=64,
                        help='BEV feature channels (default: 64)')
    parser.add_argument('--num_cameras', type=int, default=6,
                        help='Number of camera views (default: 6)')
    parser.add_argument('--num_seg_classes', type=int, default=4,
                        help='Number of BEV segmentation classes (default: 4)')
    parser.add_argument('--num_waypoints', type=int, default=6,
                        help='Number of planned waypoints (default: 6)')
    parser.add_argument('--temporal_frames', type=int, default=4,
                        help='Number of temporal input frames (default: 4)')

    # Training options
    parser.add_argument('--grad_clip', type=float, default=5.0,
                        help='Gradient clipping max norm (default: 5.0)')
    parser.add_argument('--amp', action='store_true',
                        help='Use automatic mixed precision (AMP)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='DataLoader workers (default: 0)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')

    # Dataset config (synthetic)
    parser.add_argument('--train_samples', type=int, default=64,
                        help='Number of synthetic training samples (default: 64)')
    parser.add_argument('--val_samples', type=int, default=16,
                        help='Number of synthetic validation samples (default: 16)')
    parser.add_argument('--img_h', type=int, default=128,
                        help='Input image height (default: 128, paper uses 224)')
    parser.add_argument('--img_w', type=int, default=256,
                        help='Input image width (default: 256, paper uses 400)')

    return parser.parse_args()


def main():
    args = parse_args()

    # Device setup  # [SELF-IMPLEMENTED]
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # =========================================================================
    # Model
    # =========================================================================
    print("\n" + "=" * 60)
    print("ST-P3: Spatial Temporal Feature Learning for E2E Driving")
    print("=" * 60)

    model = STP3(  # [FROM PAPER]
        bev_channels=args.bev_channels,
        bev_h=args.bev_h,
        bev_w=args.bev_w,
        num_cameras=args.num_cameras,
        num_seg_classes=args.num_seg_classes,
        num_waypoints=args.num_waypoints,
        temporal_frames=args.temporal_frames,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # =========================================================================
    # Dataset and DataLoader
    # =========================================================================
    print("\nCreating synthetic datasets...")
    train_dataset = STP3Dataset(  # [SELF-IMPLEMENTED]
        num_samples=args.train_samples,
        temporal_frames=args.temporal_frames,
        num_cameras=args.num_cameras,
        img_h=args.img_h,
        img_w=args.img_w,
        bev_h=args.bev_h,
        bev_w=args.bev_w,
        num_seg_classes=args.num_seg_classes,
        num_future_steps=5,
        num_waypoints=args.num_waypoints,
    )
    val_dataset = STP3Dataset(  # [SELF-IMPLEMENTED]
        num_samples=args.val_samples,
        temporal_frames=args.temporal_frames,
        num_cameras=args.num_cameras,
        img_h=args.img_h,
        img_w=args.img_w,
        bev_h=args.bev_h,
        bev_w=args.bev_w,
        num_seg_classes=args.num_seg_classes,
        num_future_steps=5,
        num_waypoints=args.num_waypoints,
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
    print(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  Val:   {len(val_dataset)} samples, {len(val_loader)} batches")

    # =========================================================================
    # Loss, Optimizer, Scheduler
    # =========================================================================
    criterion = STP3Loss(  # [FROM PAPER]
        seg_weight=1.0,
        occ_weight=1.0,
        plan_weight=1.0,
        num_seg_classes=args.num_seg_classes,
    ).to(device)

    optimizer = torch.optim.AdamW(  # [SELF-IMPLEMENTED]
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Cosine annealing LR schedule  # [SELF-IMPLEMENTED]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    # Mixed precision scaler  # [SELF-IMPLEMENTED]
    scaler = GradScaler() if (args.amp and device.type == 'cuda') else None
    if scaler:
        print("  Mixed precision (AMP): enabled")

    # =========================================================================
    # Resume from checkpoint
    # =========================================================================
    start_epoch = 0
    if args.resume:  # [SELF-IMPLEMENTED]
        start_epoch = load_checkpoint(
            model, optimizer, scheduler, args.resume, device
        )

    # =========================================================================
    # Training Loop
    # =========================================================================
    print(f"\nStarting training from epoch {start_epoch + 1} to {args.epochs}")
    print(f"  LR: {args.lr}, Batch size: {args.batch_size}")
    print(f"  Grad clip: {args.grad_clip}, Weight decay: {args.weight_decay}")
    print("-" * 60)

    best_val_loss = float('inf')

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()

        # Train  # [SELF-IMPLEMENTED]
        train_losses = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            scaler=scaler,
            grad_clip=args.grad_clip,
        )

        # Validate  # [SELF-IMPLEMENTED]
        val_metrics = validate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
            num_seg_classes=args.num_seg_classes,
        )

        # Step LR scheduler  # [SELF-IMPLEMENTED]
        scheduler.step()

        # Logging
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']

        print(f"\nEpoch {epoch+1}/{args.epochs} ({epoch_time:.1f}s) | LR: {current_lr:.6f}")
        print(f"  Train - Total: {train_losses['total_loss']:.4f} | "
              f"Seg: {train_losses['seg_loss']:.4f} | "
              f"Occ: {train_losses['occ_loss']:.4f} | "
              f"Plan: {train_losses['plan_l1_loss']:.4f}")
        print(f"  Val   - Loss: {val_metrics['val_loss']:.4f} | "
              f"mIoU: {val_metrics['miou']:.4f} | "
              f"Occ IoU: {val_metrics['occ_iou']:.4f} | "
              f"Plan L2: {val_metrics['plan_l2']:.4f} | "
              f"Collision: {val_metrics['collision_rate']:.4f}")

        # Checkpoint  # [SELF-IMPLEMENTED]
        is_best = val_metrics['val_loss'] < best_val_loss
        if is_best:
            best_val_loss = val_metrics['val_loss']

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            metrics=val_metrics,
            save_dir=args.save_dir,
            is_best=is_best,
        )

    # =========================================================================
    # Final Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"  Best validation loss: {best_val_loss:.4f}")
    print(f"  Checkpoints saved to: {os.path.abspath(args.save_dir)}")
    print(f"  Best model: {os.path.join(args.save_dir, 'checkpoint_best.pth')}")


if __name__ == '__main__':
    main()
