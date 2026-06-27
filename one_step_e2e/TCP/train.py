"""
TCP Training Script
===================
ATTRIBUTION:
- Loss functions: Based on TCP paper (Wu et al., NeurIPS 2022)
  - Trajectory branch: L1 waypoint loss (Section 3.2)
  - Control branch: L1 loss on steer/throttle/brake (Section 3.3)
  - Fused steering loss: L1 on adaptive-fusion output (Section 3.4)
  - Speed prediction auxiliary: L1 loss
- Training strategy: Joint multi-task from paper, balanced loss weighting
- Adaptive fusion gate: Learned sigmoid gate (from paper Section 3.4)
- Implementation: Self-implemented in PyTorch (simplified from official TCP)
- Synthetic dataset: Self-implemented for demonstration (real uses CARLA data)
"""

import argparse
import os
import time
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.amp import GradScaler, autocast

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm not installed  # [SELF-IMPLEMENTED]
    def tqdm(iterable, **kwargs):
        desc = kwargs.get("desc", "")
        total = kwargs.get("total", None)
        for i, item in enumerate(iterable):
            if total:
                print(f"\r{desc} [{i+1}/{total}]", end="", flush=True)
            yield item
        print()

from model import TCP


# =============================================================================
# Synthetic Dataset  # [SELF-IMPLEMENTED]
# =============================================================================

class TCPDataset(Dataset):
    """
    Synthetic dataset for TCP training demonstration.

    In the real TCP pipeline, data comes from CARLA simulator recordings with:
    - Front-facing camera images
    - LiDAR point clouds projected to BEV
    - Vehicle speed from CAN bus
    - Expert waypoints from autopilot planner
    - Expert control signals (steer, throttle, brake)

    This synthetic version generates random but physically plausible data
    for demonstration and testing purposes.
    """  # [SELF-IMPLEMENTED]

    def __init__(self, num_samples: int = 1000, num_waypoints: int = 4,
                 img_h: int = 256, img_w: int = 512,
                 lidar_size: int = 256, seed: int = 42):
        """
        Args:
            num_samples: Number of synthetic samples to generate.
            num_waypoints: Number of future waypoints (T).
            img_h: Image height.
            img_w: Image width.
            lidar_size: LiDAR BEV spatial resolution.
            seed: Random seed for reproducibility.
        """  # [SELF-IMPLEMENTED]
        super().__init__()
        self.num_samples = num_samples
        self.num_waypoints = num_waypoints
        self.img_h = img_h
        self.img_w = img_w
        self.lidar_size = lidar_size

        # Pre-generate random seed offsets for reproducibility  # [SELF-IMPLEMENTED]
        self.rng = torch.Generator()
        self.rng.manual_seed(seed)
        self.seed_offsets = torch.randint(0, 100000, (num_samples,),
                                          generator=self.rng)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Generate a single synthetic sample."""  # [SELF-IMPLEMENTED]
        # Use deterministic seed per sample for reproducibility
        g = torch.Generator()
        g.manual_seed(int(self.seed_offsets[idx].item()))

        # Front camera image: (3, 256, 512) - synthetic noise pattern  # [SELF-IMPLEMENTED]
        image = torch.randn(3, self.img_h, self.img_w, generator=g) * 0.5 + 0.5
        image = image.clamp(0, 1)

        # LiDAR BEV: (2, 256, 256) - channel 0: occupancy, channel 1: height  # [SELF-IMPLEMENTED]
        lidar_bev = torch.randn(2, self.lidar_size, self.lidar_size, generator=g)
        lidar_bev = lidar_bev.clamp(-1, 1)

        # Speed: (1,) - range [0, 40] m/s (typical driving)  # [SELF-IMPLEMENTED]
        speed = torch.rand(1, generator=g) * 40.0

        # Generate physically plausible waypoints  # [SELF-IMPLEMENTED]
        # Waypoints are in ego vehicle frame: (T, 2) for (dx, dy)
        # Vehicle moves forward (positive x) with lateral offset (y)
        base_steer = (torch.rand(1, generator=g).item() - 0.5) * 0.6  # [-0.3, 0.3]
        waypoints = torch.zeros(self.num_waypoints, 2)
        for t in range(self.num_waypoints):
            dt = (t + 1) * 0.5  # 0.5s intervals
            waypoints[t, 0] = dt * (speed.item() / 10.0 + 1.0)  # forward
            waypoints[t, 1] = dt * base_steer * 2.0  # lateral

        # Add small noise to waypoints  # [SELF-IMPLEMENTED]
        waypoints += torch.randn_like(waypoints, generator=g) * 0.1

        # Control signals: (3,) [steer, throttle, brake]  # [SELF-IMPLEMENTED]
        steer = torch.tensor([base_steer]).clamp(-1, 1)
        # Simple throttle/brake logic based on speed
        if speed.item() < 20.0:
            throttle = torch.rand(1, generator=g) * 0.6 + 0.2  # accelerating
            brake = torch.zeros(1)
        else:
            throttle = torch.rand(1, generator=g) * 0.3
            brake = torch.rand(1, generator=g) * 0.3
        control = torch.cat([steer, throttle, brake], dim=0)

        return {
            'image': image,           # (3, H, W)
            'lidar_bev': lidar_bev,   # (2, lidar_size, lidar_size)
            'speed': speed,           # (1,)
            'waypoints': waypoints,   # (T, 2)
            'control': control,       # (3,) [steer, throttle, brake]
        }


# =============================================================================
# Loss Functions  # [FROM PAPER]
# =============================================================================

class TCPLoss(nn.Module):
    """
    Multi-task loss for TCP training.

    From TCP paper (Wu et al., NeurIPS 2022):
    - Trajectory branch: L1 loss on predicted waypoints (Section 3.2)
    - Control branch: L1 loss on steer/throttle/brake (Section 3.3)
    - Fused steering: L1 loss on adaptive fusion output (Section 3.4)
    - Speed prediction: Auxiliary L1 loss for speed estimation

    Total loss = w_traj * L_traj + w_ctrl * L_ctrl + w_fuse * L_fuse + w_spd * L_spd
    """  # [FROM PAPER]

    def __init__(self, w_traj: float = 1.0, w_ctrl: float = 1.0,
                 w_fuse: float = 0.5, w_spd: float = 0.5):
        """
        Args:
            w_traj: Weight for trajectory (waypoint) loss.
            w_ctrl: Weight for control signal loss.
            w_fuse: Weight for fused steering loss.
            w_spd: Weight for speed prediction loss.
        """  # [FROM PAPER] - balanced weighting from paper
        super().__init__()
        self.w_traj = w_traj
        self.w_ctrl = w_ctrl
        self.w_fuse = w_fuse
        self.w_spd = w_spd

    def forward(self, output: Dict[str, torch.Tensor],
                gt_waypoints: torch.Tensor,
                gt_control: torch.Tensor,
                gt_speed: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task TCP loss.

        Args:
            output: Model output dict with keys:
                - 'waypoints': (B, T, 2) predicted waypoints
                - 'control': (B, 3) predicted [steer, throttle, brake]
                - 'fused_steer': (B, 1) fused steering prediction
            gt_waypoints: (B, T, 2) ground truth waypoints.
            gt_control: (B, 3) ground truth [steer, throttle, brake].
            gt_speed: (B, 1) ground truth speed (for auxiliary loss).

        Returns:
            Dict with individual losses and weighted total.
        """
        # Trajectory L1 loss  # [FROM PAPER] - Section 3.2
        traj_loss = F.l1_loss(output['waypoints'], gt_waypoints)

        # Control L1 loss  # [FROM PAPER] - Section 3.3
        ctrl_loss = F.l1_loss(output['control'], gt_control)

        # Fused steering L1 loss  # [FROM PAPER] - Section 3.4
        fuse_loss = F.l1_loss(output['fused_steer'], gt_control[:, 0:1])

        # Speed auxiliary loss  # [FROM PAPER] - auxiliary task
        # Note: model doesn't have separate speed head; we use trajectory
        # first-step magnitude as speed proxy for the auxiliary loss
        pred_speed_proxy = torch.norm(
            output['waypoints'][:, 0, :], dim=-1, keepdim=True
        )  # [SIMPLIFIED] - proxy for speed since model lacks explicit speed head
        spd_loss = F.l1_loss(pred_speed_proxy, gt_speed / 10.0)  # normalize

        # Weighted total  # [FROM PAPER]
        total_loss = (
            self.w_traj * traj_loss +
            self.w_ctrl * ctrl_loss +
            self.w_fuse * fuse_loss +
            self.w_spd * spd_loss
        )

        return {
            'total': total_loss,
            'trajectory': traj_loss,
            'control': ctrl_loss,
            'fused_steer': fuse_loss,
            'speed': spd_loss,
        }


# =============================================================================
# Validation Metrics  # [SELF-IMPLEMENTED]
# =============================================================================

class TCPMetrics:
    """
    Validation metrics for TCP evaluation.

    Computes:
    - Waypoint L1 error (meters, averaged over timesteps)
    - Control L1 error (per-signal)
    - Fused steering MAE
    - Driving score proxies (simplified from CARLA benchmark)
    """  # [SELF-IMPLEMENTED]

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset accumulated metrics."""  # [SELF-IMPLEMENTED]
        self.waypoint_l1_sum = 0.0
        self.control_l1_sum = 0.0
        self.steer_l1_sum = 0.0
        self.throttle_l1_sum = 0.0
        self.brake_l1_sum = 0.0
        self.fused_steer_l1_sum = 0.0
        self.gate_mean_sum = 0.0
        self.lateral_error_sum = 0.0
        self.longitudinal_error_sum = 0.0
        self.count = 0

    @torch.no_grad()
    def update(self, output: Dict[str, torch.Tensor],
               gt_waypoints: torch.Tensor,
               gt_control: torch.Tensor):
        """
        Update metrics with a batch of predictions.

        Args:
            output: Model output dict.
            gt_waypoints: (B, T, 2) ground truth waypoints.
            gt_control: (B, 3) ground truth control.
        """  # [SELF-IMPLEMENTED]
        B = gt_waypoints.shape[0]

        # Waypoint L1 (average over all waypoints)  # [SELF-IMPLEMENTED]
        wp_error = F.l1_loss(output['waypoints'], gt_waypoints, reduction='none')
        self.waypoint_l1_sum += wp_error.mean().item() * B

        # Per-signal control errors  # [SELF-IMPLEMENTED]
        ctrl_error = (output['control'] - gt_control).abs()
        self.steer_l1_sum += ctrl_error[:, 0].mean().item() * B
        self.throttle_l1_sum += ctrl_error[:, 1].mean().item() * B
        self.brake_l1_sum += ctrl_error[:, 2].mean().item() * B
        self.control_l1_sum += ctrl_error.mean().item() * B

        # Fused steer error  # [SELF-IMPLEMENTED]
        fused_err = (output['fused_steer'] - gt_control[:, 0:1]).abs()
        self.fused_steer_l1_sum += fused_err.mean().item() * B

        # Fusion gate statistics  # [SELF-IMPLEMENTED]
        self.gate_mean_sum += output['fusion_gate'].mean().item() * B

        # Driving metrics (simplified proxies)  # [SIMPLIFIED]
        # Lateral error: deviation of last waypoint in y-direction
        lat_err = (output['waypoints'][:, -1, 1] - gt_waypoints[:, -1, 1]).abs()
        self.lateral_error_sum += lat_err.mean().item() * B

        # Longitudinal error: deviation in x-direction
        lon_err = (output['waypoints'][:, -1, 0] - gt_waypoints[:, -1, 0]).abs()
        self.longitudinal_error_sum += lon_err.mean().item() * B

        self.count += B

    def compute(self) -> Dict[str, float]:
        """Compute final metrics."""  # [SELF-IMPLEMENTED]
        if self.count == 0:
            return {}
        n = self.count
        return {
            'waypoint_l1': self.waypoint_l1_sum / n,
            'control_l1': self.control_l1_sum / n,
            'steer_l1': self.steer_l1_sum / n,
            'throttle_l1': self.throttle_l1_sum / n,
            'brake_l1': self.brake_l1_sum / n,
            'fused_steer_l1': self.fused_steer_l1_sum / n,
            'fusion_gate_mean': self.gate_mean_sum / n,
            'lateral_error': self.lateral_error_sum / n,
            'longitudinal_error': self.longitudinal_error_sum / n,
        }


# =============================================================================
# Checkpoint Management  # [SELF-IMPLEMENTED]
# =============================================================================

def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer,
                    scheduler, scaler: GradScaler, epoch: int,
                    best_metric: float, metrics: Dict, save_path: str):
    """
    Save training checkpoint.

    Args:
        model: TCP model.
        optimizer: Optimizer state.
        scheduler: LR scheduler state.
        scaler: AMP GradScaler state.
        epoch: Current epoch number.
        best_metric: Best validation metric so far.
        metrics: Current validation metrics dict.
        save_path: Path to save checkpoint file.
    """  # [SELF-IMPLEMENTED]
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'scaler_state_dict': scaler.state_dict(),
        'best_metric': best_metric,
        'metrics': metrics,
    }
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    torch.save(checkpoint, save_path)
    print(f"  [Checkpoint] Saved to {save_path}")


def load_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer,
                    scheduler, scaler: GradScaler,
                    checkpoint_path: str, device: torch.device) -> Tuple[int, float]:
    """
    Load training checkpoint.

    Returns:
        Tuple of (start_epoch, best_metric).
    """  # [SELF-IMPLEMENTED]
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    epoch = checkpoint['epoch']
    best_metric = checkpoint['best_metric']
    print(f"  [Checkpoint] Resumed from epoch {epoch}, best_metric={best_metric:.4f}")
    return epoch + 1, best_metric


# =============================================================================
# Training Loop  # [SELF-IMPLEMENTED]
# =============================================================================

def train_one_epoch(model: nn.Module, dataloader: DataLoader,
                    criterion: TCPLoss, optimizer: torch.optim.Optimizer,
                    scaler: GradScaler, device: torch.device,
                    epoch: int, max_grad_norm: float = 1.0,
                    use_amp: bool = True) -> Dict[str, float]:
    """
    Train for one epoch.

    Args:
        model: TCP model.
        dataloader: Training data loader.
        criterion: TCPLoss criterion.
        optimizer: Optimizer.
        scaler: AMP GradScaler.
        device: Compute device.
        epoch: Current epoch number.
        max_grad_norm: Gradient clipping max norm.
        use_amp: Whether to use automatic mixed precision.

    Returns:
        Dict of average training losses.
    """  # [SELF-IMPLEMENTED]
    model.train()

    loss_accum = {'total': 0.0, 'trajectory': 0.0, 'control': 0.0,
                  'fused_steer': 0.0, 'speed': 0.0}
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Train Epoch {epoch}", leave=True)
    for batch in pbar:
        # Move to device  # [SELF-IMPLEMENTED]
        image = batch['image'].to(device, non_blocking=True)
        lidar_bev = batch['lidar_bev'].to(device, non_blocking=True)
        speed = batch['speed'].to(device, non_blocking=True)
        gt_waypoints = batch['waypoints'].to(device, non_blocking=True)
        gt_control = batch['control'].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)  # [SELF-IMPLEMENTED] - memory efficient

        # Forward pass with AMP  # [SELF-IMPLEMENTED]
        with autocast(device_type=device.type, enabled=use_amp):
            output = model(image, lidar_bev, speed)
            losses = criterion(output, gt_waypoints, gt_control, speed)

        # Backward pass with gradient scaling  # [SELF-IMPLEMENTED]
        scaler.scale(losses['total']).backward()

        # Gradient clipping  # [SELF-IMPLEMENTED]
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        scaler.step(optimizer)
        scaler.update()

        # Accumulate losses  # [SELF-IMPLEMENTED]
        for key in loss_accum:
            loss_accum[key] += losses[key].item()
        num_batches += 1

        # Update progress bar  # [SELF-IMPLEMENTED]
        pbar.set_postfix({
            'loss': f"{losses['total'].item():.4f}",
            'traj': f"{losses['trajectory'].item():.4f}",
            'ctrl': f"{losses['control'].item():.4f}",
        })

    # Average losses  # [SELF-IMPLEMENTED]
    avg_losses = {k: v / max(num_batches, 1) for k, v in loss_accum.items()}
    return avg_losses


@torch.no_grad()
def validate(model: nn.Module, dataloader: DataLoader,
             criterion: TCPLoss, device: torch.device,
             epoch: int, use_amp: bool = True) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Validate the model.

    Args:
        model: TCP model.
        dataloader: Validation data loader.
        criterion: TCPLoss criterion.
        device: Compute device.
        epoch: Current epoch number.
        use_amp: Whether to use AMP.

    Returns:
        Tuple of (avg_losses dict, metrics dict).
    """  # [SELF-IMPLEMENTED]
    model.eval()

    loss_accum = {'total': 0.0, 'trajectory': 0.0, 'control': 0.0,
                  'fused_steer': 0.0, 'speed': 0.0}
    num_batches = 0
    metrics = TCPMetrics()

    pbar = tqdm(dataloader, desc=f"Val   Epoch {epoch}", leave=True)
    for batch in pbar:
        # Move to device  # [SELF-IMPLEMENTED]
        image = batch['image'].to(device, non_blocking=True)
        lidar_bev = batch['lidar_bev'].to(device, non_blocking=True)
        speed = batch['speed'].to(device, non_blocking=True)
        gt_waypoints = batch['waypoints'].to(device, non_blocking=True)
        gt_control = batch['control'].to(device, non_blocking=True)

        # Forward  # [SELF-IMPLEMENTED]
        with autocast(device_type=device.type, enabled=use_amp):
            output = model(image, lidar_bev, speed)
            losses = criterion(output, gt_waypoints, gt_control, speed)

        # Accumulate losses  # [SELF-IMPLEMENTED]
        for key in loss_accum:
            loss_accum[key] += losses[key].item()
        num_batches += 1

        # Update metrics  # [SELF-IMPLEMENTED]
        metrics.update(output, gt_waypoints, gt_control)

        pbar.set_postfix({'val_loss': f"{losses['total'].item():.4f}"})

    avg_losses = {k: v / max(num_batches, 1) for k, v in loss_accum.items()}
    computed_metrics = metrics.compute()

    return avg_losses, computed_metrics


# =============================================================================
# Main Training Function  # [SELF-IMPLEMENTED]
# =============================================================================

def main():
    """Main training entry point with argparse configuration."""  # [SELF-IMPLEMENTED]

    parser = argparse.ArgumentParser(
        description="TCP: Trajectory-guided Control Prediction - Training Script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model arguments  # [SELF-IMPLEMENTED]
    parser.add_argument('--num-waypoints', type=int, default=4,
                        help='Number of future waypoints to predict')
    parser.add_argument('--hidden-dim', type=int, default=256,
                        help='Hidden dimension for feature representations')

    # Data arguments  # [SELF-IMPLEMENTED]
    parser.add_argument('--train-samples', type=int, default=800,
                        help='Number of synthetic training samples')
    parser.add_argument('--val-samples', type=int, default=200,
                        help='Number of synthetic validation samples')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Training batch size')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='DataLoader worker processes')
    parser.add_argument('--img-h', type=int, default=256,
                        help='Input image height')
    parser.add_argument('--img-w', type=int, default=512,
                        help='Input image width')
    parser.add_argument('--lidar-size', type=int, default=256,
                        help='LiDAR BEV spatial resolution')

    # Training arguments  # [SELF-IMPLEMENTED]
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay (L2 regularization)')
    parser.add_argument('--max-grad-norm', type=float, default=1.0,
                        help='Max gradient norm for clipping')
    parser.add_argument('--warmup-epochs', type=int, default=3,
                        help='Number of warmup epochs for LR scheduler')

    # Loss weights  # [FROM PAPER]
    parser.add_argument('--w-traj', type=float, default=1.0,
                        help='Trajectory loss weight (from paper)')
    parser.add_argument('--w-ctrl', type=float, default=1.0,
                        help='Control loss weight (from paper)')
    parser.add_argument('--w-fuse', type=float, default=0.5,
                        help='Fused steer loss weight (from paper)')
    parser.add_argument('--w-spd', type=float, default=0.5,
                        help='Speed auxiliary loss weight')

    # AMP and device  # [SELF-IMPLEMENTED]
    parser.add_argument('--no-amp', action='store_true',
                        help='Disable automatic mixed precision')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device: auto, cuda, cpu')

    # Checkpoint arguments  # [SELF-IMPLEMENTED]
    parser.add_argument('--output-dir', type=str, default='./checkpoints_tcp',
                        help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--save-every', type=int, default=5,
                        help='Save checkpoint every N epochs')

    # Misc  # [SELF-IMPLEMENTED]
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--print-freq', type=int, default=10,
                        help='Print frequency (batches)')

    args = parser.parse_args()

    # =========================================================================
    # Setup  # [SELF-IMPLEMENTED]
    # =========================================================================

    # Seed everything  # [SELF-IMPLEMENTED]
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Device selection  # [SELF-IMPLEMENTED]
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    use_amp = (not args.no_amp) and (device.type == 'cuda')

    print("=" * 70)
    print("TCP: Trajectory-guided Control Prediction - Training")
    print("  Paper: Wu et al., NeurIPS 2022")
    print("=" * 70)
    print(f"  Device:        {device}")
    print(f"  AMP:           {use_amp}")
    print(f"  Epochs:        {args.epochs}")
    print(f"  Batch size:    {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Loss weights:  traj={args.w_traj}, ctrl={args.w_ctrl}, "
          f"fuse={args.w_fuse}, spd={args.w_spd}")
    print(f"  Output dir:    {args.output_dir}")
    print("=" * 70)

    # =========================================================================
    # Dataset and DataLoaders  # [SELF-IMPLEMENTED]
    # =========================================================================

    print("\n[1/4] Creating synthetic datasets...")

    train_dataset = TCPDataset(
        num_samples=args.train_samples,
        num_waypoints=args.num_waypoints,
        img_h=args.img_h,
        img_w=args.img_w,
        lidar_size=args.lidar_size,
        seed=args.seed,
    )  # [SELF-IMPLEMENTED]

    val_dataset = TCPDataset(
        num_samples=args.val_samples,
        num_waypoints=args.num_waypoints,
        img_h=args.img_h,
        img_w=args.img_w,
        lidar_size=args.lidar_size,
        seed=args.seed + 1000,  # Different seed for validation
    )  # [SELF-IMPLEMENTED]

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
        drop_last=True,
    )  # [SELF-IMPLEMENTED]

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
        drop_last=False,
    )  # [SELF-IMPLEMENTED]

    print(f"  Train samples: {len(train_dataset)}, "
          f"batches: {len(train_loader)}")
    print(f"  Val samples:   {len(val_dataset)}, "
          f"batches: {len(val_loader)}")

    # =========================================================================
    # Model, Loss, Optimizer, Scheduler  # [SELF-IMPLEMENTED]
    # =========================================================================

    print("\n[2/4] Initializing model and optimizer...")

    model = TCP(
        num_waypoints=args.num_waypoints,
        hidden_dim=args.hidden_dim,
        img_channels=3,
        lidar_channels=2,
    ).to(device)  # [FROM PAPER] - TCP architecture

    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters:     {num_params:,}")
    print(f"  Trainable parameters: {num_trainable:,}")

    # Loss function with paper-specified weights  # [FROM PAPER]
    criterion = TCPLoss(
        w_traj=args.w_traj,
        w_ctrl=args.w_ctrl,
        w_fuse=args.w_fuse,
        w_spd=args.w_spd,
    )

    # AdamW optimizer  # [SELF-IMPLEMENTED]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )

    # Cosine annealing with warmup  # [SELF-IMPLEMENTED]
    def lr_lambda(epoch):
        """Linear warmup + cosine decay schedule."""
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        else:
            import math
            progress = (epoch - args.warmup_epochs) / max(
                1, args.epochs - args.warmup_epochs)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # AMP Grad Scaler  # [SELF-IMPLEMENTED]
    scaler = GradScaler(device.type, enabled=use_amp)

    # =========================================================================
    # Resume from checkpoint  # [SELF-IMPLEMENTED]
    # =========================================================================

    start_epoch = 0
    best_metric = float('inf')  # Lower is better (validation loss)

    if args.resume and os.path.isfile(args.resume):
        print(f"\n  Resuming from checkpoint: {args.resume}")
        start_epoch, best_metric = load_checkpoint(
            model, optimizer, scheduler, scaler, args.resume, device)

    # =========================================================================
    # Training Loop  # [SELF-IMPLEMENTED]
    # =========================================================================

    print("\n[3/4] Starting training loop...")
    print("-" * 70)

    os.makedirs(args.output_dir, exist_ok=True)
    training_start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch+1}/{args.epochs} (lr={current_lr:.2e})")
        print("-" * 40)

        # Train  # [SELF-IMPLEMENTED]
        train_losses = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            epoch=epoch + 1,
            max_grad_norm=args.max_grad_norm,
            use_amp=use_amp,
        )

        # Validate  # [SELF-IMPLEMENTED]
        val_losses, val_metrics = validate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch + 1,
            use_amp=use_amp,
        )

        # Step scheduler  # [SELF-IMPLEMENTED]
        scheduler.step()

        # Print epoch summary  # [SELF-IMPLEMENTED]
        epoch_time = time.time() - epoch_start
        print(f"\n  [Train] total={train_losses['total']:.4f} "
              f"traj={train_losses['trajectory']:.4f} "
              f"ctrl={train_losses['control']:.4f} "
              f"fuse={train_losses['fused_steer']:.4f} "
              f"spd={train_losses['speed']:.4f}")
        print(f"  [Val]   total={val_losses['total']:.4f} "
              f"traj={val_losses['trajectory']:.4f} "
              f"ctrl={val_losses['control']:.4f} "
              f"fuse={val_losses['fused_steer']:.4f} "
              f"spd={val_losses['speed']:.4f}")
        print(f"  [Metrics] waypoint_L1={val_metrics['waypoint_l1']:.4f} "
              f"steer_L1={val_metrics['steer_l1']:.4f} "
              f"fused_steer_L1={val_metrics['fused_steer_l1']:.4f}")
        print(f"  [Metrics] lateral_err={val_metrics['lateral_error']:.4f} "
              f"longitudinal_err={val_metrics['longitudinal_error']:.4f} "
              f"gate_mean={val_metrics['fusion_gate_mean']:.3f}")
        print(f"  [Time]  {epoch_time:.1f}s")

        # Checkpoint management  # [SELF-IMPLEMENTED]
        is_best = val_losses['total'] < best_metric
        if is_best:
            best_metric = val_losses['total']
            save_checkpoint(
                model, optimizer, scheduler, scaler, epoch,
                best_metric, val_metrics,
                os.path.join(args.output_dir, 'best_model.pth')
            )
            print(f"  ** New best model! val_loss={best_metric:.4f}")

        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(
                model, optimizer, scheduler, scaler, epoch,
                best_metric, val_metrics,
                os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1:03d}.pth')
            )

    # Save final model  # [SELF-IMPLEMENTED]
    save_checkpoint(
        model, optimizer, scheduler, scaler, args.epochs - 1,
        best_metric, val_metrics,
        os.path.join(args.output_dir, 'final_model.pth')
    )

    # =========================================================================
    # Training Summary  # [SELF-IMPLEMENTED]
    # =========================================================================

    total_time = time.time() - training_start_time
    print("\n" + "=" * 70)
    print("[4/4] Training Complete!")
    print("=" * 70)
    print(f"  Total time:       {total_time/60:.1f} minutes")
    print(f"  Best val loss:    {best_metric:.4f}")
    print(f"  Final val metrics:")
    for key, value in val_metrics.items():
        print(f"    {key:25s}: {value:.4f}")
    print(f"  Checkpoints saved to: {os.path.abspath(args.output_dir)}")
    print("=" * 70)


if __name__ == '__main__':
    main()
