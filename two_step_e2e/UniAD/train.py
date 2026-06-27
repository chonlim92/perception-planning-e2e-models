"""
UniAD Training Script
=====================
ATTRIBUTION:
- Loss functions: Based on the UniAD paper (Li et al., CVPR 2023)
  - Planning L2 loss, collision loss from paper Section 3.5
  - Multi-task weighting strategy from paper Section 4.1
- Training strategy: 3-stage training from paper (perception -> motion -> planning)
- Implementation: Self-implemented in PyTorch (simplified from mmdet3d-based original)
- Synthetic dataset: Self-implemented for demonstration (real training uses nuScenes)
"""

import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Optional, Tuple

try:
    from tqdm import tqdm
except ImportError:
    # [SELF-IMPLEMENTED] Fallback if tqdm is not installed
    def tqdm(iterable, **kwargs):
        return iterable

from model import UniAD
from config import UniADConfig


# =============================================================================
# Synthetic Dataset
# =============================================================================

class UniADDataset(Dataset):
    """
    [SELF-IMPLEMENTED] Synthetic dataset for UniAD training demonstration.

    Generates random multi-view camera images and ground truth annotations.
    In real training, this would load nuScenes data with:
    - 6 surround-view camera images (front, front-left, front-right, back, back-left, back-right)
    - 3D bounding box annotations for all agents
    - HD map polylines (lane dividers, road boundaries, crossings)
    - Expert ego trajectory from CAN bus data

    This synthetic version allows the training loop to run end-to-end without
    any external data dependencies.
    """

    def __init__(self, num_samples: int = 100, config: Optional[UniADConfig] = None):
        """
        Args:
            num_samples: Number of synthetic samples to generate
            config: UniAD configuration for determining output shapes
        """
        super().__init__()
        self.num_samples = num_samples
        self.config = config or UniADConfig()

        # [SELF-IMPLEMENTED] Image dimensions (downscaled from original 900x1600)
        self.img_h = 224
        self.img_w = 400
        self.num_cameras = 6

        # [FROM PAPER] Ground truth dimensions match model output
        self.num_future_steps = self.config.planner.num_future_steps  # 6 steps = 3s at 2Hz
        self.num_agents = 20  # max agents per scene for GT
        self.num_map_polylines = 30  # max map elements
        self.num_points_per_polyline = self.config.map.num_points_per_polyline

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns a dictionary with:
            - images: (6, 3, 224, 400) multi-view camera images
            - gt_trajectory: (6, 2) ego future trajectory in BEV coords
            - gt_agent_boxes: (num_agents, 10) 3D bounding boxes
            - gt_agent_classes: (num_agents,) class labels
            - gt_agent_mask: (num_agents,) valid agent mask
            - gt_map_polylines: (num_map_polylines, num_points, 2) map polyline points
            - gt_map_classes: (num_map_polylines,) polyline class labels
            - gt_map_mask: (num_map_polylines,) valid polyline mask
            - gt_agent_futures: (num_agents, 12, 2) agent future trajectories
            - gt_agent_future_mask: (num_agents,) valid future mask
        """
        # [SELF-IMPLEMENTED] Seed based on index for reproducibility
        rng = np.random.RandomState(idx)

        # Multi-view camera images (synthetic noise)
        images = torch.randn(self.num_cameras, 3, self.img_h, self.img_w) * 0.5

        # [FROM PAPER] Ego trajectory: smooth forward-moving trajectory
        # Simulate a vehicle moving forward with slight lateral variation
        dt = 0.5  # 2Hz sampling
        speeds = 5.0 + rng.randn() * 2.0  # ~5 m/s forward speed
        gt_trajectory = torch.zeros(self.num_future_steps, 2)
        for t in range(self.num_future_steps):
            gt_trajectory[t, 0] = speeds * dt * (t + 1)  # forward (x)
            gt_trajectory[t, 1] = rng.randn() * 0.3 * (t + 1)  # lateral (y)

        # Agent bounding boxes: (cx, cy, cz, w, l, h, sin, cos, vx, vy)
        num_valid_agents = rng.randint(5, self.num_agents + 1)
        gt_agent_boxes = torch.zeros(self.num_agents, 10)
        gt_agent_classes = torch.zeros(self.num_agents, dtype=torch.long)
        gt_agent_mask = torch.zeros(self.num_agents, dtype=torch.bool)

        for i in range(num_valid_agents):
            cx = rng.uniform(-30, 30)
            cy = rng.uniform(-15, 15)
            cz = rng.uniform(-1, 1)
            w = rng.uniform(1.5, 2.5)
            l = rng.uniform(3.5, 5.0)
            h = rng.uniform(1.4, 2.0)
            heading = rng.uniform(-np.pi, np.pi)
            vx = rng.uniform(-5, 10)
            vy = rng.uniform(-2, 2)
            gt_agent_boxes[i] = torch.tensor([cx, cy, cz, w, l, h,
                                              np.sin(heading), np.cos(heading), vx, vy])
            gt_agent_classes[i] = rng.randint(0, self.config.track.num_classes)
            gt_agent_mask[i] = True

        # Map polylines
        num_valid_polylines = rng.randint(10, self.num_map_polylines + 1)
        gt_map_polylines = torch.zeros(self.num_map_polylines, self.num_points_per_polyline, 2)
        gt_map_classes = torch.zeros(self.num_map_polylines, dtype=torch.long)
        gt_map_mask = torch.zeros(self.num_map_polylines, dtype=torch.bool)

        for i in range(num_valid_polylines):
            # Generate smooth polyline
            start_x = rng.uniform(-30, 30)
            start_y = rng.uniform(-15, 15)
            dx = rng.uniform(-1, 1)
            dy = rng.uniform(-1, 1)
            for p in range(self.num_points_per_polyline):
                gt_map_polylines[i, p, 0] = start_x + dx * p + rng.randn() * 0.1
                gt_map_polylines[i, p, 1] = start_y + dy * p + rng.randn() * 0.1
            gt_map_classes[i] = rng.randint(0, self.config.map.num_classes)
            gt_map_mask[i] = True

        # [FROM PAPER] Agent future trajectories for motion prediction
        gt_agent_futures = torch.zeros(self.num_agents, 12, 2)
        gt_agent_future_mask = torch.zeros(self.num_agents, dtype=torch.bool)

        for i in range(num_valid_agents):
            vx = gt_agent_boxes[i, 8].item()
            vy = gt_agent_boxes[i, 9].item()
            for t in range(12):
                gt_agent_futures[i, t, 0] = gt_agent_boxes[i, 0] + vx * dt * (t + 1)
                gt_agent_futures[i, t, 1] = gt_agent_boxes[i, 1] + vy * dt * (t + 1)
            gt_agent_future_mask[i] = True

        return {
            'images': images,
            'gt_trajectory': gt_trajectory,
            'gt_agent_boxes': gt_agent_boxes,
            'gt_agent_classes': gt_agent_classes,
            'gt_agent_mask': gt_agent_mask,
            'gt_map_polylines': gt_map_polylines,
            'gt_map_classes': gt_map_classes,
            'gt_map_mask': gt_map_mask,
            'gt_agent_futures': gt_agent_futures,
            'gt_agent_future_mask': gt_agent_future_mask,
        }


# =============================================================================
# Loss Functions
# =============================================================================

class PlanningLoss(nn.Module):
    """
    [FROM PAPER] Planning loss from UniAD Section 3.5.

    Combines L2 regression loss on ego trajectory with a collision penalty
    that discourages planned waypoints from overlapping with predicted
    agent future positions.
    """

    def __init__(self, l2_weight: float = 1.0, collision_weight: float = 5.0,
                 collision_threshold: float = 2.0):
        super().__init__()
        self.l2_weight = l2_weight
        self.collision_weight = collision_weight
        # [FROM PAPER] Collision check radius (approx vehicle half-length)
        self.collision_threshold = collision_threshold

    def forward(self, pred_trajectory: torch.Tensor,
                gt_trajectory: torch.Tensor,
                agent_futures: Optional[torch.Tensor] = None,
                agent_future_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            pred_trajectory: (B, T, 2) predicted ego trajectory
            gt_trajectory: (B, T, 2) ground truth ego trajectory
            agent_futures: (B, Na, K, T, 2) predicted agent futures (from MotionFormer)
            agent_future_mask: (B, Na) valid agent mask
        Returns:
            dict with 'total', 'l2', 'collision' losses
        """
        # [FROM PAPER] L2 regression loss on waypoints
        l2_loss = F.mse_loss(pred_trajectory, gt_trajectory)

        # [FROM PAPER] Collision loss: penalize ego waypoints close to agent predictions
        collision_loss = torch.tensor(0.0, device=pred_trajectory.device)

        if agent_futures is not None and agent_future_mask is not None:
            B, T, _ = pred_trajectory.shape
            # Use the best mode (mode 0) of agent predictions
            # agent_futures shape: (B, Na_model, K, T_agent, 2)
            # agent_future_mask shape: (B, Na_gt)
            # [SELF-IMPLEMENTED] Handle mismatch between model queries and GT agents
            Na_gt = agent_future_mask.shape[1]
            agent_best = agent_futures[:, :Na_gt, 0, :T, :]  # (B, Na_gt, T, 2)

            for t in range(T):
                ego_pos = pred_trajectory[:, t:t+1, :]  # (B, 1, 2)
                agent_pos = agent_best[:, :, t, :]  # (B, Na_gt, 2)

                # Distance from ego to each agent at timestep t
                dist = torch.norm(ego_pos - agent_pos, dim=-1)  # (B, Na_gt)

                # [FROM PAPER] Soft collision penalty using sigmoid
                collision_cost = torch.sigmoid(self.collision_threshold - dist)

                # Mask invalid agents
                collision_cost = collision_cost * agent_future_mask.float()

                collision_loss = collision_loss + collision_cost.mean()

            collision_loss = collision_loss / T

        total = self.l2_weight * l2_loss + self.collision_weight * collision_loss

        return {
            'total': total,
            'l2': l2_loss,
            'collision': collision_loss,
        }


class TrackingLoss(nn.Module):
    """
    [FROM PAPER] Detection/Tracking loss combining focal loss for classification
    and L1 loss for bounding box regression.

    In the original UniAD, this uses Hungarian matching to assign predictions
    to ground truth. Here we use a simplified version with direct matching
    for demonstration purposes.
    """

    def __init__(self, cls_weight: float = 2.0, box_weight: float = 0.25,
                 num_classes: int = 10, focal_alpha: float = 0.25,
                 focal_gamma: float = 2.0):
        super().__init__()
        self.cls_weight = cls_weight
        self.box_weight = box_weight
        self.num_classes = num_classes
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    def focal_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        [FROM PAPER] Focal loss for classification (Lin et al., ICCV 2017).
        Used in detection head to handle class imbalance.
        """
        # pred: (N, C+1), target: (N,) with values in [0, C] (C = no-object)
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_term = (1 - pt) ** self.focal_gamma
        loss = self.focal_alpha * focal_term * ce_loss
        return loss.mean()

    def forward(self, pred_classes: torch.Tensor, pred_boxes: torch.Tensor,
                gt_classes: torch.Tensor, gt_boxes: torch.Tensor,
                gt_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            pred_classes: (B, Q, num_classes+1) predicted class logits
            pred_boxes: (B, Q, 10) predicted boxes
            gt_classes: (B, Na) ground truth class labels
            gt_boxes: (B, Na, 10) ground truth boxes
            gt_mask: (B, Na) valid GT mask
        Returns:
            dict with 'total', 'cls', 'box' losses
        """
        B, Q, _ = pred_classes.shape
        _, Na = gt_classes.shape
        device = pred_classes.device

        cls_loss = torch.tensor(0.0, device=device)
        box_loss = torch.tensor(0.0, device=device)

        for b in range(B):
            num_gt = gt_mask[b].sum().int().item()
            if num_gt == 0:
                # [SIMPLIFIED] All predictions should be no-object class
                no_obj_target = torch.full((Q,), self.num_classes,
                                           dtype=torch.long, device=device)
                cls_loss = cls_loss + self.focal_loss(pred_classes[b], no_obj_target)
                continue

            # [SIMPLIFIED] Simple top-k matching instead of Hungarian matching
            # Assign first num_gt predictions to GT, rest are no-object
            target_classes = torch.full((Q,), self.num_classes,
                                        dtype=torch.long, device=device)
            target_classes[:num_gt] = gt_classes[b, :num_gt]

            cls_loss = cls_loss + self.focal_loss(pred_classes[b], target_classes)

            # L1 box loss only for matched predictions
            box_loss = box_loss + F.l1_loss(
                pred_boxes[b, :num_gt], gt_boxes[b, :num_gt])

        cls_loss = cls_loss / B
        box_loss = box_loss / B

        total = self.cls_weight * cls_loss + self.box_weight * box_loss

        return {
            'total': total,
            'cls': cls_loss,
            'box': box_loss,
        }


class MappingLoss(nn.Module):
    """
    [FROM PAPER] Map prediction loss for polyline regression.

    Combines classification loss (which polylines are valid) with
    point-wise regression loss for polyline geometry.
    """

    def __init__(self, cls_weight: float = 2.0, pts_weight: float = 5.0,
                 num_classes: int = 3):
        super().__init__()
        self.cls_weight = cls_weight
        self.pts_weight = pts_weight
        self.num_classes = num_classes

    def forward(self, pred_classes: torch.Tensor, pred_polylines: torch.Tensor,
                gt_classes: torch.Tensor, gt_polylines: torch.Tensor,
                gt_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            pred_classes: (B, Q, num_classes+1) predicted polyline classes
            pred_polylines: (B, Q, num_points, 2) predicted polyline points
            gt_classes: (B, Nm) ground truth polyline classes
            gt_polylines: (B, Nm, num_points, 2) ground truth polyline points
            gt_mask: (B, Nm) valid polyline mask
        Returns:
            dict with 'total', 'cls', 'pts' losses
        """
        B, Q, _ = pred_classes.shape
        _, Nm = gt_classes.shape
        device = pred_classes.device

        cls_loss = torch.tensor(0.0, device=device)
        pts_loss = torch.tensor(0.0, device=device)

        for b in range(B):
            num_gt = gt_mask[b].sum().int().item()
            if num_gt == 0:
                no_obj_target = torch.full((Q,), self.num_classes,
                                           dtype=torch.long, device=device)
                cls_loss = cls_loss + F.cross_entropy(pred_classes[b], no_obj_target)
                continue

            # [SIMPLIFIED] Direct matching (first num_gt predictions to GT)
            num_match = min(num_gt, Q)
            target_classes = torch.full((Q,), self.num_classes,
                                        dtype=torch.long, device=device)
            target_classes[:num_match] = gt_classes[b, :num_match]

            cls_loss = cls_loss + F.cross_entropy(pred_classes[b], target_classes)

            # Point regression loss (L1)
            pts_loss = pts_loss + F.l1_loss(
                pred_polylines[b, :num_match], gt_polylines[b, :num_match])

        cls_loss = cls_loss / B
        pts_loss = pts_loss / B

        total = self.cls_weight * cls_loss + self.pts_weight * pts_loss

        return {
            'total': total,
            'cls': cls_loss,
            'pts': pts_loss,
        }


class MotionLoss(nn.Module):
    """
    [FROM PAPER] Motion prediction loss with best-of-K strategy.

    For multi-modal trajectory prediction, only the best mode (closest to GT)
    is penalized, encouraging the model to learn diverse predictions.
    This is the winner-take-all / min-over-K strategy from the paper.
    """

    def __init__(self, reg_weight: float = 1.0, cls_weight: float = 0.5):
        super().__init__()
        self.reg_weight = reg_weight
        self.cls_weight = cls_weight

    def forward(self, pred_trajectories: torch.Tensor,
                pred_mode_probs: torch.Tensor,
                gt_futures: torch.Tensor,
                gt_future_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            pred_trajectories: (B, Na, K, T, 2) multi-modal predicted trajectories
            pred_mode_probs: (B, Na, K) mode probabilities
            gt_futures: (B, Na, T, 2) ground truth future trajectories
            gt_future_mask: (B, Na) valid agent mask
        Returns:
            dict with 'total', 'reg', 'cls' losses
        """
        B, Na_pred, K, T, _ = pred_trajectories.shape
        _, Na_gt, T_gt, _ = gt_futures.shape
        device = pred_trajectories.device

        # Align dimensions
        Na = min(Na_pred, Na_gt)
        T_min = min(T, T_gt)

        pred_traj = pred_trajectories[:, :Na, :, :T_min, :]  # (B, Na, K, T_min, 2)
        gt_fut = gt_futures[:, :Na, :T_min, :]  # (B, Na, T_min, 2)
        mask = gt_future_mask[:, :Na]  # (B, Na)

        # [FROM PAPER] Best-of-K: compute ADE for each mode, select best
        gt_expanded = gt_fut.unsqueeze(2).expand_as(pred_traj)  # (B, Na, K, T_min, 2)
        displacement = torch.norm(pred_traj - gt_expanded, dim=-1)  # (B, Na, K, T_min)
        ade_per_mode = displacement.mean(dim=-1)  # (B, Na, K)

        # Find best mode per agent
        best_mode_idx = ade_per_mode.argmin(dim=-1)  # (B, Na)

        # Gather best mode trajectories
        best_mode_idx_expanded = best_mode_idx.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        best_mode_idx_expanded = best_mode_idx_expanded.expand(B, Na, 1, T_min, 2)
        best_pred = pred_traj.gather(2, best_mode_idx_expanded).squeeze(2)  # (B, Na, T_min, 2)

        # [FROM PAPER] Regression loss on best mode only (winner-take-all)
        reg_loss = F.smooth_l1_loss(best_pred, gt_fut, reduction='none')  # (B, Na, T_min, 2)
        reg_loss = reg_loss.mean(dim=(-1, -2))  # (B, Na)
        reg_loss = (reg_loss * mask.float()).sum() / (mask.float().sum() + 1e-6)

        # [FROM PAPER] Classification loss: encourage best mode to have high probability
        mode_probs = pred_mode_probs[:, :Na, :]  # (B, Na, K)
        cls_loss = F.cross_entropy(
            mode_probs.reshape(-1, K),
            best_mode_idx.reshape(-1),
            reduction='none'
        ).reshape(B, Na)
        cls_loss = (cls_loss * mask.float()).sum() / (mask.float().sum() + 1e-6)

        total = self.reg_weight * reg_loss + self.cls_weight * cls_loss

        return {
            'total': total,
            'reg': reg_loss,
            'cls': cls_loss,
        }


class UniADLoss(nn.Module):
    """
    [FROM PAPER] Combined multi-task loss for UniAD.

    The total loss is a weighted sum of all task losses:
        L = w_track * L_track + w_map * L_map + w_motion * L_motion + w_plan * L_plan

    Loss weights are from paper Section 4.1 / Table 8 (ablation study).
    """

    def __init__(self, config: Optional[UniADConfig] = None):
        super().__init__()
        self.config = config or UniADConfig()

        # [FROM PAPER] Individual task losses
        self.planning_loss = PlanningLoss(
            l2_weight=self.config.loss_weights['planning_l2'],
            collision_weight=self.config.loss_weights['planning_collision'],
        )
        self.tracking_loss = TrackingLoss(
            num_classes=self.config.track.num_classes,
        )
        self.mapping_loss = MappingLoss(
            num_classes=self.config.map.num_classes,
        )
        self.motion_loss = MotionLoss()

        # [FROM PAPER] Multi-task loss weights from Table 8
        self.task_weights = {
            'tracking': self.config.loss_weights['detection'],
            'mapping': self.config.loss_weights['mapping'],
            'motion': self.config.loss_weights['motion'],
            'planning': self.config.loss_weights['planning_l2'],
        }

    def forward(self, model_output: Dict, batch: Dict) -> Dict[str, torch.Tensor]:
        """
        Compute all task losses given model outputs and ground truth batch.

        Args:
            model_output: dict from UniAD.forward()
            batch: dict from UniADDataset.__getitem__() (batched)
        Returns:
            dict with individual and total losses
        """
        device = model_output['plan']['trajectory'].device
        losses = {}

        # --- Planning Loss ---
        plan_losses = self.planning_loss(
            pred_trajectory=model_output['plan']['trajectory'],
            gt_trajectory=batch['gt_trajectory'].to(device),
            agent_futures=model_output['motion']['predicted_trajectories'],
            agent_future_mask=batch['gt_agent_future_mask'].to(device),
        )
        losses['planning_total'] = plan_losses['total']
        losses['planning_l2'] = plan_losses['l2']
        losses['planning_collision'] = plan_losses['collision']

        # --- Tracking Loss ---
        track_losses = self.tracking_loss(
            pred_classes=model_output['track']['classes'],
            pred_boxes=model_output['track']['boxes'],
            gt_classes=batch['gt_agent_classes'].to(device),
            gt_boxes=batch['gt_agent_boxes'].to(device),
            gt_mask=batch['gt_agent_mask'].to(device),
        )
        losses['tracking_total'] = track_losses['total']
        losses['tracking_cls'] = track_losses['cls']
        losses['tracking_box'] = track_losses['box']

        # --- Mapping Loss ---
        map_losses = self.mapping_loss(
            pred_classes=model_output['map']['classes'],
            pred_polylines=model_output['map']['polylines'],
            gt_classes=batch['gt_map_classes'].to(device),
            gt_polylines=batch['gt_map_polylines'].to(device),
            gt_mask=batch['gt_map_mask'].to(device),
        )
        losses['mapping_total'] = map_losses['total']
        losses['mapping_cls'] = map_losses['cls']
        losses['mapping_pts'] = map_losses['pts']

        # --- Motion Loss ---
        motion_losses = self.motion_loss(
            pred_trajectories=model_output['motion']['predicted_trajectories'],
            pred_mode_probs=model_output['motion']['mode_probs'],
            gt_futures=batch['gt_agent_futures'].to(device),
            gt_future_mask=batch['gt_agent_future_mask'].to(device),
        )
        losses['motion_total'] = motion_losses['total']
        losses['motion_reg'] = motion_losses['reg']
        losses['motion_cls'] = motion_losses['cls']

        # [FROM PAPER] Weighted multi-task total loss
        total_loss = (
            self.task_weights['tracking'] * losses['tracking_total'] +
            self.task_weights['mapping'] * losses['mapping_total'] +
            self.task_weights['motion'] * losses['motion_total'] +
            self.task_weights['planning'] * losses['planning_total']
        )
        losses['total'] = total_loss

        return losses


# =============================================================================
# Training Functions
# =============================================================================

def train_one_epoch(model: nn.Module, dataloader: DataLoader,
                    criterion: UniADLoss, optimizer: torch.optim.Optimizer,
                    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
                    device: torch.device, epoch: int,
                    grad_clip: float = 35.0,
                    use_amp: bool = False) -> Dict[str, float]:
    """
    [SELF-IMPLEMENTED] Train the model for one epoch.

    Args:
        model: UniAD model
        dataloader: training data loader
        criterion: UniADLoss module
        optimizer: optimizer
        scheduler: learning rate scheduler (stepped per iteration)
        device: training device
        epoch: current epoch number
        grad_clip: maximum gradient norm (from paper: 35.0)
        use_amp: whether to use automatic mixed precision

    Returns:
        dict of average losses for the epoch
    """
    model.train()

    # [SELF-IMPLEMENTED] Mixed precision scaler
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp) if use_amp else None

    epoch_losses = {}
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Train]", leave=True)

    for batch_idx, batch in enumerate(pbar):
        images = batch['images'].to(device)  # (B, 6, 3, H, W)

        optimizer.zero_grad()

        # [SELF-IMPLEMENTED] Mixed precision forward pass
        if use_amp:
            with torch.amp.autocast('cuda'):
                output = model(images)
                losses = criterion(output, batch)
        else:
            output = model(images)
            losses = criterion(output, batch)

        total_loss = losses['total']

        # Backward pass
        if use_amp and scaler is not None:
            scaler.scale(total_loss).backward()
            # [FROM PAPER] Gradient clipping with max norm 35.0
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            # [FROM PAPER] Gradient clipping with max norm 35.0
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        # Step scheduler per iteration (cosine annealing)
        if scheduler is not None:
            scheduler.step()

        # Accumulate losses
        for key, val in losses.items():
            if key not in epoch_losses:
                epoch_losses[key] = 0.0
            epoch_losses[key] += val.item()
        num_batches += 1

        # Update progress bar
        pbar.set_postfix({
            'loss': f"{total_loss.item():.4f}",
            'plan_l2': f"{losses['planning_l2'].item():.4f}",
            'lr': f"{optimizer.param_groups[0]['lr']:.2e}",
        })

    # Average losses
    avg_losses = {k: v / num_batches for k, v in epoch_losses.items()}
    return avg_losses


@torch.no_grad()
def validate(model: nn.Module, dataloader: DataLoader,
             criterion: UniADLoss, device: torch.device,
             epoch: int) -> Dict[str, float]:
    """
    [SELF-IMPLEMENTED] Validate the model and compute metrics.

    Computes:
    - Standard losses (same as training)
    - Planning L2 error at 1s, 2s, 3s horizons
    - Collision rate (percentage of timesteps with collision)

    Args:
        model: UniAD model
        dataloader: validation data loader
        criterion: UniADLoss module
        device: validation device
        epoch: current epoch number

    Returns:
        dict of average losses and metrics
    """
    model.eval()

    epoch_losses = {}
    num_batches = 0

    # [FROM PAPER] Planning metrics: L2 error at different horizons
    # At 2Hz: 1s = 2 steps, 2s = 4 steps, 3s = 6 steps
    l2_errors_1s = []
    l2_errors_2s = []
    l2_errors_3s = []
    collision_count = 0
    total_timesteps = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Val]  ", leave=True)

    for batch_idx, batch in enumerate(pbar):
        images = batch['images'].to(device)
        gt_trajectory = batch['gt_trajectory'].to(device)

        output = model(images)
        losses = criterion(output, batch)

        # Accumulate losses
        for key, val in losses.items():
            if key not in epoch_losses:
                epoch_losses[key] = 0.0
            epoch_losses[key] += val.item()
        num_batches += 1

        # [FROM PAPER] Compute planning metrics
        pred_traj = output['plan']['trajectory']  # (B, T, 2)
        B, T, _ = pred_traj.shape

        # L2 displacement error at each timestep
        displacement = torch.norm(pred_traj - gt_trajectory, dim=-1)  # (B, T)

        # L2 at 1s (first 2 steps)
        if T >= 2:
            l2_errors_1s.append(displacement[:, :2].mean().item())
        # L2 at 2s (first 4 steps)
        if T >= 4:
            l2_errors_2s.append(displacement[:, :4].mean().item())
        # L2 at 3s (all 6 steps)
        if T >= 6:
            l2_errors_3s.append(displacement[:, :6].mean().item())

        # [FROM PAPER] Collision rate: check if ego is within threshold of any agent
        agent_futures = batch['gt_agent_futures'].to(device)  # (B, Na, 12, 2)
        agent_mask = batch['gt_agent_future_mask'].to(device)  # (B, Na)
        collision_threshold = 2.0  # meters

        for t in range(min(T, 12)):
            ego_pos = pred_traj[:, t:t+1, :]  # (B, 1, 2)
            agent_pos = agent_futures[:, :, t, :]  # (B, Na, 2)
            dist = torch.norm(ego_pos - agent_pos, dim=-1)  # (B, Na)
            # Check collision (dist < threshold) for valid agents
            is_collision = (dist < collision_threshold) & agent_mask
            collision_count += is_collision.any(dim=-1).sum().item()
            total_timesteps += B

        pbar.set_postfix({'val_loss': f"{losses['total'].item():.4f}"})

    # Compute averages
    avg_losses = {k: v / num_batches for k, v in epoch_losses.items()}

    # Add planning metrics
    metrics = {}
    metrics['l2_1s'] = np.mean(l2_errors_1s) if l2_errors_1s else 0.0
    metrics['l2_2s'] = np.mean(l2_errors_2s) if l2_errors_2s else 0.0
    metrics['l2_3s'] = np.mean(l2_errors_3s) if l2_errors_3s else 0.0
    metrics['collision_rate'] = collision_count / max(total_timesteps, 1) * 100.0

    avg_losses.update(metrics)
    return avg_losses


def print_metrics_table(metrics: Dict[str, float], epoch: int):
    """
    [SELF-IMPLEMENTED] Print a formatted metrics table.
    """
    print("\n" + "=" * 70)
    print(f"  Epoch {epoch+1} Validation Results")
    print("=" * 70)

    # Planning metrics
    print(f"\n  {'Planning Metrics':<30}")
    print(f"  {'-' * 40}")
    print(f"  {'L2 Error @ 1s (m):':<30} {metrics.get('l2_1s', 0.0):.4f}")
    print(f"  {'L2 Error @ 2s (m):':<30} {metrics.get('l2_2s', 0.0):.4f}")
    print(f"  {'L2 Error @ 3s (m):':<30} {metrics.get('l2_3s', 0.0):.4f}")
    print(f"  {'Collision Rate (%):':<30} {metrics.get('collision_rate', 0.0):.2f}")

    # Loss breakdown
    print(f"\n  {'Loss Breakdown':<30}")
    print(f"  {'-' * 40}")
    print(f"  {'Total Loss:':<30} {metrics.get('total', 0.0):.4f}")
    print(f"  {'Planning Loss:':<30} {metrics.get('planning_total', 0.0):.4f}")
    print(f"  {'  - L2:':<30} {metrics.get('planning_l2', 0.0):.4f}")
    print(f"  {'  - Collision:':<30} {metrics.get('planning_collision', 0.0):.4f}")
    print(f"  {'Tracking Loss:':<30} {metrics.get('tracking_total', 0.0):.4f}")
    print(f"  {'Mapping Loss:':<30} {metrics.get('mapping_total', 0.0):.4f}")
    print(f"  {'Motion Loss:':<30} {metrics.get('motion_total', 0.0):.4f}")
    print("=" * 70 + "\n")


# =============================================================================
# Checkpoint Management
# =============================================================================

def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer,
                    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
                    epoch: int, metrics: Dict[str, float],
                    save_path: str, is_best: bool = False):
    """
    [SELF-IMPLEMENTED] Save training checkpoint.

    Saves model weights, optimizer state, scheduler state, and training metrics.
    If is_best=True, also saves a copy as 'best_model.pth'.
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
    }

    # Save latest
    latest_path = os.path.join(save_path, 'latest_checkpoint.pth')
    torch.save(checkpoint, latest_path)
    print(f"  Saved checkpoint: {latest_path}")

    # Save best
    if is_best:
        best_path = os.path.join(save_path, 'best_model.pth')
        torch.save(checkpoint, best_path)
        print(f"  Saved best model: {best_path} (L2@3s = {metrics.get('l2_3s', 0.0):.4f})")

    # [SELF-IMPLEMENTED] Periodic save every 5 epochs
    if (epoch + 1) % 5 == 0:
        epoch_path = os.path.join(save_path, f'checkpoint_epoch_{epoch+1}.pth')
        torch.save(checkpoint, epoch_path)
        print(f"  Saved periodic checkpoint: {epoch_path}")


def load_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer,
                    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
                    checkpoint_path: str, device: torch.device) -> Tuple[int, Dict]:
    """
    [SELF-IMPLEMENTED] Load training checkpoint to resume training.

    Args:
        model: UniAD model
        optimizer: optimizer
        scheduler: learning rate scheduler
        checkpoint_path: path to checkpoint file
        device: device to load to

    Returns:
        (start_epoch, metrics) tuple
    """
    print(f"  Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    start_epoch = checkpoint['epoch'] + 1
    metrics = checkpoint.get('metrics', {})

    print(f"  Resumed from epoch {start_epoch}")
    print(f"  Previous metrics: L2@3s = {metrics.get('l2_3s', 'N/A')}")

    return start_epoch, metrics


# =============================================================================
# Main Training Script
# =============================================================================

def main():
    """
    [SELF-IMPLEMENTED] Main entry point for UniAD training.

    Supports:
    - Quick demo mode (5 epochs, synthetic data)
    - Full training with configurable hyperparameters
    - Resume from checkpoint
    - Mixed precision training on GPU
    - 3-stage training strategy from the paper

    Usage:
        python train.py                          # Quick 5-epoch demo
        python train.py --epochs 24 --lr 2e-4    # Full training
        python train.py --resume checkpoints/latest_checkpoint.pth  # Resume
    """
    parser = argparse.ArgumentParser(
        description='UniAD Training Script (CVPR 2023 Best Paper)')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of training epochs (default: 5 for demo)')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size per device (default: 2)')
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='Learning rate (default: 2e-4 from paper)')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay (default: 0.01)')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device: auto, cuda, cpu (default: auto)')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--num_train_samples', type=int, default=80,
                        help='Number of synthetic training samples (default: 80)')
    parser.add_argument('--num_val_samples', type=int, default=20,
                        help='Number of synthetic validation samples (default: 20)')
    parser.add_argument('--grad_clip', type=float, default=35.0,
                        help='Gradient clipping max norm (default: 35.0 from paper)')
    parser.add_argument('--use_amp', action='store_true',
                        help='Use automatic mixed precision (GPU only)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='DataLoader workers (default: 0 for Windows compat)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Setup
    # -------------------------------------------------------------------------

    # [SELF-IMPLEMENTED] Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Device selection
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    # Disable AMP on CPU
    use_amp = args.use_amp and device.type == 'cuda'

    print("=" * 70)
    print("  UniAD Training Script")
    print("  CVPR 2023 Best Paper - Planning-Oriented Autonomous Driving")
    print("=" * 70)
    print(f"\n  Device:          {device}")
    print(f"  Epochs:          {args.epochs}")
    print(f"  Batch size:      {args.batch_size}")
    print(f"  Learning rate:   {args.lr}")
    print(f"  Weight decay:    {args.weight_decay}")
    print(f"  Grad clip:       {args.grad_clip}")
    print(f"  Mixed precision: {use_amp}")
    print(f"  Save directory:  {args.save_dir}")
    print(f"  Resume from:     {args.resume or 'None'}")
    print()

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # Model, Dataset, Criterion
    # -------------------------------------------------------------------------

    # [FROM PAPER] Model configuration
    config = UniADConfig()
    model = UniAD(config).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters:     {num_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print()

    # [SELF-IMPLEMENTED] Synthetic datasets
    train_dataset = UniADDataset(num_samples=args.num_train_samples, config=config)
    val_dataset = UniADDataset(num_samples=args.num_val_samples, config=config)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
    )

    print(f"  Training samples:   {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print(f"  Train batches/epoch: {len(train_loader)}")
    print(f"  Val batches/epoch:   {len(val_loader)}")
    print()

    # [FROM PAPER] Multi-task loss with paper-specified weights
    criterion = UniADLoss(config)

    # [FROM PAPER] AdamW optimizer (from paper Section 4.1)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )

    # [FROM PAPER] Cosine annealing learning rate schedule
    total_steps = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=args.lr * 0.01,  # minimum LR = 1% of initial
    )

    # -------------------------------------------------------------------------
    # Resume from checkpoint
    # -------------------------------------------------------------------------

    start_epoch = 0
    best_l2_3s = float('inf')

    if args.resume:
        if os.path.isfile(args.resume):
            start_epoch, prev_metrics = load_checkpoint(
                model, optimizer, scheduler, args.resume, device)
            best_l2_3s = prev_metrics.get('l2_3s', float('inf'))
        else:
            print(f"  WARNING: Checkpoint not found at {args.resume}, starting from scratch.")

    # -------------------------------------------------------------------------
    # Training Loop
    # -------------------------------------------------------------------------

    print("\n" + "-" * 70)
    print("  Starting Training")
    print("-" * 70 + "\n")

    # [FROM PAPER] 3-stage training strategy note:
    # Stage 1: Perception (detection + tracking + mapping) - frozen motion/planning
    # Stage 2: Motion prediction - frozen perception, unfrozen motion
    # Stage 3: End-to-end - all modules unfrozen, planning fine-tuning
    # [SIMPLIFIED] For this demo, we train all modules jointly from the start.
    print("  NOTE: Original UniAD uses 3-stage training (perception -> motion -> planning)")
    print("  This demo trains all modules jointly for simplicity.\n")

    training_start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()

        # Train
        train_losses = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epoch=epoch,
            grad_clip=args.grad_clip,
            use_amp=use_amp,
        )

        # Validate
        val_metrics = validate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
        )

        epoch_time = time.time() - epoch_start_time

        # Print metrics table
        print_metrics_table(val_metrics, epoch)
        print(f"  Epoch time: {epoch_time:.1f}s | "
              f"Train loss: {train_losses['total']:.4f} | "
              f"Val loss: {val_metrics['total']:.4f}")

        # Check if best model
        current_l2_3s = val_metrics.get('l2_3s', float('inf'))
        is_best = current_l2_3s < best_l2_3s
        if is_best:
            best_l2_3s = current_l2_3s
            print(f"  ** New best model! L2@3s = {best_l2_3s:.4f} **")

        # Save checkpoint
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            metrics=val_metrics,
            save_path=args.save_dir,
            is_best=is_best,
        )

        print()

    # -------------------------------------------------------------------------
    # Training Complete
    # -------------------------------------------------------------------------

    total_time = time.time() - training_start_time
    print("\n" + "=" * 70)
    print("  Training Complete!")
    print("=" * 70)
    print(f"  Total training time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Best L2@3s:          {best_l2_3s:.4f} m")
    print(f"  Checkpoints saved:   {args.save_dir}/")
    print(f"  Best model:          {args.save_dir}/best_model.pth")
    print("=" * 70)


if __name__ == '__main__':
    main()
