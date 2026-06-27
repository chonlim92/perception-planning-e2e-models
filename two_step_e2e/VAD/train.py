"""
VAD Training Script
===================
ATTRIBUTION:
- Loss functions: Based on VAD paper (Jiang et al., ICCV 2023)
  - Planning loss with K ego queries and winner-take-all from Section 3.3
  - Vectorized scene loss (agent + map) from Section 3.2
  - Score supervision with soft target from Section 3.3
- Training strategy: Joint end-to-end training from paper
- Implementation: Self-implemented in PyTorch (simplified from official VAD codebase)
- Synthetic dataset: Self-implemented for demonstration (real training uses nuScenes)
"""

import argparse
import math
import os
import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

from model import VAD


# =============================================================================
# Synthetic Dataset
# =============================================================================

class VADDataset(Dataset):  # [SELF-IMPLEMENTED]
    """
    Synthetic dataset for VAD training demonstration.
    Real VAD training uses nuScenes with 3D annotations.

    Generates:
        - Multi-view images: (6, 3, 224, 400) from 6 camera views
        - GT trajectory: (6, 2) ego-vehicle future waypoints at 0.5s intervals (3s total)
        - GT agent positions: (num_agents, 12, 2) neighboring agent future trajectories
        - GT map vectors: (num_map_elements, 20, 2) polyline points for map elements
    """

    def __init__(self, num_samples: int = 1000, num_agents: int = 20,
                 num_map_elements: int = 30, num_waypoints: int = 6,
                 future_steps: int = 12, num_points_per_polyline: int = 20,
                 seed: int = 42):
        super().__init__()
        self.num_samples = num_samples
        self.num_agents = num_agents
        self.num_map_elements = num_map_elements
        self.num_waypoints = num_waypoints
        self.future_steps = future_steps
        self.num_points_per_polyline = num_points_per_polyline

        # [SELF-IMPLEMENTED] Generate synthetic data with fixed seed for reproducibility
        rng = np.random.RandomState(seed)

        # Multi-view images (stored as float16 to save memory)
        self.images = rng.randn(
            num_samples, 6, 3, 224, 400).astype(np.float32) * 0.1

        # [SELF-IMPLEMENTED] Generate plausible ego trajectories (smooth curves)
        self.trajectories = np.zeros((num_samples, num_waypoints, 2), dtype=np.float32)
        for i in range(num_samples):
            # Simulate forward driving with slight curvature
            speed = rng.uniform(2.0, 8.0)  # m/s
            curvature = rng.uniform(-0.05, 0.05)  # rad/step
            heading = 0.0
            x, y = 0.0, 0.0
            for t in range(num_waypoints):
                heading += curvature
                x += speed * 0.5 * np.cos(heading)  # 0.5s interval
                y += speed * 0.5 * np.sin(heading)
                self.trajectories[i, t] = [x, y]

        # [SELF-IMPLEMENTED] Generate agent trajectories
        self.agent_positions = rng.randn(
            num_samples, num_agents, future_steps, 2).astype(np.float32) * 5.0

        # [SELF-IMPLEMENTED] Generate map polylines (lane boundaries, crosswalks, dividers)
        self.map_vectors = np.zeros(
            (num_samples, num_map_elements, num_points_per_polyline, 2), dtype=np.float32)
        for i in range(num_samples):
            for j in range(num_map_elements):
                # Generate smooth polylines
                start = rng.randn(2) * 20.0
                direction = rng.randn(2)
                direction = direction / (np.linalg.norm(direction) + 1e-6)
                for k in range(num_points_per_polyline):
                    self.map_vectors[i, j, k] = start + direction * k * 1.5
                    # Add slight noise for realism
                    self.map_vectors[i, j, k] += rng.randn(2) * 0.1

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'images': torch.from_numpy(self.images[idx]),           # (6, 3, 224, 400)
            'trajectory': torch.from_numpy(self.trajectories[idx]), # (6, 2)
            'agent_positions': torch.from_numpy(self.agent_positions[idx]),  # (Na, 12, 2)
            'map_vectors': torch.from_numpy(self.map_vectors[idx]),          # (Nm, 20, 2)
        }


# =============================================================================
# Loss Functions
# =============================================================================

class VADPlanningLoss(nn.Module):  # [FROM PAPER]
    """
    VAD Planning Loss from Section 3.3:
    - K ego queries each produce a candidate trajectory
    - Winner-take-all: only the closest trajectory to GT is supervised
    - Score head is supervised with soft targets based on distance to GT

    L_plan = L_reg(best_k) + lambda_score * L_score
    """

    def __init__(self, score_weight: float = 0.5, temperature: float = 1.0):
        super().__init__()
        self.score_weight = score_weight  # [FROM PAPER] lambda_score
        self.temperature = temperature    # [FROM PAPER] temperature for soft targets

    def forward(self, trajectories: torch.Tensor, scores: torch.Tensor,
                gt_trajectory: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            trajectories: (B, K, T, 2) K candidate trajectories
            scores: (B, K) predicted quality scores (logits)
            gt_trajectory: (B, T, 2) ground truth ego trajectory
        Returns:
            dict with planning_reg_loss, score_loss, total
        """
        B, K, T, _ = trajectories.shape

        # [FROM PAPER] Compute L2 distance from each candidate to GT
        gt_expanded = gt_trajectory.unsqueeze(1).expand_as(trajectories)  # (B, K, T, 2)
        distances = torch.norm(trajectories - gt_expanded, dim=-1)  # (B, K, T)
        avg_distances = distances.mean(dim=-1)  # (B, K) average over timesteps

        # [FROM PAPER] Winner-take-all: select closest trajectory
        best_k = avg_distances.argmin(dim=-1)  # (B,)

        # [FROM PAPER] Regression loss on winner trajectory only
        reg_loss = torch.zeros(B, device=trajectories.device)
        for b in range(B):
            reg_loss[b] = F.smooth_l1_loss(
                trajectories[b, best_k[b]], gt_trajectory[b])
        reg_loss = reg_loss.mean()

        # [FROM PAPER] Score supervision with soft targets
        # Soft target: trajectories closer to GT get higher target scores
        with torch.no_grad():
            soft_targets = F.softmax(-avg_distances / self.temperature, dim=-1)  # (B, K)

        # [SIMPLIFIED] Cross-entropy between predicted score distribution and soft targets
        log_pred_scores = F.log_softmax(scores, dim=-1)
        score_loss = F.kl_div(log_pred_scores, soft_targets, reduction='batchmean')

        total = reg_loss + self.score_weight * score_loss

        return {
            'planning_reg_loss': reg_loss,
            'score_loss': score_loss,
            'total': total,
            'best_k': best_k,
        }


class AgentLoss(nn.Module):  # [FROM PAPER]
    """
    Vectorized Agent Loss from Section 3.2:
    - Multi-modal motion prediction loss
    - Classification loss for agent type
    - Winner-take-all across modes for each agent

    Note: In the full implementation, Hungarian matching assigns predictions
    to GT agents. Here we use simplified direct supervision.
    """

    def __init__(self, cls_weight: float = 1.0, reg_weight: float = 1.0):
        super().__init__()
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight

    def forward(self, motion_vectors: torch.Tensor, mode_probs: torch.Tensor,
                gt_agent_positions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            motion_vectors: (B, Q, K_modes, T, 2) predicted agent motions
            mode_probs: (B, Q, K_modes) mode probabilities
            gt_agent_positions: (B, Na, T, 2) ground truth agent trajectories
        Returns:
            dict with agent motion loss
        """
        B, Q, K_modes, T, _ = motion_vectors.shape
        Na = gt_agent_positions.shape[1]

        # [SIMPLIFIED] Use first Na queries to match Na GT agents (simplified from Hungarian)
        num_matched = min(Q, Na)
        pred_matched = motion_vectors[:, :num_matched]  # (B, Na, K, T, 2)
        gt_matched = gt_agent_positions[:, :num_matched]  # (B, Na, T, 2)

        # [FROM PAPER] Winner-take-all across modes per agent
        gt_expanded = gt_matched.unsqueeze(2).expand_as(pred_matched)  # (B, Na, K, T, 2)
        mode_distances = torch.norm(
            pred_matched - gt_expanded, dim=-1).mean(dim=-1)  # (B, Na, K)
        best_mode = mode_distances.argmin(dim=-1)  # (B, Na)

        # [FROM PAPER] Regression loss on best mode
        reg_loss = torch.zeros(B, device=motion_vectors.device)
        for b in range(B):
            for a in range(num_matched):
                reg_loss[b] += F.smooth_l1_loss(
                    pred_matched[b, a, best_mode[b, a]],
                    gt_matched[b, a])
        reg_loss = reg_loss.mean() / max(num_matched, 1)

        # [FROM PAPER] Mode probability loss: encourage correct mode to have high prob
        mode_target = torch.zeros(B, num_matched, K_modes, device=motion_vectors.device)
        for b in range(B):
            for a in range(num_matched):
                mode_target[b, a, best_mode[b, a]] = 1.0
        mode_loss = F.cross_entropy(
            mode_probs[:, :num_matched].reshape(-1, K_modes),
            mode_target.reshape(-1, K_modes).argmax(dim=-1))

        total = self.reg_weight * reg_loss + self.cls_weight * mode_loss

        return {
            'agent_reg_loss': reg_loss,
            'agent_mode_loss': mode_loss,
            'agent_total': total,
        }


class MapLoss(nn.Module):  # [FROM PAPER]
    """
    Vectorized Map Loss from Section 3.2:
    - Polyline regression loss
    - Map element classification loss

    Note: Full implementation uses Hungarian matching for polyline assignment.
    Here we use simplified direct correspondence.
    """

    def __init__(self, cls_weight: float = 1.0, reg_weight: float = 5.0):
        super().__init__()
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight  # [FROM PAPER] higher weight on map regression

    def forward(self, polylines: torch.Tensor,
                gt_map_vectors: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            polylines: (B, Q_map, num_points, 2) predicted map polylines
            gt_map_vectors: (B, Nm, num_points, 2) ground truth map polylines
        Returns:
            dict with map regression loss
        """
        B, Q_map, P, _ = polylines.shape
        Nm = gt_map_vectors.shape[1]

        # [SIMPLIFIED] Match first Nm queries to GT (simplified from Hungarian matching)
        num_matched = min(Q_map, Nm)
        pred_matched = polylines[:, :num_matched]  # (B, Nm, P, 2)
        gt_matched = gt_map_vectors[:, :num_matched]  # (B, Nm, P, 2)

        # [FROM PAPER] Polyline regression loss (Chamfer-like, simplified to L1)
        reg_loss = F.smooth_l1_loss(pred_matched, gt_matched)

        return {
            'map_reg_loss': reg_loss,
            'map_total': self.reg_weight * reg_loss,
        }


class VADLoss(nn.Module):  # [FROM PAPER]
    """
    Combined VAD loss with paper-specified weighting:
    L_total = L_plan + lambda_agent * L_agent + lambda_map * L_map

    From VAD paper Section 3.4, the weighting balances perception and planning.
    """

    def __init__(self, plan_weight: float = 1.0, agent_weight: float = 0.5,
                 map_weight: float = 0.5, score_weight: float = 0.5):
        super().__init__()
        self.plan_weight = plan_weight    # [FROM PAPER]
        self.agent_weight = agent_weight  # [FROM PAPER]
        self.map_weight = map_weight      # [FROM PAPER]

        self.planning_loss = VADPlanningLoss(score_weight=score_weight)
        self.agent_loss = AgentLoss()
        self.map_loss = MapLoss()

    def forward(self, outputs: Dict, gt_trajectory: torch.Tensor,
                gt_agent_positions: torch.Tensor,
                gt_map_vectors: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            outputs: model forward outputs
            gt_trajectory: (B, T, 2) ego trajectory GT
            gt_agent_positions: (B, Na, T_agent, 2) agent trajectory GT
            gt_map_vectors: (B, Nm, P, 2) map polyline GT
        Returns:
            dict with all loss terms and total
        """
        plan_output = outputs['plan']
        agent_output = outputs['agents']
        map_output = outputs['map']

        # [FROM PAPER] Planning loss (winner-take-all + score supervision)
        plan_losses = self.planning_loss(
            plan_output['trajectories'],
            plan_output['scores'],
            gt_trajectory)

        # [FROM PAPER] Agent motion prediction loss
        agent_losses = self.agent_loss(
            agent_output['motion_vectors'],
            agent_output['mode_probs'],
            gt_agent_positions)

        # [FROM PAPER] Map reconstruction loss
        map_losses = self.map_loss(
            map_output['polylines'],
            gt_map_vectors)

        # [FROM PAPER] Combined loss with weighting
        total_loss = (self.plan_weight * plan_losses['total'] +
                      self.agent_weight * agent_losses['agent_total'] +
                      self.map_weight * map_losses['map_total'])

        return {
            'total': total_loss,
            'planning_reg': plan_losses['planning_reg_loss'],
            'planning_score': plan_losses['score_loss'],
            'agent_reg': agent_losses['agent_reg_loss'],
            'agent_mode': agent_losses['agent_mode_loss'],
            'map_reg': map_losses['map_reg_loss'],
        }


# =============================================================================
# Validation Metrics
# =============================================================================

class VADMetrics:  # [FROM PAPER] + [SELF-IMPLEMENTED]
    """
    Compute VAD evaluation metrics:
    - Planning L2 error at 1s, 2s, 3s (paper Table 1)
    - Collision rate (simplified; paper uses full geometry check)
    - Score accuracy (whether highest-scored trajectory is best)
    """

    def __init__(self, dt: float = 0.5, vehicle_length: float = 4.5,
                 vehicle_width: float = 2.0):
        self.dt = dt  # [FROM PAPER] time step between waypoints
        self.vehicle_length = vehicle_length  # [SELF-IMPLEMENTED]
        self.vehicle_width = vehicle_width    # [SELF-IMPLEMENTED]
        self.reset()

    def reset(self):
        self.l2_errors_per_step = []  # list of (B, T) tensors
        self.collision_counts = 0
        self.total_samples = 0
        self.score_correct = 0
        self.score_total = 0

    @torch.no_grad()
    def update(self, outputs: Dict, gt_trajectory: torch.Tensor,
               gt_agent_positions: torch.Tensor):
        """
        Update metrics with a batch of predictions.

        Args:
            outputs: model outputs
            gt_trajectory: (B, T, 2)
            gt_agent_positions: (B, Na, T_agent, 2)
        """
        plan = outputs['plan']
        best_traj = plan['best_trajectory']  # (B, T, 2)
        trajectories = plan['trajectories']  # (B, K, T, 2)
        scores = plan['scores']              # (B, K)

        B, T, _ = best_traj.shape

        # [FROM PAPER] L2 error per timestep
        l2_per_step = torch.norm(best_traj - gt_trajectory, dim=-1)  # (B, T)
        self.l2_errors_per_step.append(l2_per_step.cpu())

        # [SELF-IMPLEMENTED] Collision rate (simplified: check if ego trajectory
        # comes within threshold distance of any agent)
        collision_threshold = (self.vehicle_length + self.vehicle_width) / 2.0
        Na = gt_agent_positions.shape[1]
        T_agent = gt_agent_positions.shape[2]
        T_check = min(T, T_agent)

        for b in range(B):
            ego_pos = best_traj[b, :T_check]  # (T_check, 2)
            agent_pos = gt_agent_positions[b, :, :T_check]  # (Na, T_check, 2)
            # Distance from ego to each agent at each timestep
            dist = torch.norm(
                ego_pos.unsqueeze(0) - agent_pos, dim=-1)  # (Na, T_check)
            if (dist < collision_threshold).any():
                self.collision_counts += 1

        self.total_samples += B

        # [SELF-IMPLEMENTED] Score accuracy: check if highest-scored trajectory
        # is actually closest to GT
        gt_expanded = gt_trajectory.unsqueeze(1).expand_as(trajectories)
        distances = torch.norm(trajectories - gt_expanded, dim=-1).mean(dim=-1)  # (B, K)
        actual_best = distances.argmin(dim=-1)  # (B,)
        predicted_best = scores.argmax(dim=-1)  # (B,)
        self.score_correct += (actual_best == predicted_best).sum().item()
        self.score_total += B

    def compute(self) -> Dict[str, float]:
        """
        Compute final metrics.

        Returns:
            dict with:
                - l2_1s, l2_2s, l2_3s: Planning L2 at 1s, 2s, 3s
                - collision_rate: fraction of samples with collision
                - score_accuracy: fraction where top-scored traj is closest to GT
        """
        if not self.l2_errors_per_step:
            return {'l2_1s': 0.0, 'l2_2s': 0.0, 'l2_3s': 0.0,
                    'collision_rate': 0.0, 'score_accuracy': 0.0}

        # [FROM PAPER] Concatenate all L2 errors
        all_l2 = torch.cat(self.l2_errors_per_step, dim=0)  # (N_total, T)
        T = all_l2.shape[1]

        # [FROM PAPER] L2 at 1s, 2s, 3s (waypoints at 0.5s intervals)
        # 1s = index 1 (0-indexed), 2s = index 3, 3s = index 5
        idx_1s = min(1, T - 1)  # timestep at 1.0s
        idx_2s = min(3, T - 1)  # timestep at 2.0s
        idx_3s = min(5, T - 1)  # timestep at 3.0s

        l2_1s = all_l2[:, idx_1s].mean().item()
        l2_2s = all_l2[:, idx_2s].mean().item()
        l2_3s = all_l2[:, idx_3s].mean().item()

        # [SELF-IMPLEMENTED] Collision and score metrics
        collision_rate = self.collision_counts / max(self.total_samples, 1)
        score_accuracy = self.score_correct / max(self.score_total, 1)

        return {
            'l2_1s': l2_1s,
            'l2_2s': l2_2s,
            'l2_3s': l2_3s,
            'collision_rate': collision_rate,
            'score_accuracy': score_accuracy,
        }


# =============================================================================
# Training Loop
# =============================================================================

def train_one_epoch(model: nn.Module, dataloader: DataLoader,
                    criterion: VADLoss, optimizer: torch.optim.Optimizer,
                    scaler: torch.amp.GradScaler, device: torch.device,
                    epoch: int, max_grad_norm: float = 5.0,
                    use_amp: bool = True, scheduler=None) -> Dict[str, float]:  # [SELF-IMPLEMENTED]
    """
    Train VAD for one epoch.

    Args:
        model: VAD model
        dataloader: training DataLoader
        criterion: VADLoss
        optimizer: optimizer
        scaler: GradScaler for mixed precision
        device: training device
        epoch: current epoch number
        max_grad_norm: gradient clipping threshold
        use_amp: whether to use automatic mixed precision

    Returns:
        dict with average loss values for the epoch
    """
    model.train()
    epoch_losses = {
        'total': 0.0, 'planning_reg': 0.0, 'planning_score': 0.0,
        'agent_reg': 0.0, 'agent_mode': 0.0, 'map_reg': 0.0,
    }
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Train]", leave=True)
    for batch in pbar:
        # Move data to device
        images = batch['images'].to(device)              # (B, 6, 3, 224, 400)
        gt_trajectory = batch['trajectory'].to(device)    # (B, 6, 2)
        gt_agents = batch['agent_positions'].to(device)   # (B, Na, 12, 2)
        gt_map = batch['map_vectors'].to(device)          # (B, Nm, 20, 2)

        optimizer.zero_grad()

        # [SELF-IMPLEMENTED] Mixed precision forward pass
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(images)
            losses = criterion(outputs, gt_trajectory, gt_agents, gt_map)

        # [SELF-IMPLEMENTED] Backward pass with gradient scaling
        if use_amp:
            scaler.scale(losses['total']).backward()
            scaler.unscale_(optimizer)
            # [FROM PAPER] Gradient clipping for stable training
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            losses['total'].backward()
            # [FROM PAPER] Gradient clipping for stable training
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        # Step LR scheduler per iteration (not per epoch)
        if scheduler is not None:
            scheduler.step()

        # Accumulate losses
        for key in epoch_losses:
            epoch_losses[key] += losses[key].item()
        num_batches += 1

        # Update progress bar
        pbar.set_postfix({
            'loss': f"{losses['total'].item():.4f}",
            'plan': f"{losses['planning_reg'].item():.4f}",
            'score': f"{losses['planning_score'].item():.4f}",
        })

    # Average losses
    for key in epoch_losses:
        epoch_losses[key] /= max(num_batches, 1)

    return epoch_losses


@torch.no_grad()
def validate(model: nn.Module, dataloader: DataLoader,
             criterion: VADLoss, device: torch.device,
             use_amp: bool = True) -> Tuple[Dict[str, float], Dict[str, float]]:  # [SELF-IMPLEMENTED]
    """
    Validate VAD model.

    Args:
        model: VAD model
        dataloader: validation DataLoader
        criterion: VADLoss
        device: device
        use_amp: whether to use AMP

    Returns:
        Tuple of (loss_dict, metrics_dict)
    """
    model.eval()
    metrics = VADMetrics()

    epoch_losses = {
        'total': 0.0, 'planning_reg': 0.0, 'planning_score': 0.0,
        'agent_reg': 0.0, 'agent_mode': 0.0, 'map_reg': 0.0,
    }
    num_batches = 0

    pbar = tqdm(dataloader, desc="[Validate]", leave=True)
    for batch in pbar:
        images = batch['images'].to(device)
        gt_trajectory = batch['trajectory'].to(device)
        gt_agents = batch['agent_positions'].to(device)
        gt_map = batch['map_vectors'].to(device)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(images)
            losses = criterion(outputs, gt_trajectory, gt_agents, gt_map)

        # Accumulate losses
        for key in epoch_losses:
            epoch_losses[key] += losses[key].item()
        num_batches += 1

        # [FROM PAPER] Update planning metrics (L2 at 1s/2s/3s, collision rate)
        metrics.update(outputs, gt_trajectory, gt_agents)

        pbar.set_postfix({'loss': f"{losses['total'].item():.4f}"})

    # Average losses
    for key in epoch_losses:
        epoch_losses[key] /= max(num_batches, 1)

    metrics_dict = metrics.compute()
    return epoch_losses, metrics_dict


# =============================================================================
# Checkpoint Management
# =============================================================================

def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer,
                    scheduler, scaler: torch.amp.GradScaler,
                    epoch: int, best_metric: float,
                    save_path: str, is_best: bool = False):  # [SELF-IMPLEMENTED]
    """
    Save training checkpoint.

    Args:
        model: VAD model
        optimizer: optimizer state
        scheduler: LR scheduler state
        scaler: GradScaler state
        epoch: current epoch
        best_metric: best validation metric so far
        save_path: directory to save checkpoints
        is_best: whether this is the best model so far
    """
    os.makedirs(save_path, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'scaler_state_dict': scaler.state_dict(),
        'best_metric': best_metric,
    }

    # Save latest checkpoint
    latest_path = os.path.join(save_path, 'checkpoint_latest.pth')
    torch.save(checkpoint, latest_path)
    print(f"  Saved latest checkpoint: {latest_path}")

    # Save best checkpoint
    if is_best:
        best_path = os.path.join(save_path, 'checkpoint_best.pth')
        torch.save(checkpoint, best_path)
        print(f"  Saved best checkpoint: {best_path}")

    # Save periodic checkpoint every 10 epochs
    if (epoch + 1) % 10 == 0:
        periodic_path = os.path.join(save_path, f'checkpoint_epoch_{epoch+1}.pth')
        torch.save(checkpoint, periodic_path)
        print(f"  Saved periodic checkpoint: {periodic_path}")


def load_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer,
                    scheduler, scaler: torch.amp.GradScaler,
                    checkpoint_path: str, device: torch.device
                    ) -> Tuple[int, float]:  # [SELF-IMPLEMENTED]
    """
    Load training checkpoint for resume.

    Args:
        model: VAD model
        optimizer: optimizer
        scheduler: LR scheduler
        scaler: GradScaler
        checkpoint_path: path to checkpoint file
        device: device to load onto

    Returns:
        Tuple of (start_epoch, best_metric)
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])

    start_epoch = checkpoint['epoch'] + 1
    best_metric = checkpoint['best_metric']

    print(f"  Resumed from epoch {start_epoch}, best L2@3s: {best_metric:.4f}")
    return start_epoch, best_metric


# =============================================================================
# Main Training Script
# =============================================================================

def get_cosine_schedule_with_warmup(optimizer: torch.optim.Optimizer,
                                     num_warmup_steps: int,
                                     num_training_steps: int,
                                     min_lr_ratio: float = 0.01):  # [SELF-IMPLEMENTED]
    """
    Cosine learning rate schedule with linear warmup.
    VAD uses cosine annealing as common in transformer-based models.
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, num_warmup_steps))
        # Cosine annealing
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main():
    # [SELF-IMPLEMENTED] Argument parsing
    parser = argparse.ArgumentParser(
        description='VAD Training Script - Vectorized Autonomous Driving (ICCV 2023)')

    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=24,
                        help='Number of training epochs (default: 24)')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size per GPU (default: 4)')
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='Peak learning rate (default: 2e-4)')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay (default: 0.01)')
    parser.add_argument('--max_grad_norm', type=float, default=5.0,
                        help='Gradient clipping norm (default: 5.0)')

    # Model configuration
    parser.add_argument('--embed_dim', type=int, default=256,
                        help='Embedding dimension (default: 256)')
    parser.add_argument('--num_ego_queries', type=int, default=6,
                        help='Number of ego planning queries K (default: 6)')
    parser.add_argument('--num_waypoints', type=int, default=6,
                        help='Number of future waypoints T (default: 6)')

    # Loss weights
    parser.add_argument('--plan_weight', type=float, default=1.0,
                        help='Planning loss weight (default: 1.0)')
    parser.add_argument('--agent_weight', type=float, default=0.5,
                        help='Agent loss weight (default: 0.5)')
    parser.add_argument('--map_weight', type=float, default=0.5,
                        help='Map loss weight (default: 0.5)')

    # Dataset
    parser.add_argument('--train_samples', type=int, default=800,
                        help='Number of training samples (default: 800)')
    parser.add_argument('--val_samples', type=int, default=200,
                        help='Number of validation samples (default: 200)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='DataLoader workers (default: 0)')

    # Device and precision
    parser.add_argument('--device', type=str, default='auto',
                        help='Device: auto, cuda, cpu (default: auto)')
    parser.add_argument('--amp', action='store_true', default=True,
                        help='Use automatic mixed precision (default: True)')
    parser.add_argument('--no_amp', action='store_true',
                        help='Disable automatic mixed precision')

    # Checkpoint
    parser.add_argument('--save_dir', type=str, default='./checkpoints/vad',
                        help='Directory to save checkpoints (default: ./checkpoints/vad)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')

    # Logging
    parser.add_argument('--val_interval', type=int, default=2,
                        help='Validate every N epochs (default: 2)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')

    args = parser.parse_args()

    # Resolve AMP flag
    use_amp = args.amp and not args.no_amp

    # [SELF-IMPLEMENTED] Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # [SELF-IMPLEMENTED] Device selection
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    # Disable AMP on CPU (not supported well)
    if device.type == 'cpu':
        use_amp = False

    print("=" * 70)
    print("VAD: Vectorized Autonomous Driving - Training Script")
    print("Paper: Jiang et al., ICCV 2023")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Device:          {device}")
    print(f"  Mixed Precision: {use_amp}")
    print(f"  Epochs:          {args.epochs}")
    print(f"  Batch Size:      {args.batch_size}")
    print(f"  Learning Rate:   {args.lr}")
    print(f"  Embed Dim:       {args.embed_dim}")
    print(f"  Ego Queries (K): {args.num_ego_queries}")
    print(f"  Waypoints (T):   {args.num_waypoints}")
    print(f"  Save Dir:        {args.save_dir}")
    print()

    # =========================================================================
    # Dataset and DataLoader
    # =========================================================================
    print("Creating synthetic datasets...")  # [SELF-IMPLEMENTED]
    train_dataset = VADDataset(
        num_samples=args.train_samples, seed=args.seed)
    val_dataset = VADDataset(
        num_samples=args.val_samples, seed=args.seed + 1)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == 'cuda'),
        drop_last=True)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == 'cuda'))

    print(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  Val:   {len(val_dataset)} samples, {len(val_loader)} batches")
    print()

    # =========================================================================
    # Model
    # =========================================================================
    print("Building VAD model...")
    model = VAD(
        embed_dim=args.embed_dim,
        num_ego_queries=args.num_ego_queries,
        num_waypoints=args.num_waypoints,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters:     {num_params:,}")
    print(f"  Trainable parameters: {num_trainable:,}")
    print()

    # =========================================================================
    # Loss, Optimizer, Scheduler
    # =========================================================================
    # [FROM PAPER] Combined loss with paper weighting
    criterion = VADLoss(
        plan_weight=args.plan_weight,
        agent_weight=args.agent_weight,
        map_weight=args.map_weight,
    )

    # [FROM PAPER] AdamW optimizer (standard for transformers)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # [SELF-IMPLEMENTED] Cosine schedule with warmup
    num_training_steps = args.epochs * len(train_loader)
    num_warmup_steps = int(0.1 * num_training_steps)  # 10% warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps, num_training_steps)

    # [SELF-IMPLEMENTED] Mixed precision gradient scaler
    scaler = torch.amp.GradScaler(enabled=use_amp)

    # =========================================================================
    # Resume from checkpoint
    # =========================================================================
    start_epoch = 0
    best_l2_3s = float('inf')

    if args.resume:
        if os.path.isfile(args.resume):
            start_epoch, best_l2_3s = load_checkpoint(
                model, optimizer, scheduler, scaler, args.resume, device)
        else:
            print(f"  WARNING: Checkpoint not found: {args.resume}")
            print(f"  Starting from scratch.")
    print()

    # =========================================================================
    # Training Loop
    # =========================================================================
    print("Starting training...")
    print("-" * 70)

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()

        # Train one epoch
        train_losses = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler,
            device, epoch, max_grad_norm=args.max_grad_norm, use_amp=use_amp,
            scheduler=scheduler)

        # Scheduler stepping is handled inside train_one_epoch (per-iteration)
        # No additional stepping needed here

        epoch_time = time.time() - epoch_start

        # Print training summary
        print(f"\n  Epoch {epoch+1}/{args.epochs} "
              f"({epoch_time:.1f}s) - LR: {optimizer.param_groups[0]['lr']:.2e}")
        print(f"    Train Loss: {train_losses['total']:.4f} | "
              f"Plan: {train_losses['planning_reg']:.4f} | "
              f"Score: {train_losses['planning_score']:.4f} | "
              f"Agent: {train_losses['agent_reg']:.4f} | "
              f"Map: {train_losses['map_reg']:.4f}")

        # Validate periodically
        is_best = False
        if (epoch + 1) % args.val_interval == 0 or epoch == args.epochs - 1:
            val_losses, val_metrics = validate(
                model, val_loader, criterion, device, use_amp=use_amp)

            # [FROM PAPER] Primary metric: L2 error at 3s
            current_l2_3s = val_metrics['l2_3s']
            is_best = current_l2_3s < best_l2_3s
            if is_best:
                best_l2_3s = current_l2_3s

            print(f"    Val Loss:   {val_losses['total']:.4f}")
            print(f"    Metrics:")
            print(f"      L2 @1s: {val_metrics['l2_1s']:.4f} m")
            print(f"      L2 @2s: {val_metrics['l2_2s']:.4f} m")
            print(f"      L2 @3s: {val_metrics['l2_3s']:.4f} m "
                  f"{'(BEST)' if is_best else ''}")
            print(f"      Collision Rate: {val_metrics['collision_rate']:.4f}")
            print(f"      Score Accuracy: {val_metrics['score_accuracy']:.4f}")

        # Save checkpoint
        save_checkpoint(
            model, optimizer, scheduler, scaler,
            epoch, best_l2_3s, args.save_dir, is_best=is_best)

        print("-" * 70)

    # =========================================================================
    # Training Complete
    # =========================================================================
    print("\n" + "=" * 70)
    print("Training Complete!")
    print(f"  Best L2 @3s: {best_l2_3s:.4f} m")
    print(f"  Checkpoints saved to: {args.save_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()
