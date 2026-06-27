"""
Training script for learned trajectory scorers.

Supports multiple loss functions:
- Binary Cross-Entropy (classification: good/bad trajectory)
- Margin Ranking Loss (pairwise: expert > non-expert)
- InfoNCE / Contrastive Loss (1 positive + N negatives)
- Combined loss (weighted sum of above)

Usage:
    python train.py --model mlp --loss combined --epochs 100
    python train.py --model transformer --loss contrastive --epochs 50
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
import os
import time
from typing import Tuple

from config import FullConfig, TrainingConfig
from mlp_scorer import MLPScorer
from transformer_scorer import TransformerScorer


class SyntheticScorerDataset(Dataset):
    """
    Synthetic dataset for training trajectory scorers.

    Generates:
    - Expert trajectories (smooth, collision-free, lane-following)
    - Perturbed negatives (noisy, off-lane, collision-prone)

    In production, replace with real data from nuScenes/nuPlan.
    """

    def __init__(self, num_scenes: int = 10000, num_candidates: int = 64,
                 traj_len: int = 16, num_agents: int = 32,
                 num_map_polys: int = 64, seed: int = 42):
        self.num_scenes = num_scenes
        self.num_candidates = num_candidates
        self.traj_len = traj_len
        self.num_agents = num_agents
        self.num_map_polys = num_map_polys
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return self.num_scenes

    def __getitem__(self, idx):
        rng = np.random.RandomState(idx)

        # Generate expert trajectory: smooth forward motion
        dt = 0.5
        speed = rng.uniform(5, 15)
        slight_curve = rng.uniform(-0.02, 0.02)
        t = np.arange(self.traj_len) * dt

        expert_x = speed * t
        expert_y = slight_curve * t ** 2
        expert_heading = np.arctan2(2 * slight_curve * t, speed)
        expert_v = np.ones(self.traj_len) * speed

        expert_traj = np.stack([expert_x, expert_y, expert_heading, expert_v], axis=-1)

        # Generate candidate trajectories
        candidates = np.zeros((self.num_candidates, self.traj_len, 4))
        labels = np.zeros(self.num_candidates)

        # First candidate = expert (label = 1.0)
        candidates[0] = expert_traj
        labels[0] = 1.0

        # Rest = perturbed versions (varying quality)
        for i in range(1, self.num_candidates):
            noise_level = rng.uniform(0.1, 3.0)
            lateral_offset = rng.uniform(-2, 2)
            speed_factor = rng.uniform(0.5, 1.5)

            cand_x = expert_x + rng.randn(self.traj_len) * noise_level * 0.3
            cand_y = expert_y + lateral_offset + rng.randn(self.traj_len) * noise_level
            cand_heading = expert_heading + rng.randn(self.traj_len) * noise_level * 0.1
            cand_v = expert_v * speed_factor + rng.randn(self.traj_len) * noise_level * 0.5

            candidates[i] = np.stack([cand_x, cand_y, cand_heading, cand_v], axis=-1)

            # Quality label based on deviation from expert
            deviation = np.mean((candidates[i, :, :2] - expert_traj[:, :2]) ** 2)
            labels[i] = max(0, 1.0 - deviation / 20.0)

        # Generate agent features (random for synthetic data)
        agents = rng.randn(self.num_agents, 7).astype(np.float32)
        agent_mask = np.ones(self.num_agents, dtype=np.bool_)
        num_valid = rng.randint(5, self.num_agents)
        agent_mask[num_valid:] = False

        # Generate map features
        map_features = rng.randn(self.num_map_polys, 5).astype(np.float32)
        map_mask = np.ones(self.num_map_polys, dtype=np.bool_)
        num_valid_map = rng.randint(10, self.num_map_polys)
        map_mask[num_valid_map:] = False

        return {
            'candidates': torch.tensor(candidates, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.float32),
            'agents': torch.tensor(agents, dtype=torch.float32),
            'agent_mask': torch.tensor(agent_mask, dtype=torch.bool),
            'map_features': torch.tensor(map_features, dtype=torch.float32),
            'map_mask': torch.tensor(map_mask, dtype=torch.bool),
            'expert_idx': torch.tensor(0, dtype=torch.long),
        }


class ScorerLoss(nn.Module):
    """Combined loss for trajectory scoring."""

    def __init__(self, loss_type: str = 'combined',
                 temperature: float = 0.07, margin: float = 0.5,
                 weights: dict = None):
        super().__init__()
        self.loss_type = loss_type
        self.temperature = temperature
        self.margin = margin
        self.weights = weights or {'classification': 1.0, 'ranking': 0.5, 'contrastive': 0.3}

    def forward(self, scores: torch.Tensor, labels: torch.Tensor,
                expert_idx: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            scores: (B, K) predicted scores for K candidates
            labels: (B, K) ground truth quality labels
            expert_idx: (B,) index of expert trajectory
        Returns:
            total_loss, loss_dict
        """
        losses = {}

        if self.loss_type in ('bce', 'combined'):
            # Binary cross-entropy on score vs label
            bce = F.binary_cross_entropy_with_logits(scores, labels)
            losses['bce'] = bce

        if self.loss_type in ('ranking', 'combined'):
            # Margin ranking: expert should score higher than all others
            B, K = scores.shape
            expert_scores = scores.gather(1, expert_idx.unsqueeze(1))  # (B, 1)
            # Compare expert to all non-expert
            ranking_loss = torch.tensor(0.0, device=scores.device)
            count = 0
            for i in range(K):
                mask = (torch.arange(K, device=scores.device) != expert_idx.unsqueeze(0)).float()
                if mask.sum() > 0:
                    neg_scores = scores * mask + scores.min() * (1 - mask)
                    loss_i = F.relu(self.margin - expert_scores + neg_scores).mean()
                    ranking_loss = ranking_loss + loss_i
                    count += 1
                break  # simplified: compare expert to mean of others
            if count > 0:
                ranking_loss = ranking_loss / count
            losses['ranking'] = ranking_loss

        if self.loss_type in ('contrastive', 'combined'):
            # InfoNCE: expert is positive, rest are negatives
            B, K = scores.shape
            expert_scores = scores.gather(1, expert_idx.unsqueeze(1))  # (B, 1)
            logits = scores / self.temperature  # (B, K)
            info_nce = F.cross_entropy(logits, expert_idx)
            losses['contrastive'] = info_nce

        # Compute total
        if self.loss_type == 'combined':
            total = sum(self.weights.get(k, 1.0) * v for k, v in losses.items())
        else:
            total = sum(losses.values())

        losses['total'] = total
        return total, losses


def train_epoch(model, dataloader, optimizer, criterion, device, grad_clip=1.0):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0

    for batch in dataloader:
        candidates = batch['candidates'].to(device)
        labels = batch['labels'].to(device)
        agents = batch['agents'].to(device)
        agent_mask = batch['agent_mask'].to(device)
        map_features = batch['map_features'].to(device)
        expert_idx = batch['expert_idx'].to(device)

        B, K, T, D = candidates.shape

        # Score all candidates
        scores, _ = model.score_candidates(
            candidates, agents, agent_mask, map_features)

        # Compute loss
        loss, _ = criterion(scores, labels, expert_idx)

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    """Evaluate on validation set."""
    model.eval()
    total_loss = 0
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    for batch in dataloader:
        candidates = batch['candidates'].to(device)
        labels = batch['labels'].to(device)
        agents = batch['agents'].to(device)
        agent_mask = batch['agent_mask'].to(device)
        map_features = batch['map_features'].to(device)
        expert_idx = batch['expert_idx'].to(device)

        scores, best_idx = model.score_candidates(
            candidates, agents, agent_mask, map_features)

        loss, _ = criterion(scores, labels, expert_idx)
        total_loss += loss.item()

        # Accuracy: does best_idx match expert_idx?
        correct_top1 += (best_idx == expert_idx).sum().item()

        # Top-5: is expert in top-5 scores?
        top5 = scores.topk(5, dim=-1).indices
        for b in range(candidates.shape[0]):
            if expert_idx[b] in top5[b]:
                correct_top5 += 1
        total += candidates.shape[0]

    n = max(total, 1)
    return {
        'loss': total_loss / max(len(dataloader), 1),
        'top1_acc': correct_top1 / n,
        'top5_acc': correct_top5 / n,
    }


def main():
    parser = argparse.ArgumentParser(description='Train Trajectory Scorer')
    parser.add_argument('--model', choices=['mlp', 'transformer'], default='mlp')
    parser.add_argument('--loss', choices=['bce', 'ranking', 'contrastive', 'combined'], default='combined')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save_dir', default='checkpoints')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = args.device
    print(f"Training {args.model} scorer with {args.loss} loss on {device}")

    # Create model
    if args.model == 'mlp':
        model = MLPScorer(traj_points=16, traj_dim=4, agent_dim=7, map_dim=5).to(device)
    else:
        model = TransformerScorer(traj_dim=4, agent_dim=7, map_dim=5, d_model=256).to(device)

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Dataset
    train_dataset = SyntheticScorerDataset(num_scenes=5000, seed=42)
    val_dataset = SyntheticScorerDataset(num_scenes=500, seed=123)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=2)

    # Optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = ScorerLoss(loss_type=args.loss)

    best_val_loss = float('inf')

    print(f"\n{'Epoch':>5} | {'Train Loss':>10} | {'Val Loss':>8} | {'Top-1':>6} | {'Top-5':>6} | {'Time':>6}")
    print("-" * 60)

    for epoch in range(1, args.epochs + 1):
        start = time.time()

        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - start

        print(f"{epoch:5d} | {train_loss:10.4f} | {val_metrics['loss']:8.4f} | "
              f"{val_metrics['top1_acc']:6.1%} | {val_metrics['top5_acc']:6.1%} | "
              f"{elapsed:5.1f}s")

        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'val_metrics': val_metrics,
            }, os.path.join(args.save_dir, 'best.pth'))

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoint saved to {args.save_dir}/best.pth")


if __name__ == '__main__':
    main()
