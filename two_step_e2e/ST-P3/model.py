"""
ST-P3: Spatial Temporal Feature Learning for End-to-End Driving

Key components:
1. Multi-view image encoding
2. Lift-Splat-Shoot (LSS) to BEV
3. Temporal aggregation (ConvGRU)
4. BEV segmentation (perception)
5. GRU-based planning

Reference: https://github.com/OpenDriveLab/ST-P3
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List


class ConvGRUCell(nn.Module):
    """Convolutional GRU cell for temporal feature aggregation in BEV."""

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.reset_gate = nn.Conv2d(channels * 2, channels, kernel_size, padding=padding)
        self.update_gate = nn.Conv2d(channels * 2, channels, kernel_size, padding=padding)
        self.candidate = nn.Conv2d(channels * 2, channels, kernel_size, padding=padding)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) current BEV features
            h: (B, C, H, W) previous hidden state
        Returns:
            h_new: (B, C, H, W) updated hidden state
        """
        combined = torch.cat([x, h], dim=1)
        r = torch.sigmoid(self.reset_gate(combined))
        z = torch.sigmoid(self.update_gate(combined))
        combined_r = torch.cat([x, r * h], dim=1)
        candidate = torch.tanh(self.candidate(combined_r))
        h_new = (1 - z) * h + z * candidate
        return h_new


class SimplifiedLSS(nn.Module):
    """
    Simplified Lift-Splat-Shoot for BEV generation.

    Full LSS:
    1. Predicts depth distribution per pixel
    2. Creates frustum point cloud
    3. Splatts to BEV grid via pillar pooling

    Simplified: direct learned projection to BEV.
    """

    def __init__(self, in_channels: int = 256, bev_channels: int = 64,
                 bev_h: int = 200, bev_w: int = 200, num_cameras: int = 6):
        super().__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w

        # Per-camera depth prediction (simplified)
        self.depth_head = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 41, 1),  # 41 depth bins
        )

        # BEV projection (simplified)
        self.bev_proj = nn.Sequential(
            nn.Conv2d(in_channels * num_cameras, bev_channels * 4, 3, padding=1),
            nn.BatchNorm2d(bev_channels * 4),
            nn.ReLU(),
            nn.Conv2d(bev_channels * 4, bev_channels, 1),
            nn.AdaptiveAvgPool2d((bev_h, bev_w)),
        )

    def forward(self, img_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img_features: (B, num_cameras, C, H, W) multi-view features
        Returns:
            bev_features: (B, bev_channels, bev_h, bev_w)
        """
        B, N, C, H, W = img_features.shape
        # Simplified: concatenate and project
        concat = img_features.reshape(B, N * C, H, W)
        bev = self.bev_proj(concat)
        return bev


class STP3(nn.Module):
    """
    ST-P3: End-to-End Vision-Based Autonomous Driving

    Full pipeline with spatial-temporal feature learning.
    """

    def __init__(self, bev_channels: int = 64, bev_h: int = 200, bev_w: int = 200,
                 num_cameras: int = 6, num_seg_classes: int = 4,
                 num_waypoints: int = 6, temporal_frames: int = 4):
        super().__init__()
        self.bev_channels = bev_channels
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.temporal_frames = temporal_frames

        # Image backbone (simplified)
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        # LSS: images → BEV
        self.lss = SimplifiedLSS(
            in_channels=256, bev_channels=bev_channels,
            bev_h=bev_h, bev_w=bev_w, num_cameras=num_cameras)

        # Temporal aggregation
        self.temporal_gru = ConvGRUCell(bev_channels, kernel_size=3)

        # BEV encoder (spatial processing)
        self.bev_encoder = nn.Sequential(
            nn.Conv2d(bev_channels, bev_channels, 3, padding=1),
            nn.BatchNorm2d(bev_channels),
            nn.ReLU(),
            nn.Conv2d(bev_channels, bev_channels, 3, padding=1),
            nn.BatchNorm2d(bev_channels),
            nn.ReLU(),
        )

        # Perception: BEV segmentation
        self.seg_head = nn.Sequential(
            nn.Conv2d(bev_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, num_seg_classes, 1),
        )

        # Prediction: future occupancy
        self.occ_head = nn.Sequential(
            nn.Conv2d(bev_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 5, 1),  # 5 future timesteps
        )

        # Planning: GRU decoder
        plan_input_dim = bev_channels * (bev_h // 8) * (bev_w // 8)
        self.plan_pool = nn.AdaptiveAvgPool2d((bev_h // 8, bev_w // 8))
        self.plan_proj = nn.Linear(plan_input_dim, 512)
        self.plan_gru = nn.GRUCell(512, 512)
        self.waypoint_head = nn.Linear(512, 2)

        self.num_waypoints = num_waypoints

    def extract_bev(self, images: torch.Tensor) -> torch.Tensor:
        """Extract BEV features from multi-view images."""
        B, N, C, H, W = images.shape
        # Process each camera through backbone
        imgs_flat = images.reshape(B * N, C, H, W)
        features = self.backbone(imgs_flat)  # (B*N, 256, h, w)
        _, C_f, H_f, W_f = features.shape
        features = features.reshape(B, N, C_f, H_f, W_f)
        # Lift to BEV
        bev = self.lss(features)
        return bev

    def forward(self, image_sequence: torch.Tensor) -> Dict:
        """
        Args:
            image_sequence: (B, T, num_cameras, 3, H, W) temporal sequence
                           T = temporal_frames (e.g., 4 past frames)
        Returns:
            dict with segmentation, occupancy, and trajectory
        """
        B, T, N, C, H, W = image_sequence.shape

        # Process each timestep and aggregate temporally
        hidden = torch.zeros(B, self.bev_channels, self.bev_h, self.bev_w,
                             device=image_sequence.device)

        for t in range(T):
            bev_t = self.extract_bev(image_sequence[:, t])  # (B, C, H, W)
            hidden = self.temporal_gru(bev_t, hidden)

        # Spatial BEV encoding
        bev_encoded = self.bev_encoder(hidden)  # (B, C, H, W)

        # Perception: BEV segmentation
        seg_output = self.seg_head(bev_encoded)  # (B, num_classes, H, W)

        # Prediction: future occupancy
        occ_output = self.occ_head(bev_encoded)  # (B, 5, H, W)

        # Planning: GRU waypoint prediction
        plan_features = self.plan_pool(bev_encoded)  # (B, C, h, w)
        plan_features = plan_features.flatten(1)     # (B, C*h*w)
        plan_features = self.plan_proj(plan_features)  # (B, 512)

        gru_hidden = plan_features
        waypoints = []
        for _ in range(self.num_waypoints):
            gru_hidden = self.plan_gru(plan_features, gru_hidden)
            wp = self.waypoint_head(gru_hidden)
            waypoints.append(wp)

        trajectory = torch.stack(waypoints, dim=1)  # (B, T_plan, 2)

        return {
            'bev_segmentation': seg_output,
            'future_occupancy': occ_output,
            'trajectory': trajectory,
            'bev_features': bev_encoded,
        }


def demo():
    """Demo ST-P3."""
    print("ST-P3: Spatial Temporal Feature Learning Demo")
    print("=" * 50)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = STP3(bev_channels=64, bev_h=100, bev_w=100,
                 num_cameras=6, num_waypoints=6).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")

    B, T = 2, 4  # 2 batches, 4 temporal frames
    images = torch.randn(B, T, 6, 3, 128, 256, device=device)

    with torch.no_grad():
        output = model(images)

    print(f"\nInput: {images.shape} (B, T, cameras, C, H, W)")
    print(f"\nOutputs:")
    print(f"  BEV segmentation: {output['bev_segmentation'].shape}")
    print(f"  Future occupancy: {output['future_occupancy'].shape}")
    print(f"  Planned trajectory: {output['trajectory'].shape}")


if __name__ == '__main__':
    demo()
