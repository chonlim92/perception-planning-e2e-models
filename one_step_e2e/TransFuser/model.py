"""
TransFuser: Multi-Modal Fusion Transformer for End-to-End Driving

Reference implementation of TransFuser's core architecture:
- ResNet image encoder
- ResNet LiDAR BEV encoder
- Multi-scale transformer fusion
- GRU waypoint prediction

For the full implementation: https://github.com/autonomousvision/transfuser
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class TransformerFusionBlock(nn.Module):
    """
    Fusion block that applies transformer attention between
    image and LiDAR features at a single scale.
    """

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.channels = channels
        self.norm_img = nn.LayerNorm(channels)
        self.norm_lid = nn.LayerNorm(channels)

        self.attention = nn.MultiheadAttention(
            embed_dim=channels, num_heads=num_heads, batch_first=True)

        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Linear(channels * 4, channels),
        )
        self.norm_ffn = nn.LayerNorm(channels)

    def forward(self, img_feat: torch.Tensor,
                lidar_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            img_feat: (B, C, H_img, W_img) image features at this scale
            lidar_feat: (B, C, H_lid, W_lid) LiDAR features at this scale
        Returns:
            fused_img: (B, C, H_img, W_img)
            fused_lidar: (B, C, H_lid, W_lid)
        """
        B, C, Hi, Wi = img_feat.shape
        _, _, Hl, Wl = lidar_feat.shape

        # Flatten spatial dims to sequence
        img_tokens = img_feat.flatten(2).permute(0, 2, 1)    # (B, Hi*Wi, C)
        lidar_tokens = lidar_feat.flatten(2).permute(0, 2, 1)  # (B, Hl*Wl, C)

        # Concatenate for joint attention
        all_tokens = torch.cat([img_tokens, lidar_tokens], dim=1)  # (B, N_total, C)
        all_normed = torch.cat([self.norm_img(img_tokens),
                                self.norm_lid(lidar_tokens)], dim=1)

        # Self-attention across all tokens
        attended, _ = self.attention(all_normed, all_normed, all_normed)
        all_tokens = all_tokens + attended

        # FFN
        residual = all_tokens
        all_tokens = self.norm_ffn(all_tokens)
        all_tokens = residual + self.ffn(all_tokens)

        # Split back
        n_img = Hi * Wi
        fused_img = all_tokens[:, :n_img].permute(0, 2, 1).reshape(B, C, Hi, Wi)
        fused_lidar = all_tokens[:, n_img:].permute(0, 2, 1).reshape(B, C, Hl, Wl)

        return fused_img, fused_lidar


class ResNetStage(nn.Module):
    """Single ResNet stage (sequence of basic blocks)."""

    def __init__(self, in_channels: int, out_channels: int, num_blocks: int = 2,
                 stride: int = 2):
        super().__init__()
        layers = [self._make_block(in_channels, out_channels, stride)]
        for _ in range(1, num_blocks):
            layers.append(self._make_block(out_channels, out_channels, 1))
        self.blocks = nn.Sequential(*layers)

    def _make_block(self, in_ch, out_ch, stride):
        downsample = None
        if stride != 1 or in_ch != out_ch:
            downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch))
        return BasicBlock(in_ch, out_ch, stride, downsample)

    def forward(self, x):
        return self.blocks(x)


class BasicBlock(nn.Module):
    """Standard ResNet basic block."""

    def __init__(self, in_ch, out_ch, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample:
            identity = self.downsample(x)
        return F.relu(out + identity)


class TransFuser(nn.Module):
    """
    TransFuser: End-to-end driving with multi-modal transformer fusion.

    Fuses camera images and LiDAR BEV at multiple scales using transformers,
    then predicts future waypoints via a GRU decoder.
    """

    def __init__(self, img_channels: int = 3, lidar_channels: int = 2,
                 num_waypoints: int = 4, hidden_dim: int = 512):
        super().__init__()
        self.num_waypoints = num_waypoints

        # Image encoder (ResNet-34 style)
        self.img_stem = nn.Sequential(
            nn.Conv2d(img_channels, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.img_stage1 = ResNetStage(64, 64, num_blocks=3, stride=1)
        self.img_stage2 = ResNetStage(64, 128, num_blocks=4, stride=2)
        self.img_stage3 = ResNetStage(128, 256, num_blocks=6, stride=2)
        self.img_stage4 = ResNetStage(256, 512, num_blocks=3, stride=2)

        # LiDAR BEV encoder (ResNet-18 style)
        self.lid_stem = nn.Sequential(
            nn.Conv2d(lidar_channels, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.lid_stage1 = ResNetStage(64, 64, num_blocks=2, stride=1)
        self.lid_stage2 = ResNetStage(64, 128, num_blocks=2, stride=2)
        self.lid_stage3 = ResNetStage(128, 256, num_blocks=2, stride=2)
        self.lid_stage4 = ResNetStage(256, 512, num_blocks=2, stride=2)

        # Transformer fusion at each scale
        self.fusion1 = TransformerFusionBlock(64, num_heads=4)
        self.fusion2 = TransformerFusionBlock(128, num_heads=4)
        self.fusion3 = TransformerFusionBlock(256, num_heads=8)
        self.fusion4 = TransformerFusionBlock(512, num_heads=8)

        # Feature aggregation
        self.img_pool = nn.AdaptiveAvgPool2d(1)
        self.lid_pool = nn.AdaptiveAvgPool2d(1)
        self.feature_proj = nn.Linear(512 * 2, hidden_dim)

        # Speed embedding (ego vehicle speed as input context)
        self.speed_embed = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, hidden_dim),
        )

        # GRU waypoint decoder
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.waypoint_head = nn.Linear(hidden_dim, 2)  # (dx, dy)

        # Auxiliary: BEV segmentation head (training only)
        self.bev_seg_head = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 4, 1),  # 4 classes: road, vehicle, pedestrian, other
        )

    def forward(self, image: torch.Tensor, lidar_bev: torch.Tensor,
                speed: torch.Tensor,
                return_aux: bool = False) -> Dict:
        """
        Args:
            image: (B, 3, H, W) front camera image (e.g., 256×512)
            lidar_bev: (B, 2, H_bev, W_bev) LiDAR BEV (e.g., 256×256)
            speed: (B, 1) current ego speed in m/s
            return_aux: whether to return auxiliary outputs (BEV segmentation)
        Returns:
            dict with 'waypoints' (B, num_waypoints, 2) and optional aux
        """
        # Stems
        img = self.img_stem(image)
        lid = self.lid_stem(lidar_bev)

        # Stage 1 + Fusion
        img = self.img_stage1(img)
        lid = self.lid_stage1(lid)
        img, lid = self.fusion1(img, lid)

        # Stage 2 + Fusion
        img = self.img_stage2(img)
        lid = self.lid_stage2(lid)
        img, lid = self.fusion2(img, lid)

        # Stage 3 + Fusion
        img = self.img_stage3(img)
        lid = self.lid_stage3(lid)
        img, lid = self.fusion3(img, lid)

        # Stage 4 + Fusion
        img = self.img_stage4(img)
        lid = self.lid_stage4(lid)
        img, lid = self.fusion4(img, lid)

        # Aggregate features
        img_global = self.img_pool(img).flatten(1)  # (B, 512)
        lid_global = self.lid_pool(lid).flatten(1)  # (B, 512)
        fused = torch.cat([img_global, lid_global], dim=1)  # (B, 1024)
        features = self.feature_proj(fused)  # (B, hidden_dim)

        # Add speed context
        speed_feat = self.speed_embed(speed)  # (B, hidden_dim)
        features = features + speed_feat

        # GRU waypoint decoding
        hidden = features
        waypoints = []
        for _ in range(self.num_waypoints):
            hidden = self.gru(features, hidden)
            wp = self.waypoint_head(hidden)  # (B, 2)
            waypoints.append(wp)

        waypoints = torch.stack(waypoints, dim=1)  # (B, T, 2)

        output = {'waypoints': waypoints}

        if return_aux:
            bev_seg = self.bev_seg_head(lid)  # (B, 4, H, W)
            output['bev_segmentation'] = bev_seg

        return output


class PIDController:
    """
    PID controller to convert waypoints to vehicle control signals.
    Used during CARLA evaluation.
    """

    def __init__(self, K_P=1.0, K_I=0.1, K_D=0.1, fps=10):
        self.K_P = K_P
        self.K_I = K_I
        self.K_D = K_D
        self.dt = 1.0 / fps
        self._error_integral = 0.0
        self._prev_error = 0.0

    def step(self, error: float) -> float:
        self._error_integral += error * self.dt
        derivative = (error - self._prev_error) / self.dt
        self._prev_error = error
        return self.K_P * error + self.K_I * self._error_integral + self.K_D * derivative

    def reset(self):
        self._error_integral = 0.0
        self._prev_error = 0.0


def waypoints_to_control(waypoints, speed, target_speed=4.0):
    """
    Convert predicted waypoints to (steer, throttle, brake) using PID.

    Args:
        waypoints: (num_wp, 2) predicted waypoints in ego frame
        speed: current ego speed (m/s)
        target_speed: desired speed (m/s)
    Returns:
        steer, throttle, brake
    """
    # Lateral control: aim for first waypoint
    aim = waypoints[0]
    angle = torch.atan2(aim[1], aim[0]).item()
    steer = angle / (3.14159 / 4)  # normalize to [-1, 1]
    steer = max(-1.0, min(1.0, steer))

    # Longitudinal: simple speed control
    speed_error = target_speed - speed
    if speed_error > 0:
        throttle = min(0.75, speed_error * 0.3)
        brake = 0.0
    else:
        throttle = 0.0
        brake = min(1.0, -speed_error * 0.3)

    return steer, throttle, brake


def demo():
    """Demo TransFuser."""
    print("TransFuser - End-to-End Driving Demo")
    print("=" * 45)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TransFuser(num_waypoints=4, hidden_dim=512).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")
    print(f"Device: {device}")

    # Dummy inputs
    B = 2
    image = torch.randn(B, 3, 256, 512, device=device)
    lidar_bev = torch.randn(B, 2, 256, 256, device=device)
    speed = torch.tensor([[5.0], [8.0]], device=device)

    with torch.no_grad():
        output = model(image, lidar_bev, speed, return_aux=True)

    print(f"\nInputs:")
    print(f"  Camera: {image.shape}")
    print(f"  LiDAR BEV: {lidar_bev.shape}")
    print(f"  Speed: {speed.shape}")
    print(f"\nOutputs:")
    print(f"  Waypoints: {output['waypoints'].shape}")
    print(f"  BEV segmentation: {output['bev_segmentation'].shape}")
    print(f"  Waypoint values (batch 0):")
    for i, wp in enumerate(output['waypoints'][0]):
        print(f"    t={0.5*(i+1):.1f}s: ({wp[0].item():.3f}, {wp[1].item():.3f})")


if __name__ == '__main__':
    demo()
