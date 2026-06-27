"""
TCP: Trajectory-guided Control Prediction

Dual-branch architecture:
- Trajectory branch: predicts future waypoints (GRU decoder)
- Control branch: predicts control signals guided by trajectory features
- Adaptive fusion combines both outputs

Reference: https://github.com/OpenDriveLab/TCP
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class TCP(nn.Module):
    """
    TCP: Trajectory-guided Control Prediction.

    Combines trajectory prediction and direct control prediction,
    using the trajectory to guide control generation.
    """

    def __init__(self, num_waypoints: int = 4, hidden_dim: int = 512,
                 img_channels: int = 3, lidar_channels: int = 2):
        super().__init__()
        self.num_waypoints = num_waypoints
        self.hidden_dim = hidden_dim

        # Shared feature encoder
        self.img_encoder = nn.Sequential(
            nn.Conv2d(img_channels, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.BatchNorm2d(512), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

        self.lidar_encoder = nn.Sequential(
            nn.Conv2d(lidar_channels, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

        self.feature_fusion = nn.Sequential(
            nn.Linear(512 + 256, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Speed embedding
        self.speed_embed = nn.Sequential(
            nn.Linear(1, 64), nn.ReLU(), nn.Linear(64, hidden_dim))

        # ===== Trajectory Branch =====
        self.traj_gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.traj_head = nn.Linear(hidden_dim, 2)  # (dx, dy) per waypoint

        # ===== Control Branch (trajectory-guided) =====
        # Trajectory features guide control through attention
        self.traj_feature_proj = nn.Linear(num_waypoints * 2, hidden_dim)

        self.control_attention = nn.MultiheadAttention(
            hidden_dim, num_heads=4, batch_first=True)
        self.control_norm = nn.LayerNorm(hidden_dim)

        self.control_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),  # (steer, throttle, brake)
        )

        # ===== Adaptive Fusion =====
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim + 3, 1),
            nn.Sigmoid(),
        )

    def forward(self, image: torch.Tensor, lidar_bev: torch.Tensor,
                speed: torch.Tensor) -> Dict:
        """
        Args:
            image: (B, 3, H, W)
            lidar_bev: (B, 2, H, W)
            speed: (B, 1)
        Returns:
            dict with waypoints, control, and fused outputs
        """
        B = image.shape[0]

        # Feature extraction
        img_feat = self.img_encoder(image).flatten(1)    # (B, 512)
        lid_feat = self.lidar_encoder(lidar_bev).flatten(1)  # (B, 256)
        features = self.feature_fusion(torch.cat([img_feat, lid_feat], dim=1))
        features = features + self.speed_embed(speed)  # (B, hidden_dim)

        # ----- Trajectory Branch -----
        hidden = features
        waypoints = []
        for _ in range(self.num_waypoints):
            hidden = self.traj_gru(features, hidden)
            wp = self.traj_head(hidden)
            waypoints.append(wp)
        waypoints_tensor = torch.stack(waypoints, dim=1)  # (B, T, 2)

        # ----- Control Branch (trajectory-guided) -----
        # Use trajectory prediction to guide control
        traj_flat = waypoints_tensor.reshape(B, -1)  # (B, T*2)
        traj_features = self.traj_feature_proj(traj_flat)  # (B, hidden_dim)

        # Cross-attention: features attend to trajectory guidance
        query = features.unsqueeze(1)  # (B, 1, hidden_dim)
        key_value = traj_features.unsqueeze(1)  # (B, 1, hidden_dim)
        attended, _ = self.control_attention(query, key_value, key_value)
        attended = self.control_norm(attended.squeeze(1))  # (B, hidden_dim)

        # Combine for control prediction
        control_input = torch.cat([features, attended], dim=1)
        control = self.control_head(control_input)  # (B, 3)
        steer = torch.tanh(control[:, 0:1])
        throttle = torch.sigmoid(control[:, 1:2])
        brake = torch.sigmoid(control[:, 2:3])
        control_output = torch.cat([steer, throttle, brake], dim=1)

        # ----- Adaptive Fusion -----
        # Convert waypoints to control via simple PID for fusion
        aim = waypoints_tensor[:, 0]  # first waypoint
        traj_steer = torch.atan2(aim[:, 1:2], aim[:, 0:1] + 1e-6) / 1.57
        traj_steer = traj_steer.clamp(-1, 1)

        # Gate: how much to trust trajectory vs direct control
        gate_input = torch.cat([features, control_output], dim=1)
        gate = self.fusion_gate(gate_input)  # (B, 1)

        # Fused steer
        fused_steer = gate * traj_steer + (1 - gate) * steer

        return {
            'waypoints': waypoints_tensor,         # (B, T, 2)
            'control': control_output,             # (B, 3) [steer, throttle, brake]
            'fused_steer': fused_steer,            # (B, 1)
            'fusion_gate': gate,                   # (B, 1)
            'trajectory_steer': traj_steer,        # (B, 1)
        }


def compute_tcp_loss(output: Dict, gt_waypoints: torch.Tensor,
                     gt_control: torch.Tensor) -> Dict:
    """Multi-task TCP loss."""
    # Trajectory loss
    traj_loss = F.l1_loss(output['waypoints'], gt_waypoints)

    # Control loss
    ctrl_loss = F.l1_loss(output['control'], gt_control)

    # Fused steer loss
    fused_loss = F.l1_loss(output['fused_steer'], gt_control[:, 0:1])

    total = traj_loss + ctrl_loss + 0.5 * fused_loss
    return {'total': total, 'trajectory': traj_loss,
            'control': ctrl_loss, 'fused': fused_loss}


def demo():
    print("TCP: Trajectory-guided Control Prediction Demo")
    print("=" * 50)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TCP(num_waypoints=4, hidden_dim=256).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    B = 4
    image = torch.randn(B, 3, 256, 512, device=device)
    lidar = torch.randn(B, 2, 256, 256, device=device)
    speed = torch.rand(B, 1, device=device) * 10

    with torch.no_grad():
        out = model(image, lidar, speed)

    print(f"\nWaypoints: {out['waypoints'].shape}")
    print(f"Control: {out['control'].shape}")
    print(f"Fused steer: {out['fused_steer'][0].item():.3f}")
    print(f"Fusion gate: {out['fusion_gate'][0].item():.3f}")
    print(f"  (gate=1 -> trust trajectory, gate=0 -> trust direct control)")


if __name__ == '__main__':
    demo()
