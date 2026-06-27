"""
InterFuser: Safety-Enhanced Sensor Fusion Transformer

Multi-modal transformer with interpretable intermediate representations.
Produces safety-related maps alongside driving waypoints.

Reference: https://github.com/opendilab/InterFuser
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class InterFuser(nn.Module):
    """
    InterFuser: Multi-modal transformer for safe E2E driving.
    """

    def __init__(self, d_model: int = 256, n_heads: int = 8,
                 num_layers: int = 6, num_waypoints: int = 4,
                 bev_size: int = 32):
        super().__init__()
        self.d_model = d_model
        self.bev_size = bev_size

        # Multi-view image encoders (front, left, right)
        self.view_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(3, 64, 7, stride=4, padding=3), nn.ReLU(),
                nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(128, d_model, 3, stride=2, padding=1), nn.ReLU(),
            ) for _ in range(3)
        ])

        # LiDAR BEV encoder
        self.lidar_encoder = nn.Sequential(
            nn.Conv2d(2, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, d_model, 3, stride=2, padding=1), nn.ReLU(),
        )

        # Joint multi-modal transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Interpretable output heads
        self.density_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, bev_size * bev_size),
        )  # traffic density map

        self.waypoint_heatmap_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, bev_size * bev_size),
        )  # where to drive

        self.safety_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )  # safety confidence

        # Waypoint decoder
        self.wp_query = nn.Parameter(torch.randn(1, num_waypoints, d_model) * 0.02)
        self.wp_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, n_heads, d_model * 4, batch_first=True),
            num_layers=2)
        self.wp_head = nn.Linear(d_model, 2)

        self.num_waypoints = num_waypoints

    def forward(self, front_img: torch.Tensor, left_img: torch.Tensor,
                right_img: torch.Tensor, lidar_bev: torch.Tensor) -> Dict:
        """
        Args:
            front_img: (B, 3, H, W)
            left_img: (B, 3, H, W)
            right_img: (B, 3, H, W)
            lidar_bev: (B, 2, H, W)
        Returns:
            dict with waypoints, density_map, waypoint_heatmap, safety_score
        """
        B = front_img.shape[0]
        images = [front_img, left_img, right_img]

        # Encode each modality into tokens
        all_tokens = []
        for i, encoder in enumerate(self.view_encoders):
            feat = encoder(images[i])  # (B, D, h, w)
            tokens = feat.flatten(2).permute(0, 2, 1)  # (B, hw, D)
            all_tokens.append(tokens)

        lidar_feat = self.lidar_encoder(lidar_bev)
        lidar_tokens = lidar_feat.flatten(2).permute(0, 2, 1)
        all_tokens.append(lidar_tokens)

        # Concatenate all modality tokens
        multi_modal_tokens = torch.cat(all_tokens, dim=1)  # (B, N_total, D)

        # Joint transformer processing
        fused = self.transformer(multi_modal_tokens)  # (B, N_total, D)

        # Global feature (mean pool)
        global_feat = fused.mean(dim=1)  # (B, D)

        # Interpretable outputs
        density_map = self.density_head(global_feat)
        density_map = density_map.reshape(B, 1, self.bev_size, self.bev_size)
        density_map = torch.sigmoid(density_map)

        wp_heatmap = self.waypoint_heatmap_head(global_feat)
        wp_heatmap = wp_heatmap.reshape(B, 1, self.bev_size, self.bev_size)
        wp_heatmap = torch.sigmoid(wp_heatmap)

        safety_score = self.safety_head(global_feat)

        # Waypoint prediction via cross-attention decoder
        wp_queries = self.wp_query.expand(B, -1, -1)
        wp_features = self.wp_decoder(wp_queries, fused)
        waypoints = self.wp_head(wp_features)  # (B, num_wp, 2)

        return {
            'waypoints': waypoints,
            'density_map': density_map,
            'waypoint_heatmap': wp_heatmap,
            'safety_score': safety_score,
        }


def demo():
    print("InterFuser Demo")
    print("=" * 40)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = InterFuser(d_model=256, num_waypoints=4).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    B = 2
    front = torch.randn(B, 3, 256, 512, device=device)
    left = torch.randn(B, 3, 256, 512, device=device)
    right = torch.randn(B, 3, 256, 512, device=device)
    lidar = torch.randn(B, 2, 256, 256, device=device)

    with torch.no_grad():
        out = model(front, left, right, lidar)

    print(f"\nWaypoints: {out['waypoints'].shape}")
    print(f"Density map: {out['density_map'].shape}")
    print(f"Waypoint heatmap: {out['waypoint_heatmap'].shape}")
    print(f"Safety score: {out['safety_score'][0].item():.3f}")


if __name__ == '__main__':
    demo()
