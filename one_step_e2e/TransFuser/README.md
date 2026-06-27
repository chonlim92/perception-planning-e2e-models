# TransFuser: Multi-Modal Fusion Transformer for End-to-End Driving

**Paper:** "TransFuser: Imitation with Transformer-Based Sensor Fusion for Autonomous Driving"  
**Authors:** Kashyap Chitta, Aditya Prakash, Bernhard Jaeger, Zehao Yu, Katrin Renz, Andreas Geiger  
**Venue:** CVPR 2022 (initial), IEEE TPAMI 2023 (extended)  
**arXiv:** https://arxiv.org/abs/2205.15997  
**Code:** https://github.com/autonomousvision/transfuser

## Overview

TransFuser is a one-step end-to-end driving model that fuses multi-modal sensor inputs (cameras + LiDAR) using transformers and directly outputs vehicle control signals or waypoints. It achieves state-of-the-art results on the CARLA autonomous driving benchmark.

## Key Insight

Instead of late fusion (fusing final features) or early fusion (fusing raw inputs), TransFuser performs **intermediate fusion** at multiple scales using transformer attention, allowing each modality to inform the other progressively.

## Architecture

```
Camera Image        LiDAR BEV
    │                   │
    ▼                   ▼
[ResNet-34]        [ResNet-18]
    │                   │
    ├── Stage 1 ────────┤  ← Transformer fusion
    │                   │
    ├── Stage 2 ────────┤  ← Transformer fusion
    │                   │
    ├── Stage 3 ────────┤  ← Transformer fusion
    │                   │
    ├── Stage 4 ────────┤  ← Transformer fusion
    │                   │
    ▼                   ▼
[Fused Features]   [Fused Features]
    │                   │
    └───── concat ──────┘
              │
              ▼
    ┌──────────────────┐
    │   GRU Waypoint   │
    │    Predictor     │
    └────────┬─────────┘
             │
             ▼
    [4 Waypoints (2s)]
             │
             ▼
    ┌──────────────────┐
    │   PID Controller │  (waypoints → steer, throttle, brake)
    └──────────────────┘
```

## Multi-Scale Fusion

At each ResNet stage, image and LiDAR features are fused via a transformer:

```python
# At stage i:
image_tokens = flatten(image_features_i)  # (B, H*W, C)
lidar_tokens = flatten(lidar_features_i)  # (B, H*W, C)
concat_tokens = cat([image_tokens, lidar_tokens])  # cross-modal tokens
fused = transformer_layer(concat_tokens)
image_fused, lidar_fused = split(fused)
```

This allows:
- Camera features to access LiDAR depth information early
- LiDAR features to access camera semantic/texture information
- Progressive refinement at multiple resolutions

## Training

### Imitation Learning
- **Data collection:** Expert agent drives in CARLA, recording observations + actions
- **Loss:** L1 loss on predicted waypoints vs. expert waypoints
- **Auxiliary losses:**
  - BEV semantic segmentation (road, vehicles, pedestrians)
  - Depth prediction
  - Traffic light detection

### Data
- ~90 hours of expert driving in CARLA
- 8 weather conditions × 4 towns
- 2 FPS recording

### Augmentation
- Random noise injection on expert actions
- DAgger-style data aggregation (optional)

## Key Results (CARLA Leaderboard)

| Method | Driving Score | Route Completion | Infraction Score |
|--------|:---:|:---:|:---:|
| TransFuser (PAMI) | 61.18 | 86.69 | 0.71 |
| InterFuser | 68.31 | 95.02 | 0.72 |
| TCP | 75.14 | 93.64 | 0.81 |
| Human Expert | 84.97 | 99.43 | 0.85 |

## Why "One-Step"?

TransFuser is one-step because:
- Raw sensor inputs (camera + LiDAR) go directly to driving waypoints
- No explicit intermediate perception output (no bounding boxes, no HD maps)
- Internal representations exist but are NOT used as interpretable outputs
- Single network: input sensors → output control/waypoints
- Auxiliary tasks (BEV segmentation) are only used during training, not inference

## Implementation Notes

- Trained in CARLA simulator
- Requires CARLA 0.9.10+
- Inference: ~10 FPS on single GPU
- Uses GPS+IMU for ego localization + route following

## Files in This Directory

```
TransFuser/
├── README.md           # This file
├── model.py            # TransFuser model implementation
├── train.py            # Training script (CARLA data)
├── config.py           # Configuration
├── pid_controller.py   # PID controller (waypoints → control)
└── evaluate.py         # CARLA evaluation
```
