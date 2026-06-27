# UniAD: Planning-oriented Autonomous Driving

**Paper:** "Planning-oriented Autonomous Driving"  
**Authors:** Yihan Hu, Jiazhi Yang, Li Chen, Keyu Li, Chonghao Sima, et al.  
**Venue:** CVPR 2023 (Best Paper Award)  
**arXiv:** https://arxiv.org/abs/2212.10156  
**Code:** https://github.com/OpenDriveLab/UniAD

## Overview

UniAD is the first end-to-end framework that unifies full-stack driving tasks (detection, tracking, mapping, motion prediction, occupancy prediction, and planning) in a single network. It demonstrates that joint optimization across all tasks benefits planning performance.

## Key Insight

Previous modular pipelines optimize each component independently. UniAD shows that when all tasks are designed around the ultimate goal of planning, the entire system benefits from gradient flow across modules.

## Architecture

```
                    ┌─────────────────────────────────────────────────┐
                    │               UniAD Architecture                 │
                    └─────────────────────────────────────────────────┘

Multi-view Images ──→ [Image Backbone] ──→ [BEV Encoder (BEVFormer)]
                            │
                            ▼
                    ┌──────────────┐
                    │  BEV Features │  (200×200, 256-dim)
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
    ┌──────────────┐ ┌──────────┐ ┌──────────────┐
    │  TrackFormer │ │ MapFormer│ │  MotionFormer │
    │  (Detection  │ │ (Online  │ │  (Prediction  │
    │  + Tracking) │ │  Mapping)│ │  + Occupancy) │
    └──────┬───────┘ └────┬─────┘ └──────┬───────┘
           │              │              │
           └──────────────┼──────────────┘
                          ▼
                ┌──────────────────┐
                │    Planner       │
                │  (GRU-based,     │
                │   ego-query)     │
                └────────┬─────────┘
                         ▼
                  Ego Trajectory
               (waypoints for 3s)
```

## Module Details

### 1. BEV Encoder (from BEVFormer)
- **Input:** 6 surround-view camera images (1600×900)
- **Backbone:** ResNet-101 or InternImage
- **BEV Generation:** Spatial cross-attention + temporal self-attention
- **Output:** BEV feature map (200×200×256), covering 60m×30m

### 2. TrackFormer (Detection + Tracking)
- Joint 3D detection and multi-object tracking
- Track queries persist across frames (newborn + existing)
- Outputs: 3D bounding boxes + track IDs + agent features
- Loss: Set prediction (Hungarian matching) + tracking consistency

### 3. MapFormer (Online Mapping)
- Predicts vectorized map elements: lane dividers, road boundaries, pedestrian crossings
- Uses deformable attention over BEV features
- Outputs: Polyline vectors + semantic labels

### 4. MotionFormer (Motion Prediction)
- Predicts future trajectories of tracked agents
- Multi-modal predictions (6 modes per agent)
- Agent-agent and agent-map interactions via attention
- Also predicts future occupancy grids

### 5. Planner
- **Input:** Ego query + agent predictions + map features + BEV features
- **Architecture:** GRU that autoregressively decodes ego waypoints
- **Interaction:** Cross-attention between ego query and predicted agent futures
- **Output:** 6 future waypoints (0.5s intervals, 3s total)
- **Loss:** L2 to expert trajectory + collision loss (penalizes overlap with predicted occupancies)

## Training

### Multi-Task Losses
```
L_total = λ_det * L_detection + λ_track * L_tracking + λ_map * L_mapping
          + λ_motion * L_motion + λ_occ * L_occupancy + λ_plan * L_planning
```

### Training Strategy
1. **Stage 1:** Train BEV encoder + perception modules (detection, tracking, mapping)
2. **Stage 2:** Add motion prediction, fine-tune end-to-end
3. **Stage 3:** Add planner, fine-tune entire network end-to-end

### Datasets
- **nuScenes:** 1000 scenes, 700 train / 150 val / 150 test
- 6 camera views, 32-beam LiDAR (for GT annotations), CAN bus (ego motion)

## Key Results

| Metric | UniAD | Previous SOTA |
|--------|-------|---------------|
| Planning L2 (3s) | 1.03m | 1.89m |
| Planning Collision Rate | 0.31% | 1.15% |
| Tracking AMOTA | 0.359 | 0.334 |
| Map mAP | 0.317 | 0.298 |
| Motion minADE | 0.708 | 0.812 |

## Why "Two-Step"?

UniAD is two-step because:
- Perception (TrackFormer + MapFormer) produces **learned intermediate features**
- These features flow directly to MotionFormer and Planner
- There is NO post-processing between perception and planning
- But the perception and planning are **distinct sub-networks** with their own architectures
- Gradient flows from planning loss all the way back to the image backbone

## Implementation Notes

- Built on mmdetection3d framework
- Requires 8× A100 GPUs for training (24h per stage)
- Inference: ~2 FPS on single A100
- Key dependencies: PyTorch, mmcv, mmdet3d, nuscenes-devkit

## Files in This Directory

```
UniAD/
├── README.md              # This file
├── model.py               # Simplified UniAD model implementation
├── planner.py             # Planning module (GRU + ego query)
├── train.py               # Training script
├── evaluate.py            # Evaluation script
├── config.py              # Model configuration
└── docs/
    └── architecture.md    # Detailed architecture documentation
```
