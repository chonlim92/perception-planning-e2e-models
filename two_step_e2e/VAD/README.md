# VAD: Vectorized Scene Representation for Efficient Autonomous Driving

**Paper:** "VAD: Vectorized Scene Representation for Efficient Autonomous Driving"  
**Authors:** Bo Jiang, Shaoyu Chen, Qing Xu, Bencheng Liao, Jiajie Chen, et al.  
**Venue:** ICCV 2023  
**arXiv:** https://arxiv.org/abs/2303.12077  
**Code:** https://github.com/hustvl/VAD

## Overview

VAD represents the driving scene entirely in vectorized form — agents as motion vectors, map elements as polyline vectors, and plans as trajectory vectors. This vectorized representation is more efficient and compact than dense BEV raster maps.

## Key Insight

Dense BEV representations (200×200 feature maps) contain massive redundancy. Most driving-relevant information can be captured by a small set of vectors: agent positions/motions, map polylines, and the ego trajectory itself.

## Architecture

```
Multi-view Images → [Backbone + BEV] → BEV Features
                                            │
                    ┌───────────────────────────────────────┐
                    │                                       │
                    ▼                                       ▼
          ┌─────────────────┐                    ┌─────────────────┐
          │  Agent Queries   │                    │   Map Queries    │
          │  (vectorized     │                    │  (vectorized     │
          │   motion)        │                    │   polylines)     │
          └────────┬────────┘                    └────────┬────────┘
                   │                                      │
                   └──────────────┬───────────────────────┘
                                  │
                                  ▼
                    ┌──────────────────────────┐
                    │      Ego Queries (K)      │
                    │  (K candidate plans)      │
                    │                           │
                    │  Cross-attention to:      │
                    │  • Agent vectors          │
                    │  • Map vectors            │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │    Scoring Head           │
                    │  (select best of K)       │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                          Best Trajectory
```

## Vectorized Representation

### Agent Vectors
Each detected agent is represented as:
- Current state: (x, y, heading, vx, vy)
- Predicted motion: sequence of (dx, dy) displacements

### Map Vectors
Map elements (lane boundaries, road edges, crosswalks) as:
- Ordered sequences of (x, y) points (polylines)
- Semantic type labels

### Ego Planning Vectors
K learnable ego queries, each producing a trajectory:
- Sequence of (x, y) waypoints for the ego vehicle
- Associated score (quality estimate)

## Training

### Losses
- **Planning L2:** Regression to expert trajectory
- **Planning scoring:** BCE loss to select best trajectory from K candidates
- **Agent motion:** L1 loss on predicted agent motions
- **Map reconstruction:** Chamfer distance on map polylines
- **Vectorized scene constraint:** Ensures consistency between vectors

### Key Training Details
- Dataset: nuScenes (700 training scenes)
- Backbone: ResNet-50 + FPN
- BEV: BEVFormer-style with temporal fusion
- Training: 60 epochs, 8× A100 GPUs
- Optimizer: AdamW, lr=2e-4, cosine schedule

## Key Results

| Metric | VAD-Tiny | VAD-Base | UniAD |
|--------|:--------:|:--------:|:-----:|
| Planning L2 (3s) | 1.01m | 0.97m | 1.03m |
| Collision Rate | 0.31% | 0.25% | 0.31% |
| FPS | 8.4 | 4.5 | 1.8 |

VAD achieves comparable or better planning performance while being 2-5× faster.

## Why "Two-Step"?

- Perception (agent detection + map construction) → Planning (ego trajectory)
- Perception outputs vectorized features that flow directly to planning
- No post-processing between perception and planning stages
- Distinct perception and planning sub-networks
- End-to-end gradient flow from planning back to perception

## Files

```
VAD/
├── README.md       # This file
├── model.py        # VAD model implementation
├── config.py       # Configuration
└── train.py        # Training script
```
