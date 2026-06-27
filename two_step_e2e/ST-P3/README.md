# ST-P3: End-to-End Vision-Based Autonomous Driving via Spatial Temporal Feature Learning

**Paper:** "ST-P3: End-to-end Vision-based Autonomous Driving via Spatial Temporal Feature Learning"  
**Authors:** Shengchao Hu, Li Chen, Penghao Wu, Hongyang Li, Junchi Yan, Dacheng Tao  
**Venue:** ECCV 2022  
**arXiv:** https://arxiv.org/abs/2207.07601  
**Code:** https://github.com/OpenDriveLab/ST-P3

## Overview

ST-P3 is an early two-step E2E model that connects perception features directly to planning through spatial-temporal learning. It introduces three key designs: spatial feature extraction in BEV, temporal feature aggregation across frames, and dual-pathway planning that combines semantic understanding with geometric reasoning.

## Architecture

```
Multi-view Cameras (t-3, t-2, t-1, t)
            │
            ▼
    [Image Backbone (EfficientNet-B4)]
            │
            ▼
    [Lift-Splat-Shoot (LSS) → BEV]
            │
            ▼
    ┌───────────────────────────┐
    │  Spatial-Temporal Encoder  │
    │  • Spatial: BEV Conv       │
    │  • Temporal: GRU/ConvLSTM  │
    │  across past N frames      │
    └─────────────┬─────────────┘
                  │
    ┌─────────────┼─────────────┐
    │             │             │
    ▼             ▼             ▼
[Perception]  [Prediction]  [Planning]
 BEV Seg       Future        GRU
 (road,        Occupancy     Waypoint
  vehicle)     Prediction    Decoder
```

## Key Innovations

### 1. Spatial Feature Learning (BEV via LSS)
- Lift-Splat-Shoot (LSS) projects multi-view images to BEV
- Explicit depth prediction per pixel
- BEV resolution: 0.5m/pixel, 100m×100m coverage

### 2. Temporal Feature Learning
- ConvGRU aggregates BEV features across past frames
- Captures motion patterns and scene dynamics
- 4-frame temporal window

### 3. Dual-Pathway Planning
- **Semantic pathway:** Uses BEV segmentation features (where is road, where are obstacles)
- **Geometric pathway:** Uses raw BEV features directly
- Both pathways feed into a GRU planner

## Training

- **Dataset:** nuScenes (700 train scenes)
- **Perception tasks:** BEV semantic segmentation
- **Prediction task:** Future occupancy prediction (next 2 seconds)
- **Planning task:** Waypoint prediction (next 3 seconds)
- **Joint training:** All tasks trained together end-to-end

## Key Results

| Metric | ST-P3 | NMP | Others |
|--------|:-----:|:---:|:------:|
| Planning L2 (3s) | 2.13m | 2.31m | >2.5m |
| Collision Rate | 1.27% | 1.92% | >2.0% |

## Why "Two-Step"?

- Perception (BEV encoding + segmentation) produces features
- These features feed directly into the planning GRU
- No post-processing between perception output and planning input
- Distinct perception encoder and planning decoder networks
- End-to-end gradient flow through the full pipeline

## Files

```
ST-P3/
├── README.md      # This file
├── model.py       # ST-P3 implementation
└── config.py      # Configuration
```
