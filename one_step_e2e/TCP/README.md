# TCP: Trajectory-guided Control Prediction for End-to-End Autonomous Driving

**Paper:** "TCP: Trajectory-guided Control Prediction for End-to-End Autonomous Driving"  
**Authors:** Penghao Wu, Xiaosong Jia, Li Chen, Junchi Yan, Hongyang Li, Yu Qiao  
**Venue:** NeurIPS 2022  
**arXiv:** https://arxiv.org/abs/2206.08129  
**Code:** https://github.com/OpenDriveLab/TCP

## Overview

TCP addresses a key limitation of previous E2E methods: trajectory prediction and direct control prediction have complementary strengths. TCP fuses both through a novel multi-step trajectory-guided control prediction.

## Key Insight

- **Trajectory branch:** Predicts future waypoints (interpretable, good for long-horizon)
- **Control branch:** Directly predicts steer/throttle/brake (responsive, good for immediate)
- **TCP fusion:** Uses predicted trajectory to GUIDE control prediction, getting the best of both

## Architecture

```
Camera Image + LiDAR BEV
         │
         ▼
   [Feature Encoder]
   (ResNet + Transformer Fusion)
         │
         ├──────────────────────────────────┐
         ▼                                  ▼
┌─────────────────┐              ┌─────────────────────┐
│ Trajectory      │              │ Control Branch      │
│ Branch          │              │                     │
│ (GRU decoder    │──────────────│ Trajectory-guided   │
│  → waypoints)   │  guidance    │ attention on control │
│                 │              │ → (steer, gas, brake)│
└─────────────────┘              └─────────────────────┘
         │                                  │
         ▼                                  ▼
   4 Waypoints                    Direct Control
   (for PID)                      (steer, throttle, brake)
         │                                  │
         └──────── Adaptive Fusion ─────────┘
                         │
                         ▼
                  Final Control Output
```

## Why Both Branches?

| Branch | Strengths | Weaknesses |
|--------|-----------|------------|
| Trajectory | Interpretable, smooth, long-horizon | PID errors, latency |
| Control | Responsive, direct, no PID | Jerky, mode collapse |
| TCP (fused) | Best of both — smooth AND responsive | More complex |

## Key Results (CARLA)

| Method | Driving Score | Route Completion |
|--------|:---:|:---:|
| TransFuser | 54.52 | 78.41 |
| InterFuser | 68.31 | 95.02 |
| **TCP** | **75.14** | **93.64** |

## Files

```
TCP/
├── README.md      # This file
├── model.py       # TCP implementation
└── config.py      # Configuration
```
