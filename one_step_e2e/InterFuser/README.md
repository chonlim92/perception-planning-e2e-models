# InterFuser: Safety-Enhanced Autonomous Driving Using Interpretable Sensor Fusion Transformer

**Paper:** "Safety-Enhanced Autonomous Driving Using Interpretable Sensor Fusion Transformer"  
**Authors:** Hao Shao, Letian Wang, RuoBing Chen, Hongsheng Li, Yu Liu  
**Venue:** CoRL 2022  
**arXiv:** https://arxiv.org/abs/2207.14024  
**Code:** https://github.com/opendilab/InterFuser

## Overview

InterFuser uses a transformer-based architecture for multi-sensor fusion that produces interpretable intermediate representations (safety maps, waypoints) while remaining end-to-end. It emphasizes safety through explicit density and waypoint map predictions.

## Architecture

```
Multi-view Cameras (front, left, right) + LiDAR BEV
                    │
                    ▼
        [Multi-Modal Transformer Encoder]
        (separate tokenization per modality,
         then joint attention across all)
                    │
                    ▼
        [Interpretable Feature Maps]
        ├── Waypoint heatmap (where to go)
        ├── Traffic density map (where obstacles are)
        ├── Junction indicator
        └── Safety score
                    │
                    ▼
           [Waypoint Decoder (GRU)]
                    │
                    ▼
            Planned Waypoints → PID → Control
```

## Key Innovation: Interpretable Safety

Unlike pure black-box E2E models, InterFuser produces interpretable intermediate maps:
- **Waypoint heatmap:** Shows where the model thinks it should drive
- **Density map:** Shows detected obstacles (implicit perception)
- **Safety score:** Confidence in the current plan's safety

## Key Results (CARLA)

| Method | Driving Score | Route Comp. | Infraction |
|--------|:---:|:---:|:---:|
| TransFuser | 54.52 | 78.41 | 0.76 |
| **InterFuser** | **68.31** | **95.02** | **0.72** |
| TCP | 75.14 | 93.64 | 0.81 |

## Files

```
InterFuser/
├── README.md       # This file
├── model.py        # InterFuser implementation
└── config.py       # Configuration
```
