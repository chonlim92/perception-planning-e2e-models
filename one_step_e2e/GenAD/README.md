# GenAD: Generative End-to-End Autonomous Driving

**Paper:** "GenAD: Generalized Predictive Model for Autonomous Driving"  
**Authors:** Zhiqi Li, Zhiding Yu, et al.  
**Year:** 2024  
**arXiv:** https://arxiv.org/abs/2402.11502

## Overview

GenAD uses a generative model (diffusion/autoregressive) to produce diverse planning outputs. Unlike discriminative models that output a single trajectory, GenAD generates multiple possible futures and selects among them — naturally handling the multi-modal nature of driving.

## Key Insight

Driving is inherently multi-modal: at any decision point, multiple valid trajectories exist. Generative models naturally capture this by sampling from a learned distribution over trajectories.

## Architecture

```
Sensor Input → [Encoder] → Scene Representation
                                    │
                                    ▼
                    ┌─────────────────────────────┐
                    │  Generative Trajectory Model  │
                    │  (Diffusion / Autoregressive) │
                    │                               │
                    │  Noise → Denoise → Trajectory │
                    │       (conditioned on scene)   │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                        Multiple Trajectory Samples
                                    │
                                    ▼
                          [Scorer / Selector]
                                    │
                                    ▼
                            Best Trajectory
```

## Diffusion for Trajectory Generation

```python
# Training: add noise to expert trajectory, learn to denoise
noisy_traj = expert_traj + noise * sigma_t
predicted_noise = model(noisy_traj, t, scene_context)
loss = MSE(predicted_noise, noise)

# Inference: start from pure noise, iteratively denoise
trajectory = random_noise
for t in reversed(timesteps):
    trajectory = denoise_step(trajectory, t, scene_context)
```

## Why Generative?

| Approach | Multi-modal? | Diversity | Training |
|----------|:---:|:---:|----------|
| Regression (L2) | No (averages modes) | None | Simple |
| K-modes (VAD) | Limited (K fixed) | Fixed K | Winner-take-all |
| **Generative** | **Yes (unlimited)** | **High** | **Diffusion/AR** |

## Files

```
GenAD/
├── README.md       # This file
├── model.py        # Diffusion-based trajectory generator
└── config.py       # Configuration
```
