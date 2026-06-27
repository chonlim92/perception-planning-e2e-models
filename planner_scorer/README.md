# Planner Scorer: Trajectory Scoring and Selection for Autonomous Driving

## Overview

In autonomous driving, planning is inherently **multi-modal** — multiple valid trajectories exist for any given scenario. For example, when approaching an obstacle, the ego vehicle can either:
- Change lanes to avoid it
- Slow down and follow behind
- Both are valid driving behaviors

A **Planner Scorer** evaluates and ranks candidate trajectories to select the best one based on safety, comfort, efficiency, and rule compliance.

```
Trajectory Proposals → [Scorer] → Ranked Scores → Best Trajectory → Controller
```

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Scene Context                      │
│  (BEV map, detected agents, traffic lights, route)  │
└────────────────────────┬────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│              Candidate Trajectory Set                 │
│  t1: [wp1, wp2, ..., wpT]  (change lane left)      │
│  t2: [wp1, wp2, ..., wpT]  (slow down)             │
│  t3: [wp1, wp2, ..., wpT]  (change lane right)     │
│  ...                                                 │
│  tK: [wp1, wp2, ..., wpT]  (accelerate through)    │
└────────────────────────┬────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│                   SCORER MODULE                       │
│                                                      │
│  ┌──────────────┐  ┌──────────────┐                │
│  │  Classical    │  │   Learned    │                │
│  │  (rule-based) │  │  (neural)    │                │
│  └──────┬───────┘  └──────┬───────┘                │
│         │                  │                         │
│         ▼                  ▼                         │
│  ┌──────────────────────────────┐                   │
│  │     Score Aggregation         │                   │
│  │  s(t) = w_cls*s_cls + w_nn*s_nn │                │
│  └──────────────┬───────────────┘                   │
└─────────────────┼───────────────────────────────────┘
                  │
                  ▼
         Best Trajectory t*
```

## Directory Structure

```
planner_scorer/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── docs/
│   ├── technical_overview.md    # Detailed technical documentation
│   ├── training_guide.md        # How to train scorer models
│   └── evaluation_guide.md      # How to evaluate scorers
├── classical/
│   ├── cost_function.py         # Weighted multi-criteria cost function
│   ├── frenet_scorer.py         # Frenet-frame trajectory evaluation
│   ├── safety_checker.py        # TTC, RSS, collision checking
│   └── README.md                # Classical methods documentation
├── learned/
│   ├── mlp_scorer.py            # MLP-based trajectory scorer
│   ├── transformer_scorer.py    # Cross-attention transformer scorer
│   ├── contrastive_scorer.py    # Contrastive learning scorer
│   ├── dataset.py               # Training data preparation
│   ├── train.py                 # Training loop
│   ├── config.py                # Hyperparameters
│   └── README.md                # Learned methods documentation
└── evaluation/
    ├── metrics.py               # Evaluation metrics (ADE, FDE, collision rate)
    ├── benchmark.py             # Benchmark against baselines
    ├── visualize.py             # Visualization tools
    └── README.md                # Evaluation documentation
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run classical scorer on sample data
python classical/cost_function.py --demo

# Train learned scorer
python learned/train.py --config learned/config.py

# Evaluate
python evaluation/benchmark.py --scorer learned --checkpoint best.pth
```

## Key Concepts

### Scoring Formulation

Given a candidate trajectory `t` and scene context `C`:

```
Score(t | C) = Σᵢ wᵢ · sᵢ(t, C)
```

Where sub-scores include:
- **Safety**: Collision risk, TTC, distance to obstacles
- **Comfort**: Acceleration, jerk, curvature limits
- **Progress**: Route progress, speed efficiency
- **Compliance**: Lane keeping, speed limits, traffic rules

### Training Approaches

| Approach | Labels | Loss |
|----------|--------|------|
| Regression | Continuous score [0,1] | MSE / Smooth L1 |
| Classification | Binary (good/bad) | BCE |
| Ranking | Pairwise (A > B) | Margin loss |
| Contrastive | Expert = positive | InfoNCE |
| IRL | Expert demos | MaxEnt IRL |

## References

- [nuPlan Challenge](https://www.nuscenes.org/nuplan) — Planning benchmark with scoring
- [Werling et al., 2010](https://ieeexplore.ieee.org/document/5509799) — Frenet-frame trajectory planning
- [GameFormer (ICCV 2023)](https://arxiv.org/abs/2303.05760) — Game-theoretic scoring
- [DIPP](https://arxiv.org/abs/2305.12071) — Differentiable integrated prediction and planning
- [CTG (NeurIPS 2023)](https://arxiv.org/abs/2304.01223) — Controlled trajectory generation
