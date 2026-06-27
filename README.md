# End-to-End Perception + Planning Models for Autonomous Driving

A comprehensive research repository covering state-of-the-art end-to-end (E2E) models for autonomous driving, including implementations, technical documentation, and training/evaluation code.

## Repository Structure

```
.
├── two_step_e2e/          # Two-step E2E models (perception → planning, connected but distinct)
│   ├── UniAD/             # Unified Autonomous Driving (CVPR 2023)
│   ├── VAD/               # Vectorized Autonomous Driving (ICCV 2023)
│   └── ST-P3/             # Spatial-Temporal Feature Learning (ECCV 2022)
│
├── one_step_e2e/          # One-step E2E models (sensor in → planning out)
│   ├── TransFuser/        # Transformer-based sensor fusion (CVPR 2022 / PAMI 2023)
│   ├── InterFuser/        # Interpretable Sensor Fusion (CoRL 2022)
│   ├── TCP/               # Trajectory-guided Control Prediction (NeurIPS 2022)
│   ├── DriveVLM/          # Vision-Language Model for Driving (2024)
│   ├── GenAD/             # Generative End-to-End Autonomous Driving (2024)
│   └── GAIA-1/            # Generative World Model (Wayve, 2023)
│
└── planner_scorer/        # Trajectory scoring and selection methods
    ├── docs/              # Technical documentation and guides
    ├── classical/         # Rule-based and cost-function scoring
    ├── learned/           # Neural network-based scoring
    └── evaluation/        # Benchmarks and metrics
```

## What is End-to-End Autonomous Driving?

Traditional autonomous driving stacks decompose the problem into sequential modules:
**Perception → Prediction → Planning → Control**

Each module has its own post-processing, hand-crafted interfaces, and potential for error accumulation.

End-to-end models aim to learn the mapping from raw sensor inputs to driving decisions in a more integrated fashion, reducing information loss between modules.

### Two-Step E2E Models

In two-step E2E models, perception and planning remain as **distinct sub-networks**, but they are trained jointly and the perception features flow **directly** into the planning module without hand-crafted post-processing:

```
Sensors → [Perception Network] → (learned features) → [Planning Network] → Trajectory
                                    ↑ no post-processing
                                    ↑ direct feature passing
```

**Key characteristics:**
- Perception still produces interpretable intermediate representations (BEV maps, object queries)
- But these representations are LEARNED features passed directly, not post-processed detections
- Joint end-to-end training allows gradients to flow from planning loss back to perception
- More interpretable than one-step models due to visible intermediate representations

**Examples:** UniAD, VAD, ST-P3

### One-Step E2E Models

One-step models map directly from raw sensor input to planning output with no exposed intermediate representation:

```
Sensors → [Single Neural Network] → Trajectory / Control
```

**Key characteristics:**
- No intermediate perception output
- Potentially learns optimal internal representations for driving
- Less interpretable but potentially more optimal
- Includes both traditional CNN/Transformer approaches and newer foundation model/LLM approaches

**Examples:** TransFuser, InterFuser, TCP (traditional); DriveVLM, GAIA-1, GenAD (foundation model)

### Foundation Model Paradigm (New)

A emerging approach applies the LLM training paradigm to driving:
1. **Pre-training**: Large-scale self-supervised learning on driving data (world models)
2. **Fine-tuning**: Task-specific adaptation for planning
3. **Reinforcement Learning**: Online improvement via reward signals

This includes vision-language models (DriveVLM), generative world models (GAIA-1), and multimodal foundation models adapted for driving.

## Planner Scorer

Because driving is inherently multi-modal (multiple valid behaviors exist for any scenario), a **planner scorer** evaluates and ranks candidate trajectories:

```
Candidate Trajectories → [Scorer Network] → Scores → Best Trajectory
```

The scorer considers: safety (collision risk), comfort (jerk, lateral acceleration), progress (efficiency), and rule compliance (lane keeping, speed limits).

## Datasets

| Dataset | Type | Size | Tasks |
|---------|------|------|-------|
| nuScenes | Real-world | 1000 scenes, 1.4M frames | Detection, Tracking, Prediction, Planning |
| CARLA | Simulation | Unlimited | Full driving stack |
| nuPlan | Real-world | 1500h driving | Planning, Scoring |
| Waymo Open | Real-world | 1150 scenes | Detection, Prediction, Planning |
| OpenScene | Real-world | 1M+ scenes | Occupancy, Planning |

## Getting Started

Each model folder contains:
- `README.md` — Model overview, architecture, and key results
- `docs/` — Detailed technical documentation
- `model.py` — PyTorch model implementation
- `train.py` — Training script
- `evaluate.py` — Evaluation script
- `config.py` — Configuration and hyperparameters
- `requirements.txt` — Dependencies

### Prerequisites

```bash
pip install torch torchvision
pip install mmdet3d mmcv-full  # For BEV-based models
pip install nuscenes-devkit    # For nuScenes dataset
```

## References

- [UniAD](https://arxiv.org/abs/2212.10156) - Planning-oriented Autonomous Driving (CVPR 2023 Best Paper)
- [VAD](https://arxiv.org/abs/2303.12077) - Vectorized Scene Representation (ICCV 2023)
- [ST-P3](https://arxiv.org/abs/2207.07601) - Spatial-Temporal Feature Learning (ECCV 2022)
- [TransFuser](https://arxiv.org/abs/2205.15997) - Multi-Modal Fusion Transformer (PAMI 2023)
- [InterFuser](https://arxiv.org/abs/2207.14024) - Safety-Enhanced Sensor Fusion (CoRL 2022)
- [TCP](https://arxiv.org/abs/2206.08129) - Trajectory-guided Control (NeurIPS 2022)
- [DriveVLM](https://arxiv.org/abs/2402.12289) - Vision-Language Driving Model (2024)
- [GAIA-1](https://arxiv.org/abs/2309.17080) - Generative World Model (2023)

## License

This repository is for research and educational purposes. Individual model implementations may have their own licenses — refer to the original papers and repositories.
