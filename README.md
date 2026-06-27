# End-to-End Perception + Planning Models for Autonomous Driving

> A comprehensive, beginner-friendly research repository covering state-of-the-art end-to-end (E2E) models for autonomous driving. Includes detailed explanations, architecture diagrams, and **working PyTorch code** you can run.

---

## Table of Contents

- [What is This Repository?](#what-is-this-repository)
- [Background: How Self-Driving Cars Work](#background-how-self-driving-cars-work)
- [Repository Structure](#repository-structure)
- [The Two Types of E2E Models](#the-two-types-of-e2e-models)
- [The Foundation Model Revolution](#the-foundation-model-revolution)
- [Planner Scorer](#planner-scorer-why-we-need-it)
- [Getting Started](#getting-started)
- [Model Comparison](#model-comparison)
- [Datasets](#datasets)
- [References](#references)

---

## What is This Repository?

This repository is a **learning resource and reference implementation** for engineers working on end-to-end autonomous driving. It contains:

1. **Detailed explanations** of each model's architecture (with diagrams)
2. **Working PyTorch code** that you can run to understand how the models work
3. **Training scripts** showing how to train these models
4. **Technical documentation** explaining the theory behind each approach

Each model folder is self-contained — you can go to any model and understand it independently.

---

## Background: How Self-Driving Cars Work

### The Traditional Approach (Modular Pipeline)

Traditional self-driving cars break the problem into separate modules:

```
Camera/LiDAR ──> [1. Perception] ──> [2. Prediction] ──> [3. Planning] ──> [4. Control]
                  "What's around me?"   "Where will       "What should    "Steer/Gas/
                  (detect cars,          they go?"         I do?"          Brake"
                   pedestrians,          (predict          (plan a
                   lanes)                futures)           path)
```

**Problem:** Each module is trained separately. Errors accumulate between modules. Information is lost at each interface (e.g., perception outputs bounding boxes but loses uncertainty information).

### The End-to-End Approach

End-to-end models learn the ENTIRE pipeline as ONE neural network:

```
Camera/LiDAR ──> [Single Neural Network] ──> Driving Decision
                  (trained end-to-end)
```

**Advantage:** No information loss between modules. The network learns optimal internal representations. Gradients flow from the driving decision all the way back to the sensor features.

---

## Repository Structure

```
perception-planning-e2e-models/
│
├── README.md                          # You are here
│
├── two_step_e2e/                      # TYPE 1: Two-Step E2E Models
│   ├── README.md                      # Overview of two-step approach
│   ├── UniAD/                         # CVPR 2023 Best Paper
│   │   ├── README.md                  # Architecture & paper details
│   │   ├── model.py                   # PyTorch implementation
│   │   ├── train.py                   # Training pipeline (multi-task, 3-stage)
│   │   └── config.py                  # Hyperparameters
│   ├── VAD/                           # ICCV 2023
│   │   ├── README.md
│   │   ├── model.py
│   │   └── train.py                   # Training (winner-take-all, multi-modal)
│   └── ST-P3/                         # ECCV 2022
│       ├── README.md
│       ├── model.py
│       └── train.py                   # Training (BEV seg + occupancy + planning)
│
├── one_step_e2e/                      # TYPE 2: One-Step E2E Models
│   ├── README.md                      # Overview of one-step approach
│   │
│   │   # --- Traditional Deep Learning ---
│   ├── TransFuser/                    # CVPR 2022 / PAMI 2023
│   │   ├── README.md
│   │   ├── model.py
│   │   └── train.py                   # Training (waypoint + BEV + speed)
│   ├── InterFuser/                    # CoRL 2022
│   │   ├── README.md
│   │   ├── model.py
│   │   └── train.py                   # Training (density map + safety score)
│   ├── TCP/                           # NeurIPS 2022
│   │   ├── README.md
│   │   ├── model.py
│   │   └── train.py                   # Training (dual-branch + adaptive fusion)
│   │
│   │   # --- Foundation Model / LLM-like (NEW PARADIGM) ---
│   ├── DriveVLM/                      # 2024 - Vision Language Model
│   │   ├── README.md
│   │   ├── model.py
│   │   └── train.py                   # 3-stage: pretrain → finetune → RL
│   ├── GAIA-1/                        # Wayve 2023 - World Model
│   │   ├── README.md
│   │   ├── model.py
│   │   └── train.py                   # 3-phase: tokenizer → world model → planner
│   └── GenAD/                         # 2024 - Diffusion-based
│       ├── README.md
│       ├── model.py
│       └── train.py                   # DDPM training + scorer + CFG
│
└── planner_scorer/                    # Trajectory Scoring & Selection
    ├── README.md                      # Overview
    ├── requirements.txt               # Dependencies
    ├── docs/
    │   └── technical_overview.md      # Theory & math
    ├── classical/                     # Rule-based scoring
    │   ├── cost_function.py           # Weighted multi-criteria scorer
    │   └── safety_checker.py          # TTC, RSS, collision checks
    └── learned/                       # Neural network scoring
        ├── mlp_scorer.py              # Simple MLP scorer
        ├── transformer_scorer.py      # Attention-based scorer
        ├── train.py                   # Training pipeline
        └── config.py                  # Hyperparameters
```

---

## The Two Types of E2E Models

### Type 1: Two-Step E2E (Perception features -> Planning)

**What it is:** The model has two distinct sub-networks (perception + planning), but they are connected directly — perception features flow into planning WITHOUT any hand-crafted post-processing in between.

```
                    Traditional Modular (NOT end-to-end):
Cameras ──> [Perception] ──> Bounding Boxes ──> NMS ──> Tracking ──> [Planning]
                              ↑ post-processing (hand-crafted, information loss)

                    Two-Step E2E:
Cameras ──> [Perception Network] ──> Learned Features ──> [Planning Network] ──> Trajectory
                                     ↑ DIRECT connection (no post-processing)
                                     ↑ Gradients flow back through both networks
```

**Key Properties:**
- You can still "see" what perception is doing (interpretable intermediate)
- Both networks are trained JOINTLY (end-to-end)
- Planning loss improves perception (gradients flow backward)

**Models:** UniAD, VAD, ST-P3

### Type 2: One-Step E2E (Sensors -> Trajectory, no intermediate)

**What it is:** A single network takes raw sensor data and directly outputs driving actions. No explicit perception output.

```
Cameras + LiDAR ──> [One Big Network] ──> Steering/Waypoints
                     (internal features
                      are NOT interpretable)
```

**Key Properties:**
- Maximum information preservation (nothing is thrown away)
- Network learns whatever internal representation is best for driving
- Less interpretable (harder to debug)

**Models:** TransFuser, InterFuser, TCP, DriveVLM, GAIA-1, GenAD

---

## The Foundation Model Revolution

### What is it?

A new approach that applies the same training paradigm as GPT/ChatGPT to autonomous driving:

| ChatGPT Training | Driving Equivalent |
|:---:|:---:|
| Pre-train on internet text | Pre-train on millions of driving videos |
| Fine-tune on Q&A dialogues | Fine-tune on driving decisions |
| RLHF (human feedback) | RL from driving rewards (safety, comfort) |
| Chat with reasoning | Drive with chain-of-thought reasoning |

### Three Sub-Approaches

1. **Vision-Language Models (DriveVLM)**
   - Uses a model like GPT-4V that can see AND reason
   - Input: camera images + text command ("turn left at next intersection")
   - Output: trajectory + natural language explanation ("Slowing down because pedestrian ahead")

2. **World Models (GAIA-1)**
   - Learns to imagine "what happens if I do X?"
   - Plans by mentally simulating different actions and picking the best outcome
   - Like how humans think: "If I turn left now, will I hit that car? No, it's far enough."

3. **Generative Models (GenAD)**
   - Uses diffusion models (like image generators) but for trajectories
   - Generates MANY possible driving paths (not just one)
   - Then scores and picks the best one

### Why It Matters

Traditional E2E models are trained with supervised learning only (copy the expert). Foundation models go beyond:
- **Pre-training** gives general understanding of the visual world
- **Fine-tuning** adapts to driving specifically
- **RL** goes BEYOND expert performance (learns things the expert didn't teach)

---

## Planner Scorer: Why We Need It

### The Problem

Driving is NOT a single-answer problem. Consider this scenario:

```
You're driving and there's a stopped car ahead. Valid options:

  Option A: Change lanes to the left     [Safe if left lane is clear]
  Option B: Change lanes to the right    [Safe if right lane is clear]
  Option C: Slow down and wait           [Always safe, but slow]
  Option D: Honk and wait                [Rude but legal]
```

ALL of these are correct! A model that tries to predict ONE answer will average between them, potentially outputting an INVALID trajectory (driving into the obstacle).

### The Solution: Generate Multiple + Score

```
Step 1: Generate K candidate trajectories (K = 64 typically)
Step 2: Score each one (safety, comfort, progress, rules)
Step 3: Pick the highest-scoring trajectory

[Planner] ──> 64 candidate paths ──> [Scorer] ──> Best path ──> Execute
```

### Two Approaches to Scoring

**Classical (Rule-based):** Explicit formulas for safety, comfort, etc.
```python
score = -5.0 * collision_risk - 1.5 * jerk - 1.0 * lateral_deviation + 2.0 * progress
```

**Learned (Neural Network):** Train a model to predict trajectory quality
```python
score = NeuralNetwork(trajectory, scene_context)  # trained on expert data
```

---

## Getting Started

### Prerequisites

```bash
# Python 3.8+ required
pip install torch numpy scipy matplotlib shapely tqdm pyyaml
```

### Run Any Model Demo

Each model has a `demo()` function you can run directly:

```bash
# Two-step models
python two_step_e2e/UniAD/model.py        # UniAD demo
python two_step_e2e/VAD/model.py          # VAD demo
python two_step_e2e/ST-P3/model.py        # ST-P3 demo

# One-step models (traditional)
python one_step_e2e/TransFuser/model.py   # TransFuser demo
python one_step_e2e/InterFuser/model.py   # InterFuser demo
python one_step_e2e/TCP/model.py          # TCP demo

# One-step models (foundation model paradigm)
python one_step_e2e/DriveVLM/model.py     # Vision-Language Model demo
python one_step_e2e/GAIA-1/model.py       # World Model demo
python one_step_e2e/GenAD/model.py        # Diffusion Model demo

# Planner Scorer
python planner_scorer/classical/cost_function.py --demo
python planner_scorer/classical/safety_checker.py
python planner_scorer/learned/mlp_scorer.py
python planner_scorer/learned/transformer_scorer.py
```

### Train Any Model

Every model has a complete `train.py` with synthetic data (no external datasets needed):

```bash
# Two-step models
python two_step_e2e/UniAD/train.py --epochs 5 --batch_size 2
python two_step_e2e/VAD/train.py --epochs 5 --batch_size 2
python two_step_e2e/ST-P3/train.py --epochs 5 --batch_size 2

# One-step models (traditional)
python one_step_e2e/TransFuser/train.py --epochs 5 --batch_size 1
python one_step_e2e/InterFuser/train.py --epochs 5 --batch_size 2
python one_step_e2e/TCP/train.py --epochs 1 --batch-size 4

# One-step models (foundation model paradigm)
python one_step_e2e/DriveVLM/train.py --epochs 5 --batch_size 1
python one_step_e2e/GAIA-1/train.py --phase tokenizer --epochs_tokenizer 5
python one_step_e2e/GenAD/train.py --epochs 5 --batch_size 4

# Planner Scorer
python planner_scorer/learned/train.py --model mlp --loss combined --epochs 50
python planner_scorer/learned/train.py --model transformer --loss contrastive --epochs 50
```

Each training script includes:
- Synthetic dataset (runs without external data)
- Full loss functions from the papers
- Training + validation loops with metrics
- Checkpoint save/load/resume
- Mixed precision (AMP) support
- Learning rate scheduling

### Attribution System

All training scripts clearly mark what comes from the original papers vs. our implementation:

| Tag | Meaning |
|:---|:---|
| `[FROM PAPER]` | Algorithm/loss/technique directly described in the original paper |
| `[SELF-IMPLEMENTED]` | Our implementation of concepts not fully detailed in the paper |
| `[SIMPLIFIED]` | Paper's approach simplified for clarity (noted what was changed) |

This helps you distinguish research contributions from engineering decisions.

---

## Model Comparison

### Two-Step E2E Models

| Model | Year | Venue | Key Idea | Planning L2 (3s) | Collision Rate |
|:---:|:---:|:---:|:---|:---:|:---:|
| ST-P3 | 2022 | ECCV | Spatial-temporal BEV + GRU planner | 2.13m | 1.27% |
| UniAD | 2023 | CVPR | Unified full-stack (Best Paper) | 1.03m | 0.31% |
| VAD | 2023 | ICCV | Vectorized (efficient, fast) | 0.97m | 0.25% |

### One-Step E2E Models (CARLA Benchmark)

| Model | Year | Venue | Key Idea | Driving Score |
|:---:|:---:|:---:|:---|:---:|
| TransFuser | 2022 | CVPR | Multi-scale transformer fusion | 54.52 |
| InterFuser | 2022 | CoRL | Safety-enhanced with density maps | 68.31 |
| TCP | 2022 | NeurIPS | Trajectory-guided control (dual branch) | 75.14 |

### Foundation Model Approaches

| Model | Year | Type | Parameters | Real-time? |
|:---:|:---:|:---|:---:|:---:|
| DriveVLM | 2024 | Vision-Language Model | ~7B | No (1-2 FPS) |
| GAIA-1 | 2023 | World Model | ~9B | No (5 FPS) |
| GenAD | 2024 | Diffusion Model | ~500M | Near (depends on steps) |

---

## Datasets

| Dataset | Type | What It Contains | Used By |
|:---|:---:|:---|:---|
| **nuScenes** | Real-world | 1000 scenes, 6 cameras, LiDAR, annotations | UniAD, VAD, ST-P3 |
| **CARLA** | Simulator | Unlimited synthetic driving data | TransFuser, InterFuser, TCP |
| **nuPlan** | Real-world | 1500 hours, planning-focused | Planner Scorer |
| **Waymo Open** | Real-world | 1150 scenes, high quality | General research |

---

## Glossary for Beginners

| Term | Meaning |
|:---|:---|
| **BEV** | Bird's Eye View — looking at the road from above (like a drone) |
| **E2E** | End-to-End — learning the whole pipeline as one system |
| **Waypoints** | Points along the planned path (x, y coordinates at future times) |
| **Trajectory** | The full planned path = sequence of waypoints over time |
| **Multi-modal** | Multiple valid options exist (not one single right answer) |
| **Imitation Learning** | Training by copying an expert driver's behavior |
| **BEVFormer** | A popular method to create BEV features from camera images |
| **nuScenes** | A large real-world driving dataset with 3D annotations |
| **CARLA** | An open-source driving simulator for training/testing |
| **PID Controller** | Simple controller that converts waypoints to steer/gas/brake |
| **VQ-VAE** | A type of autoencoder that uses discrete codes (tokens) |
| **Diffusion Model** | A generative model that creates data by removing noise step-by-step |
| **Foundation Model** | A large pre-trained model adapted to many tasks (like GPT) |
| **RLHF/RL** | Reinforcement Learning — learning by trial and error with rewards |

---

## How to Read This Repository

**If you're completely new to autonomous driving:**
1. Read this README first (you're doing it!)
2. Start with `two_step_e2e/UniAD/README.md` — it's the most well-known model
3. Then read `one_step_e2e/TransFuser/README.md` — simpler one-step model
4. Then explore `planner_scorer/` — understand trajectory scoring

**If you want to understand the new foundation model paradigm:**
1. Read `one_step_e2e/DriveVLM/README.md` — VLM approach
2. Read `one_step_e2e/GAIA-1/README.md` — World Model approach
3. Read `one_step_e2e/GenAD/README.md` — Generative approach

**If you want to run code:**
1. Install PyTorch: `pip install torch`
2. Run any `model.py` file — each has a `demo()` function
3. Try the planner scorer training: `python planner_scorer/learned/train.py`

---

## References

### Papers
- [UniAD](https://arxiv.org/abs/2212.10156) — Planning-oriented Autonomous Driving (CVPR 2023 Best Paper)
- [VAD](https://arxiv.org/abs/2303.12077) — Vectorized Scene Representation (ICCV 2023)
- [ST-P3](https://arxiv.org/abs/2207.07601) — Spatial-Temporal Feature Learning (ECCV 2022)
- [TransFuser](https://arxiv.org/abs/2205.15997) — Multi-Modal Fusion Transformer (PAMI 2023)
- [InterFuser](https://arxiv.org/abs/2207.14024) — Safety-Enhanced Sensor Fusion (CoRL 2022)
- [TCP](https://arxiv.org/abs/2206.08129) — Trajectory-guided Control (NeurIPS 2022)
- [DriveVLM](https://arxiv.org/abs/2402.12289) — Vision-Language Driving Model (2024)
- [GAIA-1](https://arxiv.org/abs/2309.17080) — Generative World Model (2023)
- [GenAD](https://arxiv.org/abs/2402.11502) — Generative Autonomous Driving (2024)

### Official Code Repositories
- UniAD: https://github.com/OpenDriveLab/UniAD
- VAD: https://github.com/hustvl/VAD
- ST-P3: https://github.com/OpenDriveLab/ST-P3
- TransFuser: https://github.com/autonomousvision/transfuser
- InterFuser: https://github.com/opendilab/InterFuser
- TCP: https://github.com/OpenDriveLab/TCP

---

## Quality Assurance: Expert ML Review

All 13 model implementations have been reviewed by a panel of 5 independent ML expert agents, covering:

- **Architecture correctness** — attention mechanisms, residual connections, positional encodings
- **Training dynamics** — loss functions, optimizer configuration, LR scheduling, gradient flow
- **Numerical stability** — NaN/Inf prevention, mixed-precision safety, device compatibility
- **Safety & planning** — trajectory feasibility, collision checking, RSS compliance
- **Runtime validation** — forward/backward pass testing on all models

### Key Improvements Made

| Issue | Severity | Model | Fix |
|-------|----------|-------|-----|
| LR scheduler bulk-stepped at epoch end | Critical | VAD | Per-iteration stepping inside training loop |
| Positional embedding buffer overflow | High | GAIA-1 | Sized for `(tokens_per_frame + 1)` per frame |
| Division-by-zero in time embedding | High | GenAD | Safe division with `max(half_dim - 1, 1)` |
| Non-differentiable collision loss | High | UniAD | Replaced `.long()` indexing with `F.grid_sample()` |
| Planner has zero gradient flow | High | GAIA-1 | Added action log-probs for policy gradient |
| Scheduler collapses per-group LRs | High | DriveVLM | Per-group `lr_scale` preservation |
| Ranking loss executes only once | High | Planner Scorer | Vectorized pairwise margin computation |
| RSS safe distance can be negative | High | Safety Checker | Clamped to `max(0, d_safe)` |

**Full validation report:** [`docs/expert_validation_report.md`](docs/expert_validation_report.md)

---

## License

This repository is for **research and educational purposes**. Individual model implementations may have their own licenses — refer to the original papers and repositories.
