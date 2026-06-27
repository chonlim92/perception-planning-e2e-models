# VAD: Vectorized Scene Representation for Efficient Autonomous Driving

> Represent the driving scene as compact vectors instead of dense grids -- achieve the same planning quality at 2-5x the speed.

**Paper:** "VAD: Vectorized Scene Representation for Efficient Autonomous Driving"  
**Authors:** Bo Jiang, Shaoyu Chen, Qing Xu, Bencheng Liao, Jiajie Chen, et al.  
**Venue:** ICCV 2023  
**arXiv:** https://arxiv.org/abs/2303.12077  
**Code:** https://github.com/hustvl/VAD

---

## Table of Contents

1. [What is VAD?](#what-is-vad)
2. [Key Innovation: Vectorized Representation](#key-innovation-vectorized-representation)
3. [Architecture](#architecture)
4. [Key Concepts](#key-concepts)
5. [How It Works Step by Step](#how-it-works-step-by-step)
6. [Why Multiple Candidate Trajectories?](#why-multiple-candidate-trajectories)
7. [Our Implementation](#our-implementation)
8. [Running the Code](#running-the-code)
9. [Results](#results)
10. [Comparison with UniAD](#comparison-with-uniad)
11. [References](#references)

---

## What is VAD?

VAD (Vectorized Autonomous Driving) is a **two-step end-to-end** model for self-driving cars. It takes camera images as input and outputs a planned trajectory for the ego vehicle (the car running the model).

**"Two-step"** means the model has two distinct stages that are trained together end-to-end:

1. **Perception** -- Understand the scene: Where are other vehicles? Where are the lane boundaries?
2. **Planning** -- Decide what to do: What trajectory should our car follow for the next 3 seconds?

The key idea is that VAD does NOT represent the scene as a dense 2D grid (like a 200x200 pixel map). Instead, it represents everything as **vectors** -- compact polylines and motion arrows. This makes the model much faster while achieving similar or better planning quality.

**In plain English:** Instead of painting a full picture of the road, VAD draws a few important arrows and lines -- where cars are going, where the lanes are -- and uses those sketches to decide where to drive.

---

## Key Innovation: Vectorized Representation

### The Problem with Dense Grids

Traditional end-to-end models (like UniAD) represent the driving scene as a dense BEV (Bird's Eye View) grid -- essentially a 200x200 feature map covering the area around the car:

```
Dense BEV Grid (200 x 200 = 40,000 cells)
┌─────────────────────────────────────────────┐
│ . . . . . . . . . . . . . . . . . . . . . . │
│ . . . . . . . . . . . . . . . . . . . . . . │
│ . . . . ████ . . . . . . . . . . . . . . . │  <- car (uses ~20 cells)
│ . . . . ████ . . . . . . . . . . . . . . . │
│ . . . . . . . . . . . . . . . . . . . . . . │
│ . . ─────────────────────── . . . . . . . . │  <- lane (uses ~50 cells)
│ . . . . . . . . . . . . . . . . . . . . . . │
│ . . . . . . . . . . . . . . . . . . . . . . │
│ . . . . . . . . . . . . . . . . . . . . . . │
│ . . . . . . . . . . . . . . . . . . . . . . │
└─────────────────────────────────────────────┘
 Problem: 99% of cells are empty/redundant!
          The planner must attend to ALL 40,000 cells.
```

### VAD's Solution: Vectors

VAD represents the same scene using a small number of vectors:

```
Vectorized Representation (~400 vectors total)
                                                    
  Agent vectors (motion arrows):                    
     ╲                                              
      ╲──→  Agent 1: position + velocity + future   
                                                    
      ←──╱  Agent 2: position + velocity + future   
          ╱                                         
                                                    
  Map vectors (polylines):                          
     ┌─·─·─·─·─·─·─·─·─·─·─·─·─┐                 
     Lane boundary: 20 ordered (x,y) points         
                                                    
     └─·─·─·─·─·─·─·─·─·─·─·─·─┘                 
     Road edge: 20 ordered (x,y) points             
                                                    
  Ego vectors (candidate trajectories):             
     ───●───●───●───●───●───●                       
     Trajectory: 6 future waypoints (x,y)           
```

### Why Vectors Win

| Aspect | Dense Grid (UniAD) | Vectors (VAD) |
|--------|:------------------:|:-------------:|
| Scene representation | 200x200 = 40,000 cells | ~400 vectors |
| Information density | Low (mostly empty space) | High (every vector matters) |
| Planner attention cost | O(40,000) | O(400) -- 100x fewer tokens |
| Interpretable? | Hard to visualize | Easy: draw arrows and lines |
| Speed (FPS) | 1.8 | 4.5 - 8.4 |

---

## Architecture

```
                        VAD Architecture (Full Pipeline)
 ═══════════════════════════════════════════════════════════════════

 INPUT: 6 surround-view camera images
 ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐
 │Front │ │F-Left│ │F-Right│ │Rear │ │R-Left│ │R-Right│
 │ Cam  │ │ Cam  │ │ Cam  │ │ Cam  │ │ Cam  │ │ Cam  │
 └──┬───┘ └──┬───┘ └──┬────┘ └──┬───┘ └──┬───┘ └──┬────┘
    │        │        │         │        │        │
    └────────┴────────┴─────────┴────────┴────────┘
                          │
                          ▼
             ┌────────────────────────┐
             │   Image Backbone +     │
             │   BEV Encoder          │    Step 0: Extract features
             │   (ResNet + BEVFormer) │    and lift to bird's eye view
             └───────────┬────────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │   BEV Feature Tokens │    (B, HW, D) = spatial tokens
              │   (flattened grid)   │    e.g., 2500 tokens x 256 dim
              └──────────┬───────────┘
                         │
         ┌───────────────┴───────────────┐
         │                               │
         ▼                               ▼
┌─────────────────────┐       ┌─────────────────────┐
│  AGENT DECODER      │       │  MAP DECODER        │
│                     │       │                     │    Step 1:
│  300 agent queries  │       │  100 map queries    │    PERCEPTION
│  → Transformer      │       │  → Transformer      │    (vectorized)
│    Decoder          │       │    Decoder          │
│                     │       │                     │
│  Outputs:           │       │  Outputs:           │
│  • class labels     │       │  • polyline type    │
│  • motion vectors   │       │  • polyline points  │
│    (K modes x T x 2)│       │    (20 pts x 2)    │
│  • mode probs       │       │                     │
│  • agent features   │       │  • map features     │
└──────────┬──────────┘       └──────────┬──────────┘
           │                             │
           └──────────────┬──────────────┘
                          │
                          ▼
             ┌────────────────────────────┐
             │     EGO PLANNING HEAD      │
             │                            │    Step 2:
             │  K=6 learnable ego queries │    PLANNING
             │                            │
             │  Cross-attention to:       │
             │    • Agent features ───────│──→ "Where are others going?"
             │    • Map features ─────────│──→ "Where are the lanes?"
             │                            │
             │  Each query → 1 candidate  │
             │  trajectory (6 waypoints)  │
             │                            │
             │  Scoring head → pick best  │
             └─────────────┬──────────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │   OUTPUT:            │
                │   Best trajectory    │
                │   (6 waypoints,      │
                │    0.5s intervals,   │
                │    3s into future)   │
                └──────────────────────┘
```

---

## Key Concepts

### 1. Vectorized Representation

**What it means:** Instead of storing scene information in a dense pixel grid, we store it as a collection of compact geometric primitives -- vectors and polylines.

- **Agent vector:** A detected vehicle/pedestrian is represented by its position (x, y), heading, and a sequence of future displacement vectors (dx, dy) at each timestep.
- **Map vector (polyline):** A lane boundary or road edge is represented as an ordered sequence of 20 (x, y) control points -- like a connect-the-dots line.

**Why this works:** On a real road, the useful information is sparse. A 60m x 60m area might contain 5-10 vehicles and 10-20 lane segments. That is ~400 vectors vs. 40,000 grid cells.

### 2. Ego Queries (Candidate Trajectories)

The planning head has **K = 6 learnable ego queries**. Each is a learned embedding vector that the model trains to specialize in a particular driving behavior:

```
Ego Query 1 → tends to produce "go straight" trajectories
Ego Query 2 → tends to produce "turn left" trajectories
Ego Query 3 → tends to produce "slow down" trajectories
Ego Query 4 → tends to produce "lane change right" trajectories
...etc.
```

Each ego query cross-attends to agent features ("where are other cars going?") and map features ("where are the lanes?") to produce one candidate trajectory.

### 3. Scoring Head

After producing K=6 candidate trajectories, a **scoring head** evaluates each one and assigns a quality score:

```
Candidate 1: ───●───●───●───●───●───●     Score: 0.12
Candidate 2: ───●───●───●──●──●──●        Score: 0.05
Candidate 3: ───●───●───●───●───●───●     Score: 0.83  ← WINNER
Candidate 4: ──●──●──●──●──●──●           Score: 0.02
Candidate 5: ───●────●────●────●──●──●    Score: 0.15
Candidate 6: ──●──●───●───●────●───●      Score: 0.01
```

At inference time, the model picks the trajectory with the highest score. During training, the scoring head learns to assign high scores to trajectories that are closest to the ground-truth expert trajectory.

### 4. Map Polylines

Map elements are predicted as **polylines** -- ordered sequences of points:

```
Lane divider (polyline with 20 points):

    ·─────·─────·─────·─────·─────·─────·─────·─────·─────·
    p1    p2    p3    p4    p5    p6    p7    p8    p9    p10...

Each point is an (x, y) coordinate in the ego-vehicle coordinate frame.

Three map classes:
  1. Lane dividers (dashed lines between lanes)
  2. Road boundaries (solid edges of the road)
  3. Pedestrian crossings
```

---

## How It Works Step by Step

Here is what happens when the model receives a new set of camera images:

### Step 1: BEV Feature Extraction

Six cameras capture images around the car. A CNN backbone (ResNet-50) extracts image features, and a BEV encoder (BEVFormer-style) lifts them into a bird's-eye-view feature map. This BEV map is then flattened into a sequence of tokens.

```
6 images (224x400 each) → Backbone → BEV encoder → 2500 tokens (each 256-dim)
```

### Step 2: Agent Decoding (Vectorized)

300 **agent queries** (learned embeddings) attend to the BEV tokens via a Transformer decoder. Each query tries to detect one agent and predict its future motion:

- Classification: What type of agent? (car, truck, pedestrian, cyclist, etc.)
- Motion prediction: 6 possible future trajectories (modes), each with 12 timesteps of (dx, dy) displacements
- Mode probabilities: Which of the 6 modes is most likely?

### Step 3: Map Decoding (Vectorized)

100 **map queries** attend to the same BEV tokens. Each query tries to detect one map element:

- Classification: What type? (lane divider, road boundary, crossing)
- Polyline regression: 20 ordered (x, y) points defining the shape

### Step 4: Ego Planning

6 **ego queries** cross-attend to the agent features and map features (NOT to the raw BEV -- this is the efficiency gain). Each ego query produces:

- A candidate trajectory: 6 waypoints at 0.5s intervals (covering 3s)
- A quality score

### Step 5: Trajectory Selection

The scoring head picks the trajectory with the highest score. This becomes the final output -- the planned path for the ego vehicle.

---

## Why Multiple Candidate Trajectories?

### The Multi-Modal Planning Problem

Driving is inherently **multi-modal** -- there are often multiple valid actions in the same situation:

```
Scenario: Approaching an intersection with a green light

                      │    │
                      │    │
         Valid        │    │        Valid
         Option A:    │    │        Option B:
         Turn left    │    │        Go straight
              ╲       │    │       │
               ╲      │    │       │
                ╲     │    │       │
                 ╲    │    │       ▼
    ──────────────────┼────┼──────────────────
                      │    │
              EGO ──► │    │
                      │    │
    ──────────────────┼────┼──────────────────
                      │    │
                      │    │

Both options are perfectly safe and valid!
```

If the model could only produce a single trajectory, it would be forced to **average** the two options, resulting in a trajectory that goes neither left nor straight -- possibly driving into a curb.

### The Solution: Generate K Candidates, Then Score

By generating K=6 diverse candidates, the model can represent multiple modes:

```
Query 1 → "turn left" trajectory      Score: 0.85  ← Best
Query 2 → "go straight" trajectory    Score: 0.72
Query 3 → "slow down" trajectory      Score: 0.15
Query 4 → "turn right" trajectory     Score: 0.03
Query 5 → "lane change" trajectory    Score: 0.01
Query 6 → "hard brake" trajectory     Score: 0.01
```

The scoring head then resolves the ambiguity based on context (e.g., the navigation command says "turn left at the next intersection").

### Training: Winner-Take-All

During training, only the candidate closest to the ground truth gets a regression gradient (called "winner-take-all" or WTA). This encourages each query to specialize in different maneuvers rather than all converging to the same trajectory.

---

## Our Implementation

This is a **simplified educational implementation** of VAD (16.6M parameters), designed to illustrate the architecture clearly:

### What We Include

- Full VAD pipeline: BEV encoder, Agent decoder, Map decoder, Ego planning head
- Vectorized agent prediction with multi-modal motion forecasting (6 modes, 12 timesteps)
- Vectorized map prediction with polyline regression (20 points per element)
- K=6 ego queries with scoring head and winner-take-all training
- Cross-attention from ego queries to agent/map features (the key efficiency insight)
- Complete loss computation (planning L1 + scoring BCE)

### What We Simplify

| Component | Original VAD | Our Implementation |
|-----------|--------------|-------------------|
| BEV encoder | BEVFormer with temporal fusion | Simple CNN + AdaptivePool |
| Backbone | ResNet-50 + FPN | Lightweight Conv layers |
| Temporal fusion | Multi-frame BEV aggregation | Single-frame only |
| Agent decoder | Deformable attention | Standard Transformer decoder |
| Map decoder | Deformable attention | Standard Transformer decoder |
| Vectorized constraint loss | Explicit scene consistency loss | Not included |
| Training data | Full nuScenes pipeline | Random tensors for demo |
| Parameters | ~40M (VAD-Base) | 16.6M |

### Why Simplify?

The goal is to **understand the architecture**, not reproduce the paper's results. The core insight -- vectorized representation for efficient planning -- is preserved. If you understand how ego queries cross-attend to agent/map vectors instead of a dense BEV grid, you understand VAD.

---

## Running the Code

### Prerequisites

```bash
pip install torch  # PyTorch >= 1.10
```

### Run the Demo

```bash
cd two_step_e2e/VAD
python model.py
```

### Expected Output

```
VAD: Vectorized Autonomous Driving Demo
=============================================
Parameters: 16,615,xxx

Outputs:
  Agent motion vectors: torch.Size([2, 300, 6, 12, 2])
  Map polylines: torch.Size([2, 100, 20, 2])
  Candidate trajectories: torch.Size([2, 6, 6, 2])
  Scores: torch.Size([2, 6])
  Best trajectory: torch.Size([2, 6, 2])
  Best indices: [X, X]

  Loss: X.XXXX (plan=X.XXXX, score=X.XXXX)
```

### Understanding the Output Shapes

| Output | Shape | Meaning |
|--------|-------|---------|
| Agent motion vectors | (B, 300, 6, 12, 2) | Batch, 300 agents, 6 modes, 12 timesteps, (dx,dy) |
| Map polylines | (B, 100, 20, 2) | Batch, 100 map elements, 20 points, (x,y) |
| Candidate trajectories | (B, 6, 6, 2) | Batch, 6 candidates, 6 waypoints, (x,y) |
| Scores | (B, 6) | Batch, quality score for each of 6 candidates |
| Best trajectory | (B, 6, 2) | Batch, 6 waypoints of selected best trajectory |

---

## Results

Results from the original paper on nuScenes:

### Planning Performance

| Model | L2 Error (1s) | L2 Error (2s) | L2 Error (3s) | Collision Rate |
|-------|:-------------:|:-------------:|:-------------:|:--------------:|
| VAD-Tiny | 0.20m | 0.54m | 1.01m | 0.31% |
| VAD-Base | 0.17m | 0.51m | 0.97m | 0.25% |
| UniAD | 0.22m | 0.56m | 1.03m | 0.31% |
| ST-P3 | 1.33m | 2.11m | 2.90m | 0.23% |

### Efficiency

| Model | FPS | GPU Memory | Backbone |
|-------|:---:|:----------:|----------|
| VAD-Tiny | 8.4 | ~8 GB | ResNet-50 |
| VAD-Base | 4.5 | ~16 GB | ResNet-50 |
| UniAD | 1.8 | ~24 GB | ResNet-101 |

**Key takeaway:** VAD-Base achieves slightly better planning accuracy than UniAD while running 2.5x faster. VAD-Tiny matches UniAD's accuracy at 4.7x the speed.

---

## Comparison with UniAD

VAD and UniAD are both two-step end-to-end models trained on nuScenes, but they differ significantly in design philosophy:

| Aspect | UniAD (CVPR 2023) | VAD (ICCV 2023) |
|--------|:------------------:|:----------------:|
| **Core idea** | Unify ALL driving tasks | Vectorize the scene for efficiency |
| **Scene representation** | Dense BEV grid (200x200) | Sparse vectors (~400 elements) |
| **Perception tasks** | Detection + Tracking + Mapping + Occupancy | Agent detection + Map prediction |
| **Planner input** | Dense BEV + all task features | Sparse agent/map vectors only |
| **Planner architecture** | GRU (autoregressive, 1 trajectory) | K parallel queries + scoring |
| **Multi-modal planning** | No (single output) | Yes (K=6 candidates) |
| **Speed (FPS)** | 1.8 | 4.5 - 8.4 |
| **Parameters** | ~100M+ | ~40M (Base) |
| **Training** | 3-stage curriculum | End-to-end single stage |
| **Occupancy prediction** | Yes | No |
| **Tracking** | Yes (track queries persist) | No (per-frame detection) |
| **Planning L2 (3s)** | 1.03m | 0.97m |
| **Key strength** | Comprehensive scene understanding | Speed + simplicity |
| **Key weakness** | Slow inference (dense attention) | Less rich scene model |

### The Fundamental Trade-off

```
UniAD approach:            VAD approach:
 "Understand everything     "Understand what matters
  then plan"                 for planning, efficiently"

  ┌─────────────┐            ┌─────────────┐
  │Dense 200x200│            │~300 agent   │
  │BEV features │            │ vectors     │
  │(40,000 cells)│           │~100 map     │
  │             │            │ vectors     │
  │ALL info here│            │(400 total)  │
  └──────┬──────┘            └──────┬──────┘
         │                          │
    ┌────▼────┐              ┌──────▼──────┐
    │ Planner │              │   Planner   │
    │attends  │              │  attends    │
    │to 40,000│              │  to 400     │
    │ tokens  │              │  vectors    │
    └─────────┘              └─────────────┘
    Slow but thorough         Fast and focused
```

VAD showed that you do not need to predict everything (occupancy, dense tracks) to plan well. The vectorized representation captures enough for safe planning while being 100x more compact for the planner to reason over.

---

## References

1. **VAD Paper:** Bo Jiang et al., "VAD: Vectorized Scene Representation for Efficient Autonomous Driving," ICCV 2023. [arXiv:2303.12077](https://arxiv.org/abs/2303.12077)

2. **VAD GitHub:** https://github.com/hustvl/VAD

3. **UniAD Paper:** Yihan Hu et al., "Planning-oriented Autonomous Driving," CVPR 2023. [arXiv:2212.10156](https://arxiv.org/abs/2212.10156)

4. **BEVFormer:** Zhiqi Li et al., "BEVFormer: Learning Bird's-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers," ECCV 2022.

5. **nuScenes Dataset:** Holger Caesar et al., "nuScenes: A multimodal dataset for autonomous driving," CVPR 2020.

---

## Files

```
VAD/
├── README.md       # This file
└── model.py        # Simplified VAD model (16.6M params)
```
