# Planner Scorer: Trajectory Scoring and Selection

> Given multiple candidate driving paths, how do you pick the best one? This module implements both classical (rule-based) and learned (neural network) approaches to trajectory scoring.

---

## Table of Contents

- [What is Trajectory Scoring?](#what-is-trajectory-scoring)
- [Why We Need It: The Multi-Modal Problem](#why-we-need-it-the-multi-modal-problem)
- [The Generate-Score-Select Pipeline](#the-generate-score-select-pipeline)
- [Classical vs Learned Approaches](#classical-vs-learned-approaches)
- [Module Structure](#module-structure)
- [Key Concepts](#key-concepts)
- [How Scoring Integrates with Planners](#how-scoring-integrates-with-planners)
- [Getting Started](#getting-started)
- [Safety Considerations](#safety-considerations)

---

## What is Trajectory Scoring?

A **trajectory scorer** takes a candidate driving path and assigns it a numerical score indicating how good that path is. The higher the score, the better the trajectory.

```
Input:  A candidate trajectory (sequence of x,y positions over 3-6 seconds)
        + The current scene context (other cars, lanes, speed limit)

Output: A single number (score) representing trajectory quality
```

In practice, you generate MANY candidates (typically 64-256) and pick the highest-scoring one.

---

## Why We Need It: The Multi-Modal Problem

### The Problem

Driving often has MULTIPLE correct answers. Consider this scenario:

```
                    ┌───────────────────────────┐
                    │     STOPPED TRUCK          │
                    │     ████████████           │
                    │                            │
    ═══════════════════════════════════════════════  (your lane)
                    │                            │
    ═══════════════════════════════════════════════  (left lane)
                    │                            │
    YOU ──>         │                            │
    [CAR]           │                            │
                    └───────────────────────────┘

    Valid options:
    A) Change lanes to the left     [Good if left lane is clear]
    B) Slow down and wait           [Always safe, but slow]
    C) Change lanes to the right    [Good if right lane is clear]
```

### Why This Breaks Simple Models

If you train a model with L2 loss (mean squared error) to predict ONE trajectory:
- Training data has examples of both "go left" and "slow down"  
- The network AVERAGES them
- Average of "go left" and "slow down" = "go slightly left at full speed" = **CRASH**

```
    Expert A chose:  ←←←←←← (lane change left)
    Expert B chose:  ........→→→ (slow down then proceed)

    Model average:   ←←.. (goes slightly left, not enough to avoid obstacle)
                          = COLLISION!
```

### The Solution: Generate Many + Score

Instead of predicting ONE trajectory, generate MANY and pick the best:

```
Step 1: Generate K=64 diverse candidate trajectories
Step 2: Score each one for safety, comfort, progress, rules
Step 3: Pick the highest-scoring trajectory
Step 4: Execute it
```

---

## The Generate-Score-Select Pipeline

```
┌──────────────────────────────────────────────────────────────────┐
│                                                                  │
│   [Planner / Generator]                                          │
│   (e.g., GenAD diffusion,      ┌──────────────┐                │
│    VAD K-queries,               │  Candidate 1 │──┐             │
│    sampling-based)              │  Candidate 2 │──┤             │
│                                 │  Candidate 3 │──┤  [SCORER]   │
│                                 │  ...         │──┤──────────>  │
│                                 │  Candidate K │──┘     │       │
│                                 └──────────────┘        │       │
│                                                         v       │
│                                                    Best Path    │
│                                                    (highest     │
│                                                     score)      │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## Classical vs Learned Approaches

| Aspect | Classical (Rule-Based) | Learned (Neural Network) |
|:---|:---|:---|
| **How it works** | Explicit formulas with tuned weights | Neural network trained on data |
| **Transparency** | Fully interpretable (can explain why) | Black box (harder to explain) |
| **Adaptability** | Fixed rules, manual tuning | Learns from data, adapts |
| **Edge cases** | Must explicitly code every case | Can generalize to unseen cases |
| **Development time** | Fast to prototype | Requires training infrastructure |
| **Safety guarantee** | Can provide hard constraints | Probabilistic only |
| **Example** | `score = -5*collision - 1.5*jerk + 2*progress` | `score = MLP(trajectory, scene)` |

### When to Use Which?

- **Classical:** Safety-critical checks, certification requirements, explainability needed
- **Learned:** Complex scenes, capturing human preferences, nuanced situations
- **Both (recommended):** Learned scorer for selection + classical safety checker as hard veto

---

## Module Structure

```
planner_scorer/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── docs/
│   └── technical_overview.md    # Deep technical documentation
│
├── classical/                   # Rule-based scoring
│   ├── cost_function.py         # Weighted multi-criteria scorer
│   │                            # (collision, TTC, comfort, progress, lane keeping)
│   └── safety_checker.py        # Hard safety constraints
│                                # (TTC, RSS, collision, kinematic feasibility)
│
└── learned/                     # Neural network scoring
    ├── mlp_scorer.py            # Simple MLP-based scorer
    ├── transformer_scorer.py    # Attention-based scorer (cross-attention)
    ├── train.py                 # Training pipeline with multiple losses
    └── config.py                # Hyperparameter configurations
```

### File Descriptions

| File | What It Does | Key Feature |
|:---|:---|:---|
| `cost_function.py` | Scores trajectories using weighted sum of costs | 8 cost terms, fully configurable weights |
| `safety_checker.py` | Binary pass/fail safety checks | TTC, RSS, polygon collision, kinematic limits |
| `mlp_scorer.py` | Simple neural scorer (MLP) | Fast, lightweight, good baseline |
| `transformer_scorer.py` | Advanced scorer with attention | Cross-attention between trajectory and scene |
| `train.py` | Training pipeline | BCE + ranking + contrastive losses |
| `config.py` | All hyperparameters | Dataclass-based, easy to modify |

---

## Key Concepts

### Safety Metrics

| Concept | What It Is | Why It Matters |
|:---|:---|:---|
| **TTC** (Time-to-Collision) | How many seconds until you hit something | < 1.5s = emergency braking needed |
| **RSS** (Responsibility-Sensitive Safety) | Intel/Mobileye safety framework | Guarantees ego is never "at fault" |
| **Collision Check** | Do ego and obstacle polygons overlap? | Binary: safe or not safe |
| **Kinematic Feasibility** | Can the car physically follow this path? | Checks max steering, acceleration limits |
| **Drivable Area** | Is the trajectory on the road? | Prevents driving on sidewalks |

### Comfort Metrics

| Concept | What It Is | Threshold |
|:---|:---|:---|
| **Acceleration** | Forward/backward force | < 3 m/s^2 comfortable |
| **Jerk** | Rate of acceleration change | < 2.5 m/s^3 comfortable |
| **Lateral Acceleration** | Side force in turns | < 2 m/s^2 comfortable |
| **Curvature** | Sharpness of turns | Depends on speed |

### Training Losses for Learned Scorers

| Loss | What It Does | Formula Intuition |
|:---|:---|:---|
| **BCE** (Binary Cross-Entropy) | "Is this trajectory good or bad?" | Classification: expert=1, non-expert=0 |
| **Ranking Loss** | "Is trajectory A better than B?" | Pairwise comparison |
| **Contrastive (InfoNCE)** | "Expert trajectory should score highest among all candidates" | Pull expert up, push others down |
| **Combined** | All three together | Complementary supervision signals |

---

## How Scoring Integrates with Planners

### With GenAD (Diffusion-based planner)

```
GenAD generates 64 diverse trajectories via diffusion
    -> Each trajectory scored by the learned scorer
    -> Additional safety check by classical checker
    -> Highest-scoring trajectory that passes safety = execute
```

### With VAD (K ego queries)

```
VAD generates K=6 candidate trajectories from ego queries
    -> Built-in scoring head selects the best
    -> Can add external scorer for additional safety
```

### With Sampling-based Planners

```
Sample 256 trajectories from a motion model
    -> Classical scorer ranks them by cost
    -> Safety checker vetoes dangerous ones
    -> Top trajectory after filtering = execute
```

### Recommended Architecture (Industry Practice)

```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  [Generator] -> [Learned Scorer] -> [Safety Checker]    │
│                  (selects best)      (hard veto)        │
│                                                         │
│  If safety checker rejects ALL candidates:              │
│     -> Emergency brake (fail-safe)                      │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## Getting Started

### Prerequisites

```bash
pip install torch numpy scipy shapely tqdm pyyaml
```

### Running Demos

```bash
# Classical scoring (no ML, pure math)
python planner_scorer/classical/cost_function.py --demo
# Shows: 4 trajectories scored against an obstacle scenario
# Output: Ranked trajectories with sub-cost breakdown

# Safety checking (requires shapely for polygon math)
python planner_scorer/classical/safety_checker.py
# Shows: TTC, collision, RSS, and kinematic checks

# MLP scorer (simple neural network)
python planner_scorer/learned/mlp_scorer.py
# Shows: Score multiple trajectories with a learned model

# Transformer scorer (attention-based, most advanced)
python planner_scorer/learned/transformer_scorer.py
# Shows: Cross-attention scoring with scene context
```

### Training a Scorer

```bash
cd planner_scorer/learned

# Train MLP scorer with combined loss
python train.py --model mlp --loss combined --epochs 50

# Train transformer scorer with contrastive loss
python train.py --model transformer --loss contrastive --epochs 50

# Available options:
#   --model: mlp, transformer
#   --loss: bce, ranking, contrastive, combined
#   --epochs: number of training epochs
#   --lr: learning rate (default 1e-4)
```

### Understanding the Output

When you run `cost_function.py --demo`:

```
Trajectory 1: Lane Change Left
  Total Score: -2.847          <- Higher (less negative) = better
  Sub-costs:
    collision:     0.0000      <- No collision (good!)
    ttc:           0.1234      <- Safe TTC
    comfort_accel: 0.0000      <- Smooth acceleration
    comfort_jerk:  0.2345      <- Low jerk
    progress:      0.1000      <- Good forward progress
    lane_keeping:  0.8000      <- Deviates from center (changing lanes)
```

---

## Safety Considerations

### For Real Deployment

1. **Never rely solely on a learned scorer** -- always have a classical safety checker as backup
2. **Fail-safe behavior** -- if ALL trajectories are rejected, execute emergency stop
3. **Temporal consistency** -- don't switch between wildly different trajectories frame-to-frame
4. **Latency budget** -- scoring must complete within the planning cycle (typically 50-100ms)
5. **Out-of-distribution detection** -- flag when the scene is unlike anything seen in training

### Safety Stack (Recommended)

```
Level 1: Learned Scorer         (picks best trajectory)
Level 2: Classical Safety Check  (vetoes dangerous ones)
Level 3: Kinematic Feasibility   (ensures physically possible)
Level 4: Emergency Stop          (if everything fails)
```

---

## Further Reading

- `docs/technical_overview.md` -- Deep dive into scoring theory, loss functions, and math
- `../one_step_e2e/GenAD/` -- Diffusion-based planner that naturally pairs with scoring
- `../two_step_e2e/VAD/` -- Model with built-in K-trajectory scoring head

### Key References

- [RSS Paper](https://arxiv.org/abs/1708.06374) -- Responsibility-Sensitive Safety (Intel/Mobileye)
- [nuPlan](https://www.nuscenes.org/nuplan) -- Planning-focused dataset with scoring metrics
- [CoverNet](https://arxiv.org/abs/1911.10298) -- Fixed trajectory set + classification approach
- [DIPP](https://arxiv.org/abs/2207.09434) -- Differentiable Integrated Prediction and Planning
