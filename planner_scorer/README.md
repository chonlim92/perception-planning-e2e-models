# Planner Scorer: Trajectory Scoring and Selection for Autonomous Driving

> A beginner-friendly guide to trajectory scoring — the critical module that evaluates, ranks, and selects the best driving trajectory from multiple candidates. This module bridges the gap between trajectory generation and safe execution.

---

## Table of Contents

- [What is Trajectory Scoring?](#what-is-trajectory-scoring)
- [The Multi-Modal Planning Problem](#the-multi-modal-planning-problem)
- [The Generate-Score-Select Pipeline](#the-generate-score-select-pipeline)
- [Classical vs Learned Approaches](#classical-vs-learned-approaches)
- [Files in This Module](#files-in-this-module)
- [Key Concepts](#key-concepts)
- [How Scoring Integrates with Planners](#how-scoring-integrates-with-planners)
- [Getting Started / Running Demos](#getting-started--running-demos)
- [Safety Considerations in Real Deployment](#safety-considerations-in-real-deployment)
- [References](#references)

---

## What is Trajectory Scoring?

A **trajectory scorer** is a module that takes a candidate driving trajectory (a sequence of future positions the ego vehicle could follow) and assigns it a **score** indicating how good that trajectory is. "Good" means safe, comfortable, efficient, and rule-compliant.

```
What the scorer does:

  Input:  A candidate trajectory = [(x1,y1,t1), (x2,y2,t2), ..., (xT,yT,tT)]
          + Scene context (where are other cars? what does the map look like?)

  Output: A scalar score (higher = better trajectory)

  Example:
    Trajectory A (smooth lane change):    Score = 0.92  <-- Best!
    Trajectory B (aggressive swerve):     Score = 0.35
    Trajectory C (stop in the road):      Score = 0.58
    Trajectory D (rear-end collision):    Score = 0.01  <-- Rejected!
```

### Why Do We Need Scoring?

Modern E2E planners (especially generative models like GenAD) do not output a single trajectory. They output **many candidate trajectories** — typically 32 to 128. Someone needs to decide which one to actually execute. That "someone" is the scorer.

Without a scorer, a generative planner is like a brainstorming session with no decision-maker. Lots of ideas, but no way to pick the best one.

---

## The Multi-Modal Planning Problem

### Why Driving Has Multiple Correct Answers

Driving is inherently **multi-modal** — at any given moment, there are multiple valid actions a driver could take. This is fundamentally different from, say, image classification where there is ONE correct label.

### Example: The Stopped Car Scenario

```
  Scenario: You are driving on a two-lane road. A car is stopped ahead.
  ========================================================================

  Your car (ego):            Stopped car:
       ___                      ___
      | E |---->               | S |
      |___|                    |___|
  ----lane 1--------------------------------------------
  ----lane 2--------------------------------------------


  Valid options (ALL are correct!):

  Option A: Change to lane 2 early, pass on the left
            Trajectory: [(0,0), (5,-1), (10,-3.5), (15,-3.5), (20,-3.5)]
            Quality: Good if lane 2 is clear

  Option B: Slow down, wait for lane 2 to clear, then change lanes
            Trajectory: [(0,0), (3,0), (5,0), (8,-2), (12,-3.5)]
            Quality: Good if lane 2 has traffic

  Option C: Slow down and stop behind the car
            Trajectory: [(0,0), (5,0), (8,0), (9,0), (9,0)]
            Quality: Always safe, but inefficient

  Option D: Accelerate and cut sharply left
            Trajectory: [(0,0), (8,-1), (12,-3.5), (15,-3.5), (20,-3.5)]
            Quality: Unsafe! High lateral acceleration, dangerous.
```

### Why This Breaks Simple Models

```
  A model trained with simple L2 loss will AVERAGE all expert behaviors:

    Expert dataset contains:
      50% of experts went LEFT:  trajectory = [(-1,1), (-2,2), (-3,3)]
      50% of experts went RIGHT: trajectory = [(1,1), (2,2), (3,3)]

    A model trained with L2 loss will AVERAGE these:
      predicted = [(0,1), (0,2), (0,3)]  <-- goes STRAIGHT into the obstacle!

  This is called the "mode averaging" problem.
```

### The Solution: Generate Multiple + Score

```
  Traditional planning (regression):
    Predict ONE trajectory: t* = argmin_t L(t, t_expert)
    Problem: Averages over multiple modes --> invalid trajectory

  Multi-modal planning (generate + score):
    Step 1: Generate a SET of trajectories: T = {t1, t2, ..., tK}
    Step 2: Score each one: s(ti) = Scorer(ti, context) for all i
    Step 3: Select the best: t* = argmax_i s(ti)
    Result: Always selects a valid trajectory (no averaging!)
```

---

## The Generate-Score-Select Pipeline

This is the core workflow that the planner scorer participates in:

```
  THE GENERATE-SCORE-SELECT PIPELINE
  ====================================================================

  STEP 1: GENERATE                    STEP 2: SCORE                 STEP 3: SELECT
  (produce K candidates)              (evaluate each one)           (pick the best)

  +-----------------------+     +---------------------------+     +----------------+
  |                       |     |                           |     |                |
  | Generative Planner    |     |    Scorer Module          |     |  argmax(scores)|
  | (e.g., GenAD diffusion|     |                           |     |                |
  |  or sampling-based)   |     |  For each trajectory:     |     |  Returns the   |
  |                       |     |                           |     |  single best   |
  | Outputs:              |     |  +-------------------+    |     |  trajectory    |
  |  t1: go left early    | --> |  | Safety:    0.95   |    | --> |  for execution |
  |  t2: go left late     |     |  | Comfort:   0.80   |    |     |                |
  |  t3: slow + follow    |     |  | Progress:  0.60   |    |     +-------+--------+
  |  t4: stop completely  |     |  | Rules:     1.00   |    |             |
  |  ...                  |     |  +-------------------+    |             v
  |  t64: aggressive cut  |     |  Combined: 0.87          |     Execute t2
  |                       |     |                           |     (best score)
  +-----------------------+     |  Repeat for all 64...     |
                                |                           |
                                +---------------------------+

  Full pipeline in one line:
  Cameras --> [E2E Model] --> 64 trajectories --> [Scorer] --> best trajectory --> [Controller]
```

### Where Does the Scorer Fit in the Overall System?

```
  COMPLETE AUTONOMOUS DRIVING SYSTEM
  ====================================================================

  Sensors          Perception           Planning              Control
  --------         ----------           --------              -------
  Cameras    -->   BEV Features   -->   Trajectory      -->   PID
  LiDAR      -->   Object Queries -->   Generation      -->   Controller
  GPS/IMU    -->   Map Vectors    -->   (K candidates)  -->   Steer/Gas/Brake
                                             |
                                             v
                                    +------------------+
                                    | PLANNER SCORER   |  <-- THIS MODULE
                                    |                  |
                                    | Evaluates all K  |
                                    | candidates and   |
                                    | selects the best |
                                    +--------+---------+
                                             |
                                             v
                                      Best trajectory
                                             |
                                             v
                                    +------------------+
                                    | Safety Checker   |  <-- Also in this module
                                    | (hard veto if    |
                                    |  unsafe)         |
                                    +--------+---------+
                                             |
                                             v
                                       EXECUTE (or emergency stop)
```

---

## Classical vs Learned Approaches

This module implements both approaches to scoring. Each has strengths:

### Classical (Rule-Based) Scoring

Classical scorers use **explicit mathematical formulas** that a human engineer designs. Each formula captures one aspect of driving quality (safety, comfort, etc.), and the final score is a weighted sum.

```python
# Classical scoring formula (simplified from cost_function.py):
def score_trajectory(trajectory, scene):
    safety_score    = compute_collision_risk(trajectory, scene.obstacles)
    comfort_score   = compute_jerk_and_accel(trajectory)
    progress_score  = compute_route_progress(trajectory, scene.route)
    rules_score     = compute_rule_compliance(trajectory, scene.traffic_rules)

    # Weighted combination (weights tuned by engineers)
    total_score = (
        -5.0 * safety_score       # heavily penalize collision risk
        -1.5 * comfort_score      # penalize jerk and harsh acceleration
        +2.0 * progress_score     # reward making progress on route
        +1.0 * rules_score        # reward following traffic rules
    )
    return total_score
```

### Learned (Neural Network) Scoring

Learned scorers use a **neural network** trained on expert driving data. The network learns from examples what "good" and "bad" trajectories look like in different contexts.

```python
# Learned scoring (simplified from mlp_scorer.py):
class LearnedScorer(nn.Module):
    def forward(self, trajectory, scene_context):
        traj_features = self.traj_encoder(trajectory)        # encode trajectory
        scene_features = self.scene_encoder(scene_context)   # encode scene
        combined = torch.cat([traj_features, scene_features])
        score = self.score_head(combined)                    # predict quality
        return score  # trained to output high for expert trajectories
```

### Comparison Table

| Aspect | Classical (Rule-Based) | Learned (Neural Network) |
|--------|:---:|:---:|
| **How it works** | Hand-crafted formulas | Trained on expert data |
| **Transparency** | Fully interpretable (you know exactly why a score is high/low) | Black box (hard to explain why) |
| **Tuning** | Engineer tunes weights manually | Training algorithm tunes automatically |
| **Adaptability** | Rigid (must re-engineer for new scenarios) | Flexible (learns patterns from data) |
| **Safety guarantee** | Can provide mathematical guarantees | Statistical only (no formal guarantee) |
| **Performance ceiling** | Limited by human engineering ability | Can discover non-obvious patterns |
| **Edge cases** | Must explicitly handle each case | Learns from data (if examples exist) |
| **Failure mode** | Predictable (formula gives wrong answer) | Unpredictable (neural net gives wrong answer) |
| **Compute cost** | Very fast (simple math) | Slower (neural network forward pass) |
| **Deployment confidence** | High (well-understood behavior) | Lower (black box) |
| **Used by** | Most production vehicles today | Research, emerging in production |
| **Implementation** | `classical/cost_function.py`, `classical/safety_checker.py` | `learned/mlp_scorer.py`, `learned/transformer_scorer.py` |

### When to Use Which?

- **Classical only:** Safety-critical checks, certification requirements, explainability needed
- **Learned only:** Complex scenes, capturing nuanced human preferences
- **Both (recommended):** Learned scorer for soft ranking + classical safety checker as hard veto

---

## Files in This Module

### `classical/cost_function.py` -- Weighted Multi-Criteria Scorer

The main classical scoring implementation. Evaluates trajectories on four axes:

| Criterion | What It Measures | Example Formula |
|-----------|------------------|-----------------|
| Safety | How close to collision | TTC (time to collision), distance to obstacles |
| Comfort | How smooth the ride | Lateral/longitudinal jerk, curvature rate |
| Progress | How efficiently reaching goal | Distance along planned route per second |
| Compliance | Traffic rule adherence | Speed limit delta, lane boundary distance |

Each criterion produces a sub-score, and the final score is a weighted sum. The weights are configurable (e.g., safety weight >> comfort weight).

**Key classes:**
- `TrajectoryPoint`: A single waypoint (x, y, heading, velocity, acceleration, curvature)
- `Trajectory`: A sequence of waypoints with metadata
- `CostWeights`: Configurable weights for each cost term
- `CostFunction`: The main scorer that combines all sub-costs

```bash
python classical/cost_function.py --demo
```

---

### `classical/safety_checker.py` -- Hard Safety Constraints

A binary pass/fail checker that enforces **non-negotiable safety rules**. Unlike the cost function (which assigns soft scores), the safety checker issues hard vetoes — a trajectory that fails ANY safety check is immediately rejected regardless of its score.

**Implements:**
- **TTC (Time-to-Collision)**: Rejects trajectories where collision is imminent (< 1.5 seconds)
- **RSS (Responsibility-Sensitive Safety)**: Checks Mobileye's formal safety model — ensures safe following distances and response margins
- **Drivable area compliance**: Rejects trajectories that leave the road or enter wrong-way lanes
- **Kinematic feasibility**: Rejects physically impossible trajectories (acceleration beyond vehicle capability)

**Key classes:**
- `SafetyConfig`: Threshold configuration (min TTC, max acceleration, RSS parameters)
- `SafetyChecker`: Main checker with methods for each type of safety constraint

```bash
python classical/safety_checker.py
```

---

### `learned/mlp_scorer.py` -- MLP-Based Scorer

A simple feedforward neural network scorer. Good baseline and fast to run.

**Architecture:**
```
Trajectory (16 waypoints x 4 features) --> Flatten --> MLP --> 256-dim features
Scene context (agents + map)            --> Pool --> MLP --> 256-dim features
                                                                  |
Concatenate [traj_features, scene_features] --> MLP --> Scalar score
```

**Key parameters:**
- `traj_points=16`: Number of future waypoints (at 0.5s intervals = 8 second horizon)
- `traj_dim=4`: Features per waypoint (x, y, heading, velocity)
- `hidden_dim=256`: Internal feature dimension
- `max_agents=32`: Maximum number of other agents encoded

**When to use:** Fast inference, simple scenes, good starting point for learning.

```bash
python learned/mlp_scorer.py
```

---

### `learned/transformer_scorer.py` -- Transformer-Based Scorer

A more powerful scorer that uses **cross-attention** between trajectory waypoints and scene elements. Each waypoint "attends to" nearby agents and map features to understand interactions.

**Architecture:**
```
Trajectory waypoints -----> Positional encoding --> Query tokens
                                                         |
Scene elements (agents + map) --> Positional encoding --> Key/Value tokens
                                                         |
                            Cross-Attention (3 layers)    |
                            (trajectory queries attend    |
                             to scene keys/values)        |
                                                         |
                                    Pool --> MLP --> Scalar score
```

**Key parameters:**
- `d_model=256`: Transformer dimension
- `n_heads=8`: Number of attention heads
- `num_cross_layers=3`: Cross-attention layers (traj attends to scene)
- `num_scene_layers=2`: Self-attention layers for scene encoding

**When to use:** Complex scenes with many interacting agents. Worth the extra compute when accuracy matters more than speed.

```bash
python learned/transformer_scorer.py
```

---

### `learned/train.py` -- Training Pipeline

Trains either the MLP or Transformer scorer using one or more loss functions:

**Supported loss functions:**
- `bce` -- Binary Cross-Entropy (classify trajectories as good/bad)
- `ranking` -- Margin Ranking Loss (expert trajectory should score higher than non-expert)
- `contrastive` -- InfoNCE Contrastive Loss (1 positive + N negatives)
- `combined` -- Weighted sum of all three losses

**Key features:**
- Synthetic data generation for training (simulates scenarios with good/bad trajectories)
- Validation with accuracy metrics
- Checkpoint saving for best model

```bash
python learned/train.py --model mlp --loss combined --epochs 50
python learned/train.py --model transformer --loss contrastive --epochs 50
```

---

### `learned/config.py` -- Hyperparameters

Configuration dataclasses for model architecture, training, and data generation. Uses Python dataclasses for type-safe, documented configuration.

**Key configurations:**
- `TrainingConfig`: Learning rate, batch size, epochs, loss weights
- `ModelConfig`: Architecture parameters (dimensions, layers, heads)
- `FullConfig`: Combines all configs into one object

---

## Key Concepts

### TTC (Time-to-Collision)

TTC is the time until the ego vehicle would collide with an obstacle if both maintain their current velocities. It is the most fundamental safety metric in autonomous driving.

```
TTC Calculation (simplified):

  Ego vehicle:    position = (0, 0),  velocity = 15 m/s (forward)
  Obstacle ahead: position = (30, 0), velocity = 5 m/s (forward)

  Relative velocity = 15 - 5 = 10 m/s (ego approaching obstacle)
  Distance = 30 m
  TTC = Distance / Relative velocity = 30 / 10 = 3.0 seconds

  Rule: If TTC < 1.5 seconds --> REJECT the trajectory (too dangerous)
        If TTC < 3.0 seconds --> Penalize (getting close to danger)
        If TTC > 5.0 seconds --> No penalty (plenty of time)
```

### RSS (Responsibility-Sensitive Safety)

RSS is a mathematical model developed by Mobileye (Intel) that defines what "safe driving" means in a formal, provable way. If you follow RSS rules, you are guaranteed to NEVER be at fault in an accident.

```
RSS Safe Following Distance:

  d_safe = v_ego * t_response + (v_ego^2) / (2 * a_max_brake) - (v_front^2) / (2 * a_min_brake)

  Where:
    v_ego        = ego vehicle speed (e.g., 20 m/s)
    v_front      = front vehicle speed (e.g., 15 m/s)
    t_response   = reaction time (0.5s for automated systems)
    a_max_brake  = maximum braking deceleration of ego (6 m/s^2)
    a_min_brake  = assumed minimum braking of front car (3.5 m/s^2)

  Example:
    d_safe = 20*0.5 + (20^2)/(2*6) - (15^2)/(2*3.5)
           = 10 + 33.3 - 32.1
           = 11.2 meters

  If actual distance < 11.2m --> RSS-unsafe --> REJECT trajectory
```

### Contrastive Loss (InfoNCE)

Used to train learned scorers. The idea: given one "expert" trajectory (positive) and many "non-expert" trajectories (negatives), the scorer should assign the highest score to the expert.

```
Contrastive Loss:

  Positive: t+ (the trajectory the expert actually drove)
  Negatives: t1-, t2-, ..., tN- (randomly generated or perturbed trajectories)

  Scores:
    s+ = Scorer(t+, scene)     -- should be HIGH
    si- = Scorer(ti-, scene)   -- should be LOW

  InfoNCE Loss:
    L = -log( exp(s+ / tau) / (exp(s+ / tau) + sum_i exp(si- / tau)) )

  Where tau (temperature) controls how "peaky" the probability distribution is:
    - Low tau (0.05): Very sharp -- demands large gaps between scores
    - High tau (1.0): Softer -- tolerates smaller gaps

  This pushes the scorer to give the expert trajectory a much higher score
  than all alternatives.
```

### Ranking Loss (Margin Loss)

A simpler alternative to contrastive loss. Given pairs of trajectories where one is better:

```
Ranking Loss:

  Given: trajectory A is better than trajectory B
         (e.g., A is the expert trajectory, B is a random perturbation)

  Scores: sA = Scorer(A), sB = Scorer(B)

  Loss = max(0, margin - (sA - sB))

  If sA > sB + margin: Loss = 0 (correct ranking with sufficient gap)
  If sA < sB + margin: Loss > 0 (wrong ranking, push sA higher or sB lower)

  Typical margin = 0.3 (want at least 0.3 score gap between good and bad)
```

### BCE (Binary Cross-Entropy)

The simplest training approach. Classify each trajectory as "good" (label=1) or "bad" (label=0):

```
BCE Loss:

  Good trajectory (expert):   label = 1, Scorer should output high score (~1.0)
  Bad trajectory (random):    label = 0, Scorer should output low score (~0.0)

  L = -(label * log(score) + (1-label) * log(1-score))

  Pros: Simple, stable training, works well as a starting point
  Cons: Does not capture relative ranking (a "slightly bad" trajectory and a
        "catastrophically bad" trajectory both get label=0, treated the same)
```

---

## How Scoring Integrates with Planners

### Integration with GenAD (Diffusion-Based Planner)

GenAD is the most natural partner for the planner scorer. GenAD generates diverse trajectories via diffusion, and the scorer selects the best one:

```
  GenAD + Planner Scorer Integration:
  ====================================================================

  Scene Features (from perception)
         |
         v
  +------------------+
  | GenAD Diffusion  |     Noise_1 --> Denoise --> Trajectory_1
  | Model            |     Noise_2 --> Denoise --> Trajectory_2
  |                  |     Noise_3 --> Denoise --> Trajectory_3
  | (conditioned on  |     ...
  |  scene features) |     Noise_64 --> Denoise --> Trajectory_64
  +------------------+
         |
         | 64 diverse trajectory candidates
         v
  +------------------+
  | Planner Scorer   |     Score(Traj_1, scene) = 0.87
  | (this module!)   |     Score(Traj_2, scene) = 0.42
  |                  |     Score(Traj_3, scene) = 0.91  <-- BEST
  | MLP or           |     ...
  | Transformer      |     Score(Traj_64, scene) = 0.65
  +------------------+
         |
         | Select argmax
         v
  +------------------+
  | Safety Checker   |     TTC check: PASS
  | (hard veto)      |     RSS check: PASS
  |                  |     Drivable area: PASS
  +------------------+
         |
         v
  Execute Trajectory_3 (score=0.91, all safety checks passed)
```

### Integration with VAD (K Ego Queries)

VAD generates a smaller number of trajectory candidates from its ego queries:

```
  VAD + External Scorer:
  ====================================================================

  VAD outputs K=6 trajectories from ego queries
    -> VAD's internal scoring head ranks them (built-in)
    -> External planner scorer provides a SECOND opinion
    -> Combined score = alpha * internal + (1-alpha) * external
    -> Select the trajectory with highest combined score
    -> Safety checker as final veto
```

### Integration with Sampling-Based Planners

Classical planners that sample from a motion model:

```
  Sampling Planner + Scorer:
  ====================================================================

  Motion Model generates candidates:
    - 5 lateral offsets x 4 speeds x 3 time horizons = 60 candidates
    - Each is a kinematically feasible polynomial trajectory

  Classical scorer ranks all 60:
    - Cost = -5*collision - 1.5*jerk + 2*progress + 1*lane_keeping
    - Top-5 candidates passed to safety checker

  Safety checker validates:
    - TTC > 1.5s? RSS safe distance? On drivable area?
    - If all 5 fail --> emergency brake
    - Otherwise execute highest-scored safe trajectory
```

### Integration with Two-Step E2E Models (UniAD)

Two-step models typically output a single trajectory, but can be extended:

```
  UniAD + External Scorer (re-ranking):
  ====================================================================

  UniAD Planning Decoder outputs top-K hypotheses (beam search):
    Hypothesis 1: decoder_score = 0.95
    Hypothesis 2: decoder_score = 0.88
    Hypothesis 3: decoder_score = 0.82

  External scorer provides a complementary evaluation:
    Scorer(Hyp 1) = 0.60  (decoder liked it, but scorer says uncomfortable)
    Scorer(Hyp 2) = 0.90  (decoder ranked 2nd, but scorer says best overall)
    Scorer(Hyp 3) = 0.45  (both agree this is worse)

  Combined: 0.5 * decoder_score + 0.5 * external_score
    Hyp 1: 0.5*0.95 + 0.5*0.60 = 0.775
    Hyp 2: 0.5*0.88 + 0.5*0.90 = 0.890  <-- Selected!
    Hyp 3: 0.5*0.82 + 0.5*0.45 = 0.635
```

---

## Getting Started / Running Demos

### Prerequisites

```bash
pip install -r requirements.txt
# Core dependencies: torch, numpy, scipy, shapely, matplotlib, tqdm, pyyaml
```

### Quick Demo: Classical Scoring

```bash
# Run the cost function on synthetic trajectory candidates
python classical/cost_function.py --demo

# Expected output:
#   Trajectory 1 (smooth lane change):    Score = -2.85
#   Trajectory 2 (aggressive swerve):     Score = -8.91
#   Trajectory 3 (comfortable decel):     Score = -3.42
#   Trajectory 4 (collision trajectory):  Score = -24.10
#   Best trajectory: #1 (highest score = least negative)
```

### Quick Demo: Safety Checker

```bash
# Run the safety checker on sample trajectories
python classical/safety_checker.py

# Expected output:
#   Trajectory 1: TTC=4.2s, RSS=SAFE, Drivable=YES  --> PASS
#   Trajectory 2: TTC=0.8s, RSS=UNSAFE              --> REJECT (TTC < 1.5s)
#   Trajectory 3: TTC=3.1s, RSS=SAFE, Drivable=NO   --> REJECT (off-road)
```

### Quick Demo: Learned Scorers

```bash
# Run MLP scorer demo (creates model, generates synthetic data, scores trajectories)
python learned/mlp_scorer.py

# Run Transformer scorer demo (more complex, shows cross-attention scoring)
python learned/transformer_scorer.py
```

### Training a Learned Scorer

```bash
cd planner_scorer/learned

# Train MLP scorer with combined loss (fast, good for experimentation)
python train.py --model mlp --loss combined --epochs 50

# Train Transformer scorer with contrastive loss (slower, better performance)
python train.py --model transformer --loss contrastive --epochs 50

# All training options:
#   --model:   mlp | transformer
#   --loss:    bce | ranking | contrastive | combined
#   --epochs:  number of training epochs (default: 100)
#   --lr:      learning rate (default: 1e-4)
#   --batch:   batch size (default: 32)
```

### Expected Training Output

```
Epoch  1/50 | Loss: 2.341 | Train Acc: 52.3% | Val Acc: 51.8%
Epoch 10/50 | Loss: 0.834 | Train Acc: 74.6% | Val Acc: 73.1%
Epoch 25/50 | Loss: 0.412 | Train Acc: 88.2% | Val Acc: 85.7%
Epoch 50/50 | Loss: 0.198 | Train Acc: 94.1% | Val Acc: 91.3%

Best model saved to: checkpoints/best_model.pth
```

### Suggested Exploration Order

1. **Start with `cost_function.py --demo`** -- See how classical scoring works, understand the sub-costs
2. **Read `safety_checker.py`** -- Understand hard constraints (TTC, RSS)
3. **Run `mlp_scorer.py`** -- See the simplest learned scorer in action
4. **Run `transformer_scorer.py`** -- See how attention improves scoring
5. **Train with `train.py`** -- Experiment with different losses
6. **Read `docs/technical_overview.md`** -- Deep dive into the math

---

## Safety Considerations in Real Deployment

### The Scorer is Safety-Critical

In a real autonomous vehicle, the planner scorer is on the **critical path** — if it selects a bad trajectory, the car executes that trajectory. This imposes strict requirements:

### 1. Never Trust the Scorer Alone

Always combine the learned scorer with a classical safety checker:

```
Architecture for safe deployment:

  Learned Scorer (soft ranking)
       |
       v
  Top-K candidates (ranked by learned scorer)
       |
       v
  Classical Safety Checker (hard constraints -- TTC, RSS, drivable area)
       |
       v
  If ALL top-K fail safety check --> Emergency fallback (brake to stop)
  If at least one passes --> Execute the highest-scored safe trajectory
```

### 2. Fallback Trajectory Hierarchy

Always maintain at least one "known safe" fallback:

```
Fallback hierarchy (in order of preference):
  1. Best trajectory from scorer (if safe)       --> Execute normally
  2. Straight-line deceleration to stop          --> Execute if no scored traj is safe
  3. Emergency brake (maximum deceleration)      --> Execute if imminent collision
  4. System alert + minimal risk condition (MRC) --> Pull over if system failure
```

### 3. Scorer Failure Modes and Mitigations

| Failure Mode | Risk | Mitigation |
|--------------|------|------------|
| Scorer assigns high score to unsafe trajectory | Collision | Hard safety checker as veto layer |
| Scorer assigns low score to ALL trajectories | Unnecessary stop | Fallback to safe deceleration (not a crash) |
| Scorer confidence is low (ambiguous scores) | Suboptimal choice | Increase conservatism (reduce speed) |
| Scorer encounters unseen scenario (OOD) | Unpredictable | OOD detection + conservative fallback |
| Scorer latency exceeds time budget | Stale decision | Timeout + reuse previous trajectory |
| Scorer crashes (software error) | No scoring | Watchdog timer + emergency stop |

### 4. Temporal Consistency

The scorer should not cause the vehicle to "oscillate" between different trajectory modes:

```
BAD (oscillating):
  Frame 1: Select "go left"
  Frame 2: Select "go right"
  Frame 3: Select "go left"
  --> Car swerves dangerously!

GOOD (consistent):
  Frame 1: Select "go left"
  Frame 2: Select "go left" (commit to decision)
  Frame 3: Select "go left" (continue executing)

Mitigation: Add a "consistency bonus" for trajectories similar to the previously
executed trajectory. Penalize large deviations between consecutive frames.
```

### 5. Validation Requirements for Production

Before deploying a learned scorer in a real vehicle, you must validate:

- **Closed-loop simulation**: Run for millions of miles in simulation. Measure collision rate, discomfort events, and route completion rate.
- **Adversarial testing**: Test edge cases -- construction zones, emergency vehicles, sensor noise, extreme weather.
- **OOD detection**: Monitor if real-world inputs are different from training data. Alert when out-of-distribution.
- **Latency guarantee**: The scorer must complete within its time budget (typically < 10ms) in ALL cases, including worst-case.
- **Shadow mode**: Run the scorer in parallel with a human driver. Compare its selections to human decisions. Fix disagreements.
- **Graceful degradation**: When the scorer fails, the system must degrade safely (slow down, not crash).

### 6. The Safety Stack (Recommended Architecture)

```
Layer 1: Learned Scorer        --> Selects best trajectory (soft, data-driven)
Layer 2: Classical Checker     --> Vetoes unsafe trajectories (hard, rule-based)
Layer 3: Kinematic Feasibility --> Ensures physically possible (physics-based)
Layer 4: Emergency Fallback    --> Brake if all else fails (guaranteed safe)
```

Each layer is independent and any single layer can save the system even if others fail.

---

## References

### Papers

- **RSS**: Shalev-Shwartz et al., "On a Formal Model of Safe and Scalable Self-driving Cars," 2017. [arXiv:1708.06374](https://arxiv.org/abs/1708.06374)
- **nuPlan Challenge**: motional.com/nuplan -- The primary benchmark for planning and scoring
- **Werling et al.**: "Optimal Trajectory Generation for Dynamic Street Scenarios in a Frenet Frame," ICRA 2010 -- Foundational work on trajectory generation
- **GameFormer**: Huang et al., "GameFormer: Game-theoretic Modeling and Learning of Transformer-based Interactive Prediction and Planning," ICCV 2023. [arXiv:2303.05760](https://arxiv.org/abs/2303.05760)
- **DIPP**: Huang et al., "Differentiable Integrated Prediction and Planning," 2023. [arXiv:2305.12071](https://arxiv.org/abs/2305.12071)
- **CTG**: Zhong et al., "Guided Conditional Diffusion for Controllable Traffic Generation," NeurIPS 2023. [arXiv:2304.01223](https://arxiv.org/abs/2304.01223)
- **GenAD**: Zheng et al., "GenAD: Generative End-to-End Autonomous Driving," 2024. [arXiv:2402.11502](https://arxiv.org/abs/2402.11502) -- Primary user of this scorer module
- **CoverNet**: Phan-Minh et al., "CoverNet: Multimodal Behavior Prediction using Trajectory Sets," 2020. [arXiv:1911.10298](https://arxiv.org/abs/1911.10298)

### Background Concepts

- **Contrastive Learning**: Chen et al., "A Simple Framework for Contrastive Learning" (SimCLR), ICML 2020
- **InfoNCE**: Oord et al., "Representation Learning with Contrastive Predictive Coding," 2018
- **TTC**: NHTSA standard automotive safety metric

### Related Modules in This Repository

- `../one_step_e2e/GenAD/` -- Diffusion-based planner that generates candidates for this scorer
- `../two_step_e2e/VAD/` -- Model with built-in K-trajectory scoring head
- `docs/technical_overview.md` -- Deep mathematical treatment of scoring theory

---

## Quality Fixes (Expert Review 2026-06-27)

| Component | Issue | Severity | Fix Applied |
|-----------|-------|----------|-------------|
| `learned/train.py` | Ranking loss loop breaks after 1 iteration | High | Vectorized pairwise margin computation |
| `learned/mlp_scorer.py` | Max-pool masking incorrect for negative features | Medium | `masked_fill(~mask, -inf)` before max |
| `classical/safety_checker.py` | RSS safe distance can be negative | High | Clamped to `max(0, d_safe)` |
| `classical/safety_checker.py` | No lateral RSS check | Low | Documented (longitudinal-only is standard simplification) |
