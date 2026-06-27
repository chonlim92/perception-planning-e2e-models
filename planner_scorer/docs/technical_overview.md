# Planner Scorer: Technical Overview

## 1. Problem Statement

In autonomous driving, planning produces MULTIPLE candidate trajectories because driving is inherently multi-modal. A **planner scorer** evaluates and ranks these candidates to select the best one.

### Why Multiple Valid Trajectories Exist

```
Scenario: Obstacle ahead in lane

Valid behaviors:
├── Change lane left    (safe if left lane is clear)
├── Change lane right   (safe if right lane is clear)  
├── Slow down          (safe, but less efficient)
└── Stop completely     (always safe, but blocks traffic)
```

A single regression model (predicting one trajectory) suffers from **mode averaging** — it may output the mean of all valid behaviors, which could be an INVALID behavior (e.g., driving into the obstacle).

## 2. Mathematical Formulation

### Scoring Function

Given K candidate trajectories {τ₁, τ₂, ..., τ_K} and scene context C:

```
Score(τᵢ | C) → ℝ
τ* = argmax_i Score(τᵢ | C)
```

### Trajectory Representation

A trajectory τ is a sequence of future ego states:
```
τ = {(x_t, y_t, θ_t, v_t, a_t, κ_t)}_{t=1}^{T}
```
- (x, y): position in ego-centric frame
- θ: heading angle
- v: velocity
- a: acceleration  
- κ: curvature

Typical: T=16 waypoints at 2Hz = 8 seconds planning horizon.

### Scene Context

```
C = {
    agents: [(x, y, θ, vx, vy, L, W, type)]  × N_agents,
    map: [polyline_1, ..., polyline_M],
    ego_state: (v, a, yaw_rate),
    route: [route_points],
    traffic_lights: [(state, position)],
}
```

## 3. Classical Scoring Methods

### 3.1 Weighted Multi-Criteria

```
Score(τ) = -Σᵢ wᵢ · cᵢ(τ, C)

Costs:
  c_collision(τ) = max overlap with predicted agent occupancies
  c_comfort(τ) = ∫(a² + j² + κ²) dt
  c_progress(τ) = 1 - (route_distance / expected_distance)
  c_lane(τ) = mean(lateral_distance_to_lane_center²)
  c_speed(τ) = mean(max(0, v - v_limit)²)
```

### 3.2 Safety Constraints (Hard Gates)

Before scoring, trajectories must pass safety checks:
- TTC > 1.5 seconds (time to collision)
- No overlap with drivable area boundary
- Kinematically feasible (a < 3.5 m/s², lat_a < 4 m/s²)
- RSS-compliant longitudinal distance

Failed trajectories are REJECTED regardless of score.

## 4. Learned Scoring Methods

### 4.1 MLP Scorer

```
trajectory_features = MLP(flatten(waypoints))
scene_features = Pool(AgentEncoder(agents), MapEncoder(map))
score = ScorerHead(concat(traj_features, scene_features))
```

**Pros:** Fast, simple, easy to train  
**Cons:** Limited spatial reasoning, no interaction modeling

### 4.2 Transformer Scorer (Cross-Attention)

```
traj_tokens = PositionalEncode(Linear(waypoints))
scene_tokens = SelfAttention(concat(agent_tokens, map_tokens))
attended = CrossAttention(query=traj_tokens, kv=scene_tokens)
score = MLP(Pool(attended))
```

**Pros:** Captures spatial relationships, interaction-aware  
**Cons:** Slower, needs more data

### 4.3 Contrastive Learning

Expert trajectory = positive sample, perturbations = negatives.

```
Loss = -log[exp(S(τ⁺)/τ) / Σ_k exp(S(τ_k)/τ)]
```

**Pros:** No absolute labels needed, good ranking  
**Cons:** Sensitive to negative mining strategy

## 5. Training Strategies

### Data Sources

| Source | Labels | Quality |
|--------|--------|---------|
| Expert replay (nuScenes) | Expert traj = 1.0 | High quality, limited diversity |
| Perturbation | Add noise → label by deviation | Infinite data, may miss failure modes |
| Simulation (CARLA/nuPlan) | Outcome-based (crash=0, success=1) | Diverse, but sim-to-real gap |
| Human preference | Pairwise comparisons | Best signal, expensive |

### Negative Mining

1. **Gaussian noise:** Add N(0, σ²) to expert waypoints
2. **Lateral offset:** Shift trajectory sideways
3. **Speed perturbation:** Scale velocities randomly
4. **Collision injection:** Modify trajectory to hit an agent
5. **Wrong mode:** Use trajectory from a different scenario

### Loss Functions

```python
# Binary Classification
loss = BCE(sigmoid(score), label)

# Ranking (margin)
loss = max(0, margin - score_pos + score_neg)

# InfoNCE (contrastive)
loss = -log(exp(s_pos/τ) / sum(exp(s_k/τ)))

# Combined
loss = λ₁·BCE + λ₂·Ranking + λ₃·InfoNCE
```

## 6. Integration Patterns

### Pattern 1: Score-and-Select
```
candidates = planner.generate(K=64)
scores = scorer.score_all(candidates, scene)
best = candidates[argmax(scores)]
```

### Pattern 2: Diffusion Guidance
```
trajectory = noise
for t in reversed(timesteps):
    traj = denoise(traj, t, scene)
    traj += lr * ∇_traj Score(traj, scene)  # gradient guidance
```

### Pattern 3: CEM (Cross-Entropy Method)
```
distribution = Normal(0, 1)
for iteration in range(5):
    samples = distribution.sample(K=100)
    scores = scorer(samples, scene)
    elite = top_10%(samples, scores)
    distribution = fit_distribution(elite)
best = distribution.mean
```

## 7. Evaluation Metrics

### Planning Quality
- **ADE (Average Displacement Error):** Mean L2 error across time
- **FDE (Final Displacement Error):** L2 at last timestep
- **Miss Rate:** Fraction with FDE > threshold

### Safety
- **Collision Rate:** % of scenarios with collision
- **TTC:** Minimum time-to-collision distribution
- **Drivable Area Violation Rate**

### Comfort
- **Lateral Acceleration RMS**
- **Longitudinal Jerk RMS**
- **Yaw Rate RMS**

### Scorer-Specific
- **Top-1 Accuracy:** Does highest-scored trajectory match expert?
- **Top-5 Recall:** Is expert trajectory in top-5?
- **Ranking Correlation:** Spearman/Kendall between predicted and true ranking
