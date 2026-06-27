# GenAD: Generalized Predictive Model for Autonomous Driving

> Generate **diverse** driving trajectories using diffusion models, then score and select the best one -- like an image generator, but for driving paths.

---

## Table of Contents

1. [What is GenAD?](#what-is-genad)
2. [Diffusion Models: From Images to Trajectories](#diffusion-models-from-images-to-trajectories)
3. [Why Diffusion for Driving?](#why-diffusion-for-driving)
4. [Architecture](#architecture)
5. [Key Concepts for Beginners](#key-concepts-for-beginners)
6. [The Diffusion Process Step by Step](#the-diffusion-process-step-by-step)
7. [How Planning Works with Diffusion](#how-planning-works-with-diffusion)
8. [How It Works Step by Step](#how-it-works-step-by-step)
9. [Our Implementation](#our-implementation)
10. [Running the Code](#running-the-code)
11. [Connection to Planner Scorer](#connection-to-planner-scorer)
12. [References](#references)

---

## What is GenAD?

GenAD (2024) is an **end-to-end autonomous driving model** that uses **diffusion models** to generate multiple diverse trajectory candidates for the ego vehicle. Instead of predicting a single "best guess" trajectory (which can fail when multiple valid paths exist), GenAD learns the **distribution** of all valid trajectories and samples from it.

The core idea is simple but powerful:

1. **Learn** what valid driving trajectories look like (from expert demonstrations).
2. **Generate** many diverse candidates by sampling from the learned distribution.
3. **Score** each candidate and pick the best one.

This is fundamentally different from traditional regression-based planners that output a single trajectory. GenAD naturally handles the **multi-modal** nature of driving -- at any moment, there may be several equally valid things to do (change lanes, slow down, accelerate through a gap), and GenAD can represent all of them simultaneously.

**Paper:** "GenAD: Generalized Predictive Model for Autonomous Driving"  
**Year:** 2024  
**arXiv:** [2402.11502](https://arxiv.org/abs/2402.11502)

---

## Diffusion Models: From Images to Trajectories

You may have heard of diffusion models in the context of image generation (Stable Diffusion, DALL-E, Midjourney). These models generate images by starting from pure random noise and gradually "cleaning" it up into a realistic image. GenAD applies the exact same principle to driving trajectories.

### The Core Idea in Plain English

Imagine you have a photograph. You can destroy it by gradually adding static noise -- a little at first (image is slightly grainy), then more and more, until eventually it becomes pure random static with no trace of the original image.

Now imagine you could **reverse** that process: starting from pure static, you learn to gradually remove the noise, step by step, until a realistic image emerges. That is exactly what diffusion models do -- they learn the reverse process.

**For driving trajectories:**
- Instead of a 2D grid of pixels, we have a sequence of (x, y) waypoints describing a path.
- Instead of generating a realistic image, we generate a realistic driving trajectory.
- The "noise" is random jitter added to each waypoint's position.
- The model learns to remove that jitter, recovering a smooth, valid trajectory.

### From Pixels to Paths

| Image Diffusion | Trajectory Diffusion (GenAD) |
|-----------------|------------------------------|
| Start: random pixel noise | Start: random (x,y) waypoints |
| Condition: text prompt ("a cat on a sofa") | Condition: scene context (road layout, other cars) |
| Output: realistic image | Output: realistic driving trajectory |
| Each step: make pixels slightly less noisy | Each step: make waypoints slightly less random |
| Result: many different valid images | Result: many different valid trajectories |

---

## Why Diffusion for Driving?

### The Multi-Modality Problem

Driving is fundamentally **multi-modal**: at any decision point, multiple valid trajectories exist. Consider approaching a slow truck on a highway:

```
Option A: Change to left lane, overtake
Option B: Stay in lane, slow down, follow
Option C: Change to right lane if available
```

All three are valid! But traditional regression models struggle with this:

```
Regression (L2 loss):
   Option A:  -------> swerve left
   Option B:  -------> stay straight
   Average:   -------> drive into the truck's corner (!)
```

When you train a model to minimize average error across multiple valid behaviors, it often predicts the **average** of all options -- which may itself be invalid or dangerous. This is called "mode averaging."

### How Diffusion Solves This

Diffusion models do not predict a single output. They learn the **probability distribution** over all valid outputs:

```
Diffusion sampling:
   Sample 1:  -------> change lane left      (valid!)
   Sample 2:  -------> slow down and follow  (valid!)
   Sample 3:  -------> change lane right     (valid!)
   Sample 4:  -------> another lane change   (valid!)
   ...
   All 16 samples are individually valid trajectories!
```

Each sample from the diffusion model is a complete, coherent trajectory. You never get an "average" that falls between modes -- you get actual driving behaviors.

### Comparison of Trajectory Generation Approaches

| Approach | Multi-modal? | Diversity | Mode Averaging? | Example |
|----------|:---:|:---:|:---:|---------|
| L2 Regression | No | None | Yes (averages modes) | TransFuser |
| K-modes (fixed) | Limited | Fixed K clusters | Within each mode | VAD (6 modes) |
| CVAE (latent variable) | Yes | Depends on prior | Possible | PRECOG |
| **Diffusion (GenAD)** | **Yes** | **Unlimited** | **No** | **This model** |

---

## Architecture

```
                         CAMERA IMAGE (3, H, W)
                                  |
                                  v
                      +-----------------------+
                      |    Scene Encoder       |
                      |  (CNN: 3ch -> 64 ->   |
                      |   128 -> 256 features) |
                      +-----------+-----------+
                                  |
                                  v
                      Scene Context Tokens (B, N, 256)
                                  |
                                  |
             +--------------------+--------------------+
             |                                         |
             v                                         v
  +---------------------+                   +---------------------+
  |  FORWARD DIFFUSION  |                   |  REVERSE DIFFUSION  |
  |     (Training)      |                   |    (Inference)      |
  |                     |                   |                     |
  | Expert trajectory   |                   | Start: pure noise   |
  |   + random noise    |                   |   x_T ~ N(0, I)     |
  |   = noisy_traj      |                   |                     |
  |                     |                   | For t = T down to 0: |
  | Model predicts the  |                   |   Predict noise     |
  |   added noise       |                   |   Remove a little   |
  |                     |                   |   Add tiny noise    |
  | Loss = MSE(pred,    |                   |                     |
  |         true noise) |                   | End: clean traj     |
  +---------------------+                   +----------+----------+
                                                       |
                                                       v
                                            K Trajectory Samples
                                            (B, K, 12 waypoints, 2)
                                                       |
                                                       v
                                          +------------------------+
                                          |   Trajectory Scorer     |
                                          |                        |
                                          |  For each trajectory:  |
                                          |  score = f(traj, scene)|
                                          |                        |
                                          |  Pick: argmax(scores)  |
                                          +------------+-----------+
                                                       |
                                                       v
                                              Best Trajectory
                                           (12 waypoints x,y)
```

### Data Flow Summary

```
Training:   Image -> Encoder -> Scene -> [add noise to expert traj] -> predict noise -> MSE loss
Inference:  Image -> Encoder -> Scene -> [denoise from random]x100 -> K trajectories -> score -> best
```

---

## Key Concepts for Beginners

### 1. Diffusion Models

A **diffusion model** learns to generate data (trajectories) by reversing a gradual noise-adding process. The name comes from physics: like ink diffusing in water, information gradually "diffuses" away as noise is added, and the model learns to reverse this diffusion.

Two processes:
- **Forward process** (fixed, not learned): Gradually add Gaussian noise to real data until it becomes pure noise.
- **Reverse process** (learned): Gradually remove noise from random noise until realistic data emerges.

### 2. DDPM (Denoising Diffusion Probabilistic Model)

DDPM is the foundational algorithm for diffusion models (Ho et al., 2020). The key ideas:

- Define a **noise schedule** that controls how much noise is added at each step.
- Train a neural network to **predict the noise** that was added at any given step.
- At inference time, use the network to iteratively remove noise, one step at a time.

The training objective is remarkably simple:
```
Loss = MSE(predicted_noise, actual_noise)
```

That is it -- the model just learns to predict noise, and this is sufficient to generate high-quality samples.

### 3. Beta Schedule (Noise Schedule)

The **beta schedule** defines how much noise is added at each timestep t. In our implementation:

```python
betas = torch.linspace(1e-4, 0.02, num_diffusion_steps)
```

- `beta_1 = 0.0001` (very little noise at first)
- `beta_T = 0.02` (more noise at the end)
- **Linear schedule**: noise increases linearly from nearly zero to a maximum.

Related quantities:
- `alpha_t = 1 - beta_t` (how much of the signal survives at step t)
- `alpha_bar_t = product(alpha_1, ..., alpha_t)` (cumulative signal remaining)

A well-designed schedule ensures:
- Early steps preserve most of the original signal (model makes fine adjustments).
- Late steps have mostly noise (model makes coarse structure decisions).

### 4. Denoising (Noise Prediction)

The core task of the neural network: given a noisy trajectory and the current timestep, **predict what noise was added**. Once you know the noise, you can subtract it to get a cleaner version.

```
Input:  noisy trajectory + timestep + scene context
Output: predicted noise (same shape as trajectory)
```

The network architecture uses:
- **Sinusoidal time embedding**: encodes the timestep as a high-dimensional vector (same trick as positional encoding in transformers).
- **Self-attention**: lets different waypoints in the trajectory attend to each other.
- **Cross-attention**: lets the trajectory attend to scene context (so denoising is conditioned on what the model sees).

### 5. Conditional Generation

GenAD does not generate arbitrary trajectories -- it generates trajectories **conditioned on the current driving scene**. The scene context (encoded from camera images) tells the diffusion model:
- Where the road is.
- Where other vehicles are.
- What the traffic situation looks like.

This conditioning is injected via **cross-attention**: at every denoising layer, the trajectory tokens can "look at" the scene tokens to guide the denoising toward contextually appropriate trajectories.

### 6. Timestep Embedding

The diffusion model needs to know "how noisy is this input?" so it can calibrate its denoising. The timestep t (0 = clean, T = very noisy) is encoded using **sinusoidal embeddings**:

```python
# Same math as transformer positional encoding
emb = sin(t * frequencies), cos(t * frequencies)
```

This gives the model a rich representation of "where we are in the denoising process" -- early steps need fine corrections, late steps need major structural changes.

---

## The Diffusion Process Step by Step

### Forward Process (Adding Noise) -- Used in Training

```
t=0 (clean)      t=25            t=50            t=75           t=100 (pure noise)
                                                                 
   *              *               .   .           . .  .         .  . .  .
   *              *   .           .  .  .         .  . . .       . .  . .  
   *              *              .   .            .   . .        .  .  . .
   *              *  .           .  .             . .  .  .      . .  .  .
   *              *             .  .  .           .  .   .       .  . .  .
   *               *           .   .  .          . .  .         . .  .  .
                                                                 
[smooth path]  [slightly      [moderately       [very           [random
                jittered]      noisy]            noisy]          points]
```

Mathematically, at each timestep t:
```
x_t = sqrt(alpha_bar_t) * x_0  +  sqrt(1 - alpha_bar_t) * noise

where:
  x_0         = original clean trajectory (expert demonstration)
  noise       = random Gaussian noise ~ N(0, I)
  alpha_bar_t = cumulative product of (1 - beta) up to step t
```

As t increases, the signal term shrinks and the noise term grows.

### Reverse Process (Removing Noise) -- Used at Inference

```
t=100 (noise)   t=75            t=50            t=25            t=0 (clean)

.  . .  .       . .  .          .  .             *  .            *
. .  . .        .  . .           .  .            *   .           *
.  .  . .       .   .            . .             *              *
. .  .  .       . .              .  .            *  .            *
.  . .  .       .  .  .         .   .            *              *
. .  .  .       . .             .  .              *             *

[random         [vague          [trajectory      [nearly        [clean
 points]         shape]          emerging]        clean]         trajectory]
```

At each step, the model:
1. Takes the current noisy trajectory x_t and timestep t.
2. Predicts what noise is present.
3. Computes an estimate of the clean trajectory x_0.
4. Takes a small step toward x_0 (with a tiny bit of fresh noise for diversity).

After all T steps, we have a clean, valid trajectory.

### Why Add Noise During Reverse?

You might wonder: if we are trying to remove noise, why add a little back at each step? This is crucial for **diversity**. Without the added noise, every sample starting from different random noise would converge to the same trajectory. The small injected noise at each step maintains stochasticity, allowing different samples to explore different valid modes.

---

## How Planning Works with Diffusion

GenAD's planning strategy follows a **generate-then-select** paradigm:

```
Step 1: GENERATE (Diffusion)
+---------------------------------------------------+
|  Run reverse diffusion K times (e.g., K=16)       |
|                                                    |
|  Each run starts from different random noise       |
|  -> Each produces a different valid trajectory     |
|                                                    |
|  Result: K diverse trajectory candidates           |
|    traj_1: [lane change left, smooth curve]       |
|    traj_2: [slow down, follow vehicle ahead]      |
|    traj_3: [lane change right, aggressive]        |
|    traj_4: [gentle slow + slight left adjust]     |
|    ...                                             |
|    traj_16: [maintain speed, stay in lane]        |
+---------------------------------------------------+
                         |
                         v
Step 2: SCORE (Trajectory Scorer)
+---------------------------------------------------+
|  For each trajectory, compute a quality score:    |
|                                                    |
|  score(traj_i) = f(traj_i, scene_context)         |
|                                                    |
|  The scorer considers:                            |
|    - Does this trajectory stay on the road?       |
|    - Does it avoid collisions?                    |
|    - Is it smooth and comfortable?                |
|    - Does it make progress on the route?          |
+---------------------------------------------------+
                         |
                         v
Step 3: SELECT (argmax)
+---------------------------------------------------+
|  best_trajectory = argmax(scores)                 |
|                                                    |
|  Output the highest-scoring trajectory as the     |
|  final driving plan.                              |
+---------------------------------------------------+
```

### Why is This Better Than Direct Regression?

| Direct Regression | Generate-then-Select (GenAD) |
|-------------------|------------------------------|
| One shot: must get it right | Multiple chances: generate many options |
| Mode averaging risk | Each sample is individually valid |
| Cannot express uncertainty | Spread of samples shows uncertainty |
| Single output, take it or leave it | Can apply external scoring/constraints |

---

## How It Works Step by Step

Here is what happens when GenAD processes a single driving frame:

### Training (Learning to Denoise)

```python
# Step 1: Encode the scene
scene_context = model.encode_scene(camera_image)
# camera_image: (B, 3, H, W) -> scene_context: (B, N, 256) tokens

# Step 2: Sample a random timestep for each batch element
t = random_int(0, num_steps)  # e.g., t=42 out of 100

# Step 3: Add noise to the expert trajectory
noisy_traj, true_noise = model.add_noise(expert_trajectory, t)
# expert_trajectory: (B, 12, 2) -- 12 waypoints, each (x, y)
# noisy_traj: same shape, but corrupted with Gaussian noise

# Step 4: Predict the noise
predicted_noise = model.diffusion(noisy_traj, t, scene_context)

# Step 5: Compute loss
loss = MSE(predicted_noise, true_noise)
# That is the entire training objective!
```

### Inference (Generating Trajectories)

```python
# Step 1: Encode the scene (same as training)
scene_context = model.encode_scene(camera_image)

# Step 2: Start from pure random noise (K=16 samples)
x = random_normal(shape=(B*16, 12, 2))

# Step 3: Iteratively denoise (100 steps)
for t in [99, 98, 97, ..., 1, 0]:
    # Predict noise at current step
    predicted_noise = model.diffusion(x, t, scene_context)
    
    # Estimate clean trajectory
    x0_pred = (x - sqrt(1-alpha_t) * predicted_noise) / sqrt(alpha_t)
    
    # Take a denoising step (with small added noise for diversity)
    x = step_toward_x0(x, x0_pred, t)

# Step 4: x is now 16 clean trajectory samples
trajectories = x.reshape(B, 16, 12, 2)

# Step 5: Score each trajectory
scores = model.scorer(trajectories, scene_context)  # (B, 16)

# Step 6: Pick the best
best_trajectory = trajectories[argmax(scores)]  # (B, 12, 2)
```

### Inside the Denoising Network

The neural network that predicts noise has this structure:

```python
# For each denoising layer (4 layers total):

# 1. Inject timestep information (how noisy is the input?)
x = x + time_mlp(time_embedding(t))

# 2. Self-attention among waypoints
#    Waypoint 5 can look at waypoint 1 and 12 for global coherence
x = self_attention(x, x, x)

# 3. Cross-attention to scene context
#    Waypoints attend to scene tokens (road, cars, traffic)
x = cross_attention(x, scene_context, scene_context)

# 4. Feed-forward transformation
x = feed_forward_network(x)

# After all layers:
predicted_noise = output_head(x)  # (B, 12, 2)
```

---

## Our Implementation

This is a **simplified reference implementation** focused on clarity. It demonstrates the core diffusion-based trajectory generation concept from GenAD.

### Model Components

| Component | Purpose | Parameters |
|-----------|---------|-----------|
| Scene Encoder | CNN: camera image -> spatial feature tokens | 3-layer CNN (3->64->128->256) |
| Sinusoidal Time Embedding | Encode diffusion timestep | Computed (no learned params) |
| Trajectory Diffusion Model | Denoise trajectories conditioned on scene | 4-layer transformer with cross-attention |
| Trajectory Scorer | Evaluate and rank generated trajectories | MLP (traj + scene -> score) |

### Comparison with Original

| Aspect | Original GenAD | Our Implementation |
|--------|----------------|--------------------|
| Scene encoder | BEV encoder + multi-camera backbone | Simple 3-layer CNN |
| Diffusion model | Large transformer with many heads | 4-layer, 8-head transformer |
| Num diffusion steps | 1000 | 100 (configurable) |
| Waypoints | Varies | 12 waypoints |
| Scorer | Learned + safety constraints | Simple MLP scorer |
| Map/route conditioning | Yes | No (simplified) |
| Parameters | Large (full driving stack) | **5.2M** |
| Training pipeline | Multi-GPU, full dataset | Single-file reference |

### What is Preserved

- The core diffusion paradigm: forward noise addition + reverse denoising.
- DDPM noise schedule (linear beta schedule).
- Conditional generation via cross-attention to scene context.
- Multi-sample generation and scoring.
- Epsilon prediction (predict noise, not trajectory directly).

### What is Simplified

- No BEV encoding or multi-camera setup.
- No map or route conditioning.
- Fewer diffusion steps (faster demo, same concept).
- Simple MLP scorer (production would use a richer scorer).
- No DDIM or other accelerated sampling strategies.

---

## Running the Code

### Prerequisites

```bash
pip install torch  # PyTorch >= 1.10
```

### Run the Demo

```bash
python model.py
```

This will:
1. Instantiate the GenAD model (5.2M parameters) with 20 diffusion steps.
2. Create dummy camera input (2 batch, 3 channels, 128x256).
3. Run full inference: encode scene -> diffusion sampling (8 trajectories) -> score -> pick best.
4. Print trajectory shapes and scores.

**Expected output:**
```
GenAD: Generative End-to-End Driving Demo
==================================================
Parameters: 5,2XX,XXX
Generated 8 diverse trajectories
Trajectories shape: torch.Size([2, 8, 6, 2])
Scores: torch.Size([2, 8])
Best trajectory: torch.Size([2, 6, 2])

Key advantage: captures MULTIPLE valid driving behaviors!
(e.g., lane change vs slow down -- both generated as options)
```

### Using in Your Own Code

```python
import torch
from model import GenAD

# Create model
model = GenAD(
    scene_dim=256,            # scene feature dimension
    hidden_dim=512,           # transformer hidden dimension
    num_waypoints=12,         # predict 12 future waypoints
    num_diffusion_steps=100,  # denoising steps (more = higher quality)
).eval()

# Prepare input
image = torch.randn(1, 3, 256, 512)  # front camera image

# Generate diverse trajectories and pick the best
with torch.no_grad():
    output = model(image, num_samples=16)

# Access results
trajectories = output['trajectories']    # (1, 16, 12, 2) -- 16 diverse candidates
scores = output['scores']                # (1, 16) -- quality score for each
best_traj = output['best_trajectory']    # (1, 12, 2) -- highest-scoring trajectory
best_idx = output['best_idx']            # (1,) -- index of best trajectory
```

### Training Loop (Sketch)

```python
model = GenAD(num_diffusion_steps=100)
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

for batch in dataloader:
    images = batch['camera']           # (B, 3, H, W)
    expert_traj = batch['trajectory']  # (B, 12, 2) expert waypoints

    # Encode scene
    scene_context = model.encode_scene(images)

    # Compute diffusion loss (simple noise prediction MSE)
    loss = model.training_loss(expert_traj, scene_context)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

---

## Connection to Planner Scorer

GenAD and the Planner Scorer are **natural partners**. Their combination represents a powerful paradigm for autonomous driving:

### The Generate-Score Pipeline

```
                GenAD (Generator)              Planner Scorer (Evaluator)
         +--------------------------+    +------------------------------+
         |                          |    |                              |
Input -> | Diffusion generates K    | -> | Score each trajectory on:   | -> Best
Scene    | diverse, valid           |    |   - Safety (collision risk)  |    Trajectory
         | trajectory candidates    |    |   - Comfort (smoothness)     |
         |                          |    |   - Efficiency (progress)    |
         +--------------------------+    |   - Compliance (rules)       |
                                         +------------------------------+
```

### Why They Complement Each Other

| GenAD's Strength | Scorer's Strength |
|------------------|-------------------|
| Generates **diverse** options | Evaluates based on **criteria** |
| Captures multi-modal behaviors | Enforces safety constraints |
| Does not need to know "best" | Does not need to generate |
| Creative exploration | Disciplined selection |

### Separation of Concerns

This separation is architecturally elegant:

1. **Generator (GenAD):** "Here are 16 things we could do." -- Focuses purely on generating valid, diverse trajectories without worrying about which is best.

2. **Scorer:** "This one is best because it is safe, comfortable, and efficient." -- Focuses purely on evaluation without needing to imagine new possibilities.

Neither module needs to do both jobs. The generator does not need a complex reward function, and the scorer does not need to search the space of all possible trajectories.

### Practical Benefits

- **Modularity:** You can upgrade the scorer independently of the generator (e.g., add new safety rules) without retraining GenAD.
- **Interpretability:** The scorer can explain WHY a trajectory was chosen (high safety score, low jerk, follows route).
- **Flexibility:** At test time, you can adjust scoring weights (more conservative = weight safety higher) without changing the generation model.
- **Robustness:** Even if the generator produces a few poor candidates, the scorer filters them out. Even if the scorer has imperfect criteria, having many options means a good one is likely among them.

### Connection to Our Planner Scorer Module

The `planner_scorer/` directory in this repository implements various scoring approaches (classical cost functions, learned neural scorers, contrastive scorers) that can be paired with GenAD's trajectory outputs. See `planner_scorer/README.md` for details on the scoring formulations.

---

## References

1. **GenAD (2024):** Zhiqi Li, Zhiding Yu, et al. "GenAD: Generalized Predictive Model for Autonomous Driving." arXiv:2402.11502. [Paper](https://arxiv.org/abs/2402.11502)

2. **DDPM (2020):** Ho, J., Jain, A., & Abbeel, P. "Denoising Diffusion Probabilistic Models." NeurIPS 2020. [arXiv:2006.11239](https://arxiv.org/abs/2006.11239) -- The foundational diffusion model paper.

3. **DDIM (2020):** Song, J., Meng, C., & Ermon, S. "Denoising Diffusion Implicit Models." ICLR 2021. [arXiv:2010.02502](https://arxiv.org/abs/2010.02502) -- Accelerated sampling (fewer steps needed).

4. **Diffuser (2022):** Janner, M., Du, Y., Tenenbaum, J., & Levine, S. "Planning with Diffusion for Flexible Behavior Synthesis." ICML 2022. -- Diffusion for robot trajectory planning.

5. **CTG (2023):** Zhong, Z., et al. "Guided Conditional Diffusion for Controllable Traffic Generation." NeurIPS 2023. -- Diffusion for traffic scenario generation.

6. **Transformers (2017):** Vaswani, A., et al. "Attention Is All You Need." NeurIPS 2017. -- The attention mechanism used in our denoising network.

---

## Files in This Directory

```
GenAD/
  README.md   -- This documentation (you are here)
  model.py    -- GenAD model implementation (5.2M params)
                 Includes: SinusoidalTimeEmbedding,
                 TrajectoryDiffusionModel, GenAD, demo()
```
