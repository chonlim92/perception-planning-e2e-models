# TransFuser: Multi-Modal Fusion Transformer for End-to-End Autonomous Driving

> Fuse camera images and LiDAR at **every level** of the network using transformer attention, then directly predict driving waypoints -- no hand-crafted perception pipeline needed.

---

## Table of Contents

1. [What is TransFuser?](#what-is-transfuser)
2. [Why Fuse at Multiple Scales?](#why-fuse-at-multiple-scales)
3. [Architecture](#architecture)
4. [Key Concepts](#key-concepts)
5. [How It Works Step by Step](#how-it-works-step-by-step)
6. [One-Step vs Two-Step](#one-step-vs-two-step)
7. [CARLA Simulator](#carla-simulator)
8. [Our Implementation](#our-implementation)
9. [Running the Code](#running-the-code)
10. [Results](#results)
11. [References](#references)

---

## What is TransFuser?

TransFuser (CVPR 2022, extended in IEEE TPAMI 2023) is an **end-to-end autonomous driving model**. It takes raw sensor data -- a front-facing camera image and a LiDAR Bird's-Eye View (BEV) map -- and directly outputs a sequence of future waypoints that tell the car where to drive.

The key innovation is **multi-scale transformer fusion**: instead of processing camera and LiDAR independently and combining them only at the end, TransFuser lets the two modalities "talk" to each other at every stage of the feature extraction process. This means:

- The camera branch can leverage LiDAR depth information early on.
- The LiDAR branch can benefit from camera texture and color cues.
- Both modalities progressively refine each other as features become more abstract.

Think of it as two teammates constantly sharing notes at every step of a project, rather than only comparing finished reports at the end.

**Paper:** "TransFuser: Imitation with Transformer-Based Sensor Fusion for Autonomous Driving"  
**Authors:** Kashyap Chitta, Aditya Prakash, Bernhard Jaeger, Zehao Yu, Katrin Renz, Andreas Geiger  
**Venue:** CVPR 2022 (initial), IEEE TPAMI 2023 (extended)  
**arXiv:** [2205.15997](https://arxiv.org/abs/2205.15997)  
**Original Code:** https://github.com/autonomousvision/transfuser

---

## Why Fuse at Multiple Scales?

In autonomous driving, camera images are rich in texture, color, and semantics (lane markings, traffic lights), while LiDAR provides accurate 3D geometry and depth. Neither sensor alone tells the full story. The question is: **when** should we combine them?

### Three Fusion Strategies

| Strategy | When fusion happens | Limitation |
|----------|-------------------|------------|
| **Early fusion** | Combine raw inputs before any processing | Loses modality-specific patterns; raw data formats are very different |
| **Late fusion** | Combine only final high-level features | By the time information is shared, low-level geometric/texture cues are lost |
| **Multi-scale fusion (TransFuser)** | Combine at every processing stage | Gets the best of both worlds |

### An Analogy: Translating a Novel

Imagine two people translating a novel from different languages -- one translates from French, the other from Japanese. They are translating the same story, but each picks up different nuances.

- **Late fusion** = They each finish their translation independently, then compare notes at the very end. They catch major mistakes, but subtle context is already lost.
- **Early fusion** = They try to merge the raw French and Japanese text before reading it. The jumbled mixture confuses more than it helps.
- **Multi-scale fusion** = After every chapter, they share summaries and check each other's understanding. Early on they share basic plot points; later they compare character motivations and themes. By the end, both translations benefit from the other's unique perspective at every level of detail.

TransFuser uses the multi-scale approach. At each ResNet stage (processing level), a transformer attention mechanism lets camera features attend to LiDAR features and vice versa. This progressive sharing produces richer, more robust representations than any single-point fusion strategy.

---

## Architecture

```
             CAMERA IMAGE (3, 256, 512)          LiDAR BEV (2, 256, 256)
                      |                                    |
                      v                                    v
              +---------------+                   +---------------+
              |   Image Stem  |                   |  LiDAR Stem   |
              | (7x7 conv,    |                   | (7x7 conv,    |
              |  pool -> 64ch)|                   |  pool -> 64ch)|
              +-------+-------+                   +-------+-------+
                      |                                    |
                      v                                    v
              +---------------+                   +---------------+
              | ResNet Stage 1|                   | ResNet Stage 1|
              |   (64 ch)     |                   |   (64 ch)     |
              +-------+-------+                   +-------+-------+
                      |                                    |
                      +------> Transformer Fusion 1 <------+
                      |         (4 attn heads, 64ch)       |
                      v                                    v
              +---------------+                   +---------------+
              | ResNet Stage 2|                   | ResNet Stage 2|
              |   (128 ch)    |                   |   (128 ch)    |
              +-------+-------+                   +-------+-------+
                      |                                    |
                      +------> Transformer Fusion 2 <------+
                      |         (4 attn heads, 128ch)      |
                      v                                    v
              +---------------+                   +---------------+
              | ResNet Stage 3|                   | ResNet Stage 3|
              |   (256 ch)    |                   |   (256 ch)    |
              +-------+-------+                   +-------+-------+
                      |                                    |
                      +------> Transformer Fusion 3 <------+
                      |         (8 attn heads, 256ch)      |
                      v                                    v
              +---------------+                   +---------------+
              | ResNet Stage 4|                   | ResNet Stage 4|
              |   (512 ch)    |                   |   (512 ch)    |
              +-------+-------+                   +-------+-------+
                      |                                    |
                      +------> Transformer Fusion 4 <------+
                      |         (8 attn heads, 512ch)      |
                      v                                    v
              +-------+-------+                   +-------+-------+
              | Global AvgPool|                   | Global AvgPool|
              |   (512-dim)   |                   |   (512-dim)   |
              +-------+-------+                   +-------+-------+
                      |                                    |
                      +------------ Concat ----------------+
                                       |
                                       v  (1024-dim)
                              +--------+--------+
                              | Linear Proj     |
                              | (1024 -> 512)   |
                              +--------+--------+
                                       |
                                       +  <-- Speed Embedding (ego speed -> 512-dim)
                                       |
                                       v
                              +--------+--------+
                              |  GRU Decoder    |
                              |  (4 steps)      |
                              +--------+--------+
                                       |
                                       v
                              4 Waypoints (x, y)
                              [0.5s, 1.0s, 1.5s, 2.0s into the future]
                                       |
                                       v
                              +--------+--------+
                              |  PID Controller |  (at inference time)
                              |  waypoints ->   |
                              |  steer/throttle |
                              +--------+--------+
                                       |
                                       v
                              Vehicle Control Signals
```

---

## Key Concepts

### 1. Multi-Modal Fusion

**Multi-modal** means using multiple types of sensor data (here: camera + LiDAR). **Fusion** means combining information from these different sources into a unified representation. TransFuser's contribution is that it fuses modalities at multiple resolutions, not just once.

### 2. ResNet Stages

[ResNet](https://arxiv.org/abs/1512.03385) (Residual Network) is a well-known image classification backbone. It processes images through a series of "stages," where each stage:
- Doubles the number of feature channels (64 -> 128 -> 256 -> 512).
- Halves the spatial resolution (feature maps get smaller).
- Extracts increasingly abstract features (edges -> textures -> parts -> objects).

TransFuser uses two parallel ResNets:
- **Image branch:** ResNet-34 style (3+4+6+3 blocks) for the camera image.
- **LiDAR branch:** ResNet-18 style (2+2+2+2 blocks) for the LiDAR BEV.

### 3. Transformer Attention for Fusion

At each stage, a `TransformerFusionBlock` lets image and LiDAR features "attend" to each other:

1. **Flatten** both feature maps into sequences of tokens (each spatial position becomes a token).
2. **Concatenate** image tokens and LiDAR tokens into one combined sequence.
3. **Apply self-attention** so every token can attend to every other token (camera pixels attend to LiDAR points and vice versa).
4. **Split** the result back into image and LiDAR feature maps.

This is analogous to having a meeting where camera pixels and LiDAR points can each ask questions of the other group and update their understanding accordingly.

### 4. GRU Waypoint Decoder

After all four stages of fusion and feature extraction, the model needs to produce a driving plan. It uses a [GRU](https://arxiv.org/abs/1406.1078) (Gated Recurrent Unit) to autoregressively predict 4 future waypoints:

- The GRU receives the fused feature vector as input at each step.
- At each step, it outputs one (x, y) waypoint in the ego vehicle's coordinate frame.
- Each waypoint represents where the car should be 0.5s, 1.0s, 1.5s, and 2.0s into the future.

### 5. PID Controller (Inference Only)

Predicted waypoints are not directly sent to the car's actuators. A PID controller converts them into low-level control:
- **Steer:** derived from the angle to the first waypoint.
- **Throttle/Brake:** derived from the difference between current speed and target speed.

### 6. Auxiliary BEV Segmentation (Training Only)

During training, an additional head predicts a BEV semantic segmentation map (road, vehicles, pedestrians, other). This auxiliary task improves the learned representations but is discarded at inference time -- the model remains "one-step" because no intermediate perception is used for driving decisions.

---

## How It Works Step by Step

Here is what happens when TransFuser processes a single frame:

```
Step 1: Receive sensor inputs
   - Front camera image:  (3, 256, 512)  -- RGB pixels
   - LiDAR BEV:           (2, 256, 256)  -- 2-channel height map
   - Ego speed:           scalar (m/s)

Step 2: Stem processing
   - Both inputs pass through a 7x7 convolution + max pool
   - Reduces spatial size by 4x, outputs 64-channel feature maps

Step 3: Stage 1 + Fusion 1
   - Image features: (64, 64, 128)   -- low-level edges, gradients
   - LiDAR features: (64, 64, 64)    -- basic geometric patterns
   - Transformer attention: lets camera see LiDAR depth, LiDAR see texture

Step 4: Stage 2 + Fusion 2
   - Image features: (128, 32, 64)   -- textures, lane-like patterns
   - LiDAR features: (128, 32, 32)   -- object-shaped clusters
   - Transformer attention: cross-modal alignment improves

Step 5: Stage 3 + Fusion 3
   - Image features: (256, 16, 32)   -- parts of objects (wheels, signs)
   - LiDAR features: (256, 16, 16)   -- distinct object boundaries
   - Transformer attention: semantic understanding develops

Step 6: Stage 4 + Fusion 4
   - Image features: (512, 8, 16)    -- high-level scene understanding
   - LiDAR features: (512, 8, 8)     -- full 3D scene layout
   - Transformer attention: final cross-modal refinement

Step 7: Feature aggregation
   - Global average pool both branches -> 512-dim each
   - Concatenate -> 1024-dim
   - Project to 512-dim hidden state

Step 8: Add speed context
   - Embed ego speed into a 512-dim vector
   - Add to the fused feature (so the model knows how fast we are going)

Step 9: GRU waypoint prediction
   - Run GRU for 4 steps
   - Each step outputs (dx, dy): relative position of a future waypoint
   - Result: 4 waypoints spanning 2 seconds into the future

Step 10: Control (inference only)
   - PID controller converts waypoints to steer/throttle/brake
   - Send commands to the vehicle
```

---

## One-Step vs Two-Step

End-to-end autonomous driving models come in two flavors:

| Aspect | One-Step (e.g., TransFuser) | Two-Step (e.g., UniAD, VAD) |
|--------|---------------------------|---------------------------|
| Pipeline | Sensors -> Driving actions | Sensors -> Perception -> Driving actions |
| Intermediate outputs | None (or only auxiliary during training) | Explicit: 3D boxes, maps, tracks |
| Interpretability | Lower (black-box internal features) | Higher (can inspect intermediate results) |
| Simplicity | Simpler pipeline | More complex, multi-stage |
| Error propagation | End-to-end gradients, no cascading | Errors in perception can cascade |
| Example models | TransFuser, MILE, LAV | UniAD, VAD, GameFormer |

**TransFuser is one-step** because:
- Raw sensor data (camera + LiDAR) goes directly to waypoints in a single forward pass.
- There are no explicit perception outputs (bounding boxes, HD maps) used in the driving decision.
- The BEV segmentation head is an auxiliary training signal only -- it is not used at inference time.
- The internal learned representations exist but are NOT exposed as interpretable intermediate results.

**Why does this matter?**
- One-step models are simpler to train and deploy (single loss, single network).
- Two-step models offer better interpretability (you can check if perception was correct before planning).
- The field is moving toward hybrid approaches that get the benefits of both.

---

## CARLA Simulator

[CARLA](https://carla.org/) (Car Learning to Act) is an open-source simulator for autonomous driving research. It provides:

- **Realistic 3D environments:** Multiple towns with varying road layouts, intersections, roundabouts.
- **Sensor simulation:** Cameras, LiDAR, radar, GPS, IMU -- all with configurable noise.
- **Dynamic agents:** NPC vehicles, pedestrians, traffic lights that follow realistic behavior.
- **Weather/lighting:** 8+ weather conditions (clear, rain, fog, night) for robustness testing.
- **Benchmarks:** Standardized route-based evaluation (Driving Score = Route Completion x Infraction Score).

**Why use CARLA?**
- Real-world data is expensive and dangerous to collect at scale.
- CARLA provides unlimited training data from an expert autopilot.
- Standardized benchmarks enable fair comparison between methods.
- Safe testing environment -- crashes in simulation cost nothing.

**Training in CARLA:**
1. An expert autopilot (with privileged simulator information) drives pre-defined routes.
2. At each timestep, the system records: camera image, LiDAR scan, ego speed, expert waypoints.
3. The model is trained via imitation learning: predict waypoints that match the expert.
4. ~90 hours of driving data across 8 weather conditions and 4 towns.

---

## Our Implementation

This is a **simplified reference implementation** of TransFuser's core architecture. Compared to the original:

| Aspect | Original TransFuser | Our Implementation |
|--------|--------------------|--------------------|
| Image backbone | Pre-trained ResNet-34 (ImageNet) | ResNet-34 style, trained from scratch |
| LiDAR backbone | Pre-trained ResNet-18 | ResNet-18 style, trained from scratch |
| Fusion | Transformer + positional encoding + multiple attention layers | Single-layer transformer per stage |
| Waypoints | 4 waypoints + speed prediction + traffic light | 4 waypoints only |
| Auxiliary tasks | BEV seg + depth + traffic light | BEV seg only |
| Parameters | ~60M+ | **40.2M** |
| Training infra | Multi-GPU, full CARLA pipeline | Single-file reference |

**What is preserved:**
- The core multi-scale fusion architecture (parallel ResNet branches + transformer fusion at each stage).
- GRU waypoint decoder with speed conditioning.
- PID controller for converting waypoints to control signals.
- BEV segmentation auxiliary head.

**What is simplified:**
- No ImageNet pre-training (backbone is defined from scratch).
- Fewer auxiliary tasks.
- No positional encoding in the transformer fusion blocks.
- No data loading, augmentation, or CARLA interface code.

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
1. Instantiate the TransFuser model (40.2M parameters).
2. Create dummy camera, LiDAR, and speed inputs.
3. Run a forward pass and print predicted waypoints.

**Expected output:**
```
TransFuser - End-to-End Driving Demo
=============================================
Parameters: 40,247,622
Device: cuda  (or cpu)

Inputs:
  Camera: torch.Size([2, 3, 256, 512])
  LiDAR BEV: torch.Size([2, 2, 256, 256])
  Speed: torch.Size([2, 1])

Outputs:
  Waypoints: torch.Size([2, 4, 2])
  BEV segmentation: torch.Size([2, 4, 64, 64])
  Waypoint values (batch 0):
    t=0.5s: (x.xxx, y.yyy)
    t=1.0s: (x.xxx, y.yyy)
    t=1.5s: (x.xxx, y.yyy)
    t=2.0s: (x.xxx, y.yyy)
```

### Using in Your Own Code

```python
import torch
from model import TransFuser, waypoints_to_control

# Create model
model = TransFuser(
    img_channels=3,       # RGB camera
    lidar_channels=2,     # 2-channel LiDAR BEV
    num_waypoints=4,      # predict 4 future waypoints
    hidden_dim=512,       # GRU hidden dimension
)

# Prepare inputs
image = torch.randn(1, 3, 256, 512)      # front camera
lidar_bev = torch.randn(1, 2, 256, 256)  # LiDAR bird's-eye view
speed = torch.tensor([[5.0]])             # ego speed in m/s

# Forward pass
output = model(image, lidar_bev, speed)
waypoints = output['waypoints']  # (1, 4, 2) -- 4 future (x, y) positions

# Convert to control (for CARLA)
steer, throttle, brake = waypoints_to_control(
    waypoints[0], speed=5.0, target_speed=4.0
)
```

---

## Results

### CARLA Leaderboard (Longest6 Benchmark)

| Method | Driving Score | Route Completion | Infraction Score |
|--------|:---:|:---:|:---:|
| TransFuser (PAMI 2023) | 61.18 | 86.69 | 0.71 |
| InterFuser | 68.31 | 95.02 | 0.72 |
| TCP | 75.14 | 93.64 | 0.81 |
| Human Expert | 84.97 | 99.43 | 0.85 |

**Metric definitions:**
- **Driving Score** = Route Completion x Infraction Score. The primary metric.
- **Route Completion** = Percentage of the route distance completed (0-100).
- **Infraction Score** = Multiplicative penalty for collisions, red light violations, etc. (0-1, higher is better).

TransFuser was one of the first methods to demonstrate competitive performance using end-to-end learning without explicit perception modules, establishing that multi-scale fusion is a powerful design choice for sensor fusion in driving.

---

## References

1. **TransFuser (CVPR 2022):** Chitta, K., Prakash, A., Jaeger, B., Yu, Z., Renz, K., & Geiger, A. "TransFuser: Imitation with Transformer-Based Sensor Fusion for Autonomous Driving." CVPR 2022.

2. **TransFuser++ (PAMI 2023):** Chitta, K., Prakash, A., Jaeger, B., Yu, Z., Renz, K., & Geiger, A. "TransFuser: Imitation with Transformer-Based Sensor Fusion for Autonomous Driving." IEEE TPAMI, 2023. [arXiv:2205.15997](https://arxiv.org/abs/2205.15997)

3. **CARLA Simulator:** Dosovitskiy, A., Ros, G., Codevilla, F., Lopez, A., & Koltun, V. "CARLA: An Open Urban Driving Simulator." CoRL 2017. [carla.org](https://carla.org/)

4. **ResNet:** He, K., Zhang, X., Ren, S., & Sun, J. "Deep Residual Learning for Image Recognition." CVPR 2016.

5. **Transformers:** Vaswani, A., et al. "Attention Is All You Need." NeurIPS 2017.

6. **GRU:** Cho, K., et al. "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation." EMNLP 2014.

---

---

## Training

### Quick Start

```bash
python train.py --epochs 5 --batch_size 1
```

Runs with synthetic CARLA-like data (no simulator needed).

### Loss Functions

| Loss | Source | Weight | Purpose |
|:---|:---:|:---:|:---|
| Waypoint L1 | `[FROM PAPER]` | 1.0 | Future waypoint regression |
| BEV Segmentation CE | `[FROM PAPER]` | 0.5 | BEV semantic map auxiliary |
| Speed L1 | `[FROM PAPER]` | 0.1 | Speed prediction |
| Target Point L1 | `[SELF-IMPLEMENTED]` | 0.2 | GPS target point regression |

### Key Arguments

```bash
python train.py \
    --epochs 50 \
    --batch_size 4 \
    --lr 1e-4 \
    --num_waypoints 4 \
    --lidar_input True \
    --num_samples 200 \
    --resume checkpoint.pth
```

### What the Training Script Includes

- **Multi-modal input** (camera RGB + LiDAR BEV) with fusion `[FROM PAPER]`
- **Multi-scale transformer fusion** at 4 resolution levels `[FROM PAPER]`
- **Auxiliary BEV segmentation** for representation learning `[FROM PAPER]`
- **Speed prediction head** as regularizer `[FROM PAPER]`
- **Validation metrics:** waypoint L1, BEV IoU, speed MAE
- **Mixed precision + gradient clipping** `[SELF-IMPLEMENTED]`
- **Cosine annealing LR** `[SELF-IMPLEMENTED]`

## Files in This Directory

```
TransFuser/
  README.md   -- This documentation (you are here)
  model.py    -- TransFuser model implementation (40.2M params)
  train.py    -- Complete training pipeline (890+ lines)
```
