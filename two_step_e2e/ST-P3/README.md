# ST-P3: End-to-End Vision-Based Autonomous Driving via Spatial Temporal Feature Learning

**Paper:** "ST-P3: End-to-end Vision-based Autonomous Driving via Spatial Temporal Feature Learning"  
**Authors:** Shengchao Hu, Li Chen, Penghao Wu, Hongyang Li, Junchi Yan, Dacheng Tao  
**Venue:** ECCV 2022  
**arXiv:** https://arxiv.org/abs/2207.07601  
**Code:** https://github.com/OpenDriveLab/ST-P3

> One of the earliest two-step end-to-end autonomous driving models: perception features flow directly to planning, with temporal reasoning across multiple frames.

---

## What is ST-P3?

ST-P3 (Spatial Temporal feature learning for Perception, Prediction, and Planning) is a pioneering end-to-end autonomous driving model that takes multi-camera video as input and directly outputs a planned driving trajectory. Instead of treating driving as a collection of isolated tasks, ST-P3 jointly learns perception (understanding the scene), prediction (anticipating the future), and planning (deciding where to drive) in a single unified network.

What makes ST-P3 special among early E2E models is its explicit use of **spatial-temporal features**. Rather than processing only the current camera frame (a single snapshot in time), ST-P3 processes a sequence of past frames and uses a recurrent neural network to build up a memory of how the scene has been changing over time. This temporal understanding is critical for safe driving.

**In simple terms:** ST-P3 watches the road like a human driver -- not just glancing at one freeze-frame, but continuously tracking how cars move, how the road unfolds, and what is likely to happen next.

---

## Why Time Matters in Driving

Imagine you are approaching an intersection and see a car to your left. From a single photo, you cannot tell:
- Is the car moving toward you or away from you?
- Is it accelerating or braking?
- Did it just enter the intersection or has it been waiting?

Now imagine you have the last 2 seconds of video. Suddenly you can answer all of these questions. This is exactly why temporal information is critical:

| Single Frame (No Time) | Multiple Frames (With Time) |
|------------------------|----------------------------|
| See objects but not motion | See objects AND their motion |
| Cannot determine velocity | Can estimate speed and direction |
| No idea if car is braking | Can detect acceleration/deceleration |
| Static scene understanding | Dynamic scene understanding |
| Ambiguous intentions | Clearer predictions of what happens next |

**ST-P3's key insight:** By aggregating BEV (Bird's Eye View) features across multiple past frames using a ConvGRU, the model builds an internal representation that captures motion, speed, and trajectory patterns -- all essential for predicting the future and planning safe paths.

---

## Architecture

```
  Input: Multi-view cameras at times [t-3, t-2, t-1, t]
  ========================================================

  For EACH timestep t_i:
  ┌───────────────────────────────────────────────────────┐
  │  6 Camera Images (front, front-left, front-right,     │
  │                   back, back-left, back-right)         │
  │         │                                              │
  │         ▼                                              │
  │  [Image Backbone] ─── CNN feature extraction           │
  │         │              per camera                      │
  │         ▼                                              │
  │  [Lift-Splat-Shoot (LSS)] ─── project to BEV          │
  │         │                                              │
  │         ▼                                              │
  │  BEV Features (top-down feature map)                   │
  └────────────────────────┬──────────────────────────────┘
                           │
                           ▼
  ┌────────────────────────────────────────────────────────┐
  │         Temporal Aggregation (ConvGRU)                  │
  │                                                        │
  │   BEV(t-3) ──→ [GRU] ──→ hidden_1                     │
  │   BEV(t-2) ──→ [GRU] ──→ hidden_2                     │
  │   BEV(t-1) ──→ [GRU] ──→ hidden_3                     │
  │   BEV(t)   ──→ [GRU] ──→ hidden_4  (final state)      │
  │                              │                         │
  └──────────────────────────────┼─────────────────────────┘
                                 │
                                 ▼
                    [BEV Spatial Encoder]
                    (further spatial processing)
                                 │
                ┌────────────────┼────────────────┐
                │                │                │
                ▼                ▼                ▼
        ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
        │  BEV Seg     │ │  Future Occ  │ │   Planning   │
        │  Head        │ │  Head        │ │   (GRU)      │
        │              │ │              │ │              │
        │  road/lane/  │ │  where will  │ │  waypoint    │
        │  vehicle/    │ │  objects be  │ │  prediction  │
        │  background  │ │  in 1-5s?    │ │  for ego car │
        └──────────────┘ └──────────────┘ └──────────────┘

  Outputs: BEV segmentation map + future occupancy + ego trajectory
```

---

## Key Concepts

### Lift-Splat-Shoot (LSS)

LSS is a method to transform 2D camera images into a 3D Bird's Eye View (BEV) representation. This is fundamental because planning needs to reason about the world in metric space (meters), not pixel space.

**The three steps:**

1. **Lift:** For each pixel in the image, predict a distribution over depth values (how far away is this pixel?). This "lifts" 2D pixels into 3D space by associating each pixel with possible 3D locations.

2. **Splat:** Take all these 3D points (from all cameras) and project them down onto a flat 2D grid viewed from above -- the BEV grid. Features from points that land in the same BEV cell are summed together (this is the "splatting").

3. **Shoot:** (Optional) Process the BEV grid for downstream tasks. In ST-P3, the BEV features continue into temporal aggregation and task heads.

```
   Camera Image              3D Frustum           BEV Grid (top-down)
   ┌──────────┐           /  .  .  .  /|         ┌──────────────┐
   │  pixel   │  Lift    /  .  .  .  / |  Splat  │ ■  ■         │
   │  with    │ ──────→ /  .  .  .  /  | ──────→ │    ■  ■      │
   │  depth   │        /__________/   |         │       ■      │
   │  dist.   │        |__________|   /         │  car    road  │
   └──────────┘                    /           └──────────────┘
```

**Why BEV?** In BEV, distances are in meters, objects have true sizes, and the representation is natural for downstream tasks like path planning (which also operates in metric 2D space).

### ConvGRU (Convolutional Gated Recurrent Unit)

A ConvGRU is the spatial version of a standard GRU recurrent neural network. While a regular GRU processes 1D vectors over time, a ConvGRU processes 2D feature maps (like BEV maps) over time, preserving spatial structure.

**How it works:**

```
At each timestep t:
  Input:    current BEV features x_t     (shape: C x H x W)
  Previous: hidden state h_{t-1}         (shape: C x H x W)

  1. Reset gate r = sigmoid(Conv([x_t, h_{t-1}]))
     "How much of the past memory should I forget?"

  2. Update gate z = sigmoid(Conv([x_t, h_{t-1}]))
     "How much should I update vs. keep the old state?"

  3. Candidate h' = tanh(Conv([x_t, r * h_{t-1}]))
     "What's the new information to potentially remember?"

  4. Output h_t = (1-z) * h_{t-1} + z * h'
     "Blend old memory with new information"
```

**Why ConvGRU for driving?** It allows the model to:
- Track moving objects across frames (a car that was "here" last frame is now "there")
- Build up confidence about static objects (the road boundary was here 4 frames in a row -- it is reliable)
- Capture velocity and acceleration patterns spatially

### BEV Segmentation

BEV segmentation means classifying each cell in the top-down BEV grid into semantic categories:

| Class | What it means |
|-------|---------------|
| Road/Drivable Area | Where the ego car CAN drive |
| Lane Boundary | Lane markings and dividers |
| Vehicle | Other cars, trucks, buses |
| Background | Everything else (sidewalks, buildings) |

This gives the planning module a clear understanding of "where can I go" and "what should I avoid."

### Future Occupancy Prediction

Future occupancy prediction answers: "Which BEV cells will be occupied at future time steps?"

```
Time now (t):        1 second later:      2 seconds later:
┌──────────┐        ┌──────────┐         ┌──────────┐
│     ■    │        │      ■   │         │       ■  │
│     ■    │  ───→  │      ■   │  ───→   │       ■  │
│  ego     │        │  ego     │         │  ego     │
└──────────┘        └──────────┘         └──────────┘
  (car here)        (car moved right)    (car further right)
```

By predicting where other vehicles will be in the future, the planner can proactively avoid collisions rather than just reacting to the current state.

---

## How It Works Step by Step

Here is the complete forward pass of ST-P3, in plain language:

**Step 1: Capture multi-view images over time**
- 6 cameras capture images at 4 consecutive timesteps (about 2 seconds of history)
- Input shape: (batch, 4 timesteps, 6 cameras, 3 channels, H, W)

**Step 2: Extract image features**
- Each of the 24 images (4 timesteps x 6 cameras) is passed through a CNN backbone
- The backbone extracts visual features (edges, textures, object parts)
- Output: feature maps for each camera at each timestep

**Step 3: Lift to BEV (for each timestep)**
- LSS converts the 6 camera feature maps at each timestep into a single BEV feature map
- Now we have 4 BEV maps: one for t-3, t-2, t-1, and t

**Step 4: Temporal aggregation**
- The ConvGRU processes the 4 BEV maps sequentially
- Starting from an empty hidden state, it accumulates information frame by frame
- The final hidden state encodes the full spatio-temporal understanding of the scene

**Step 5: Spatial BEV encoding**
- Additional convolutional layers refine the aggregated BEV features
- This adds spatial context (understanding how nearby BEV cells relate to each other)

**Step 6: Multi-task prediction (all in parallel)**
- **BEV Segmentation Head:** Predicts road/vehicle/lane/background for each BEV cell
- **Future Occupancy Head:** Predicts which cells will be occupied at 5 future timesteps
- **Planning Head:** A GRU decoder autoregressively generates 6 waypoints for the ego vehicle

**Step 7: Planning output**
- The 6 waypoints define where the ego car should be at 0.5s, 1.0s, 1.5s, 2.0s, 2.5s, 3.0s
- Each waypoint is an (x, y) offset from the current position
- A downstream controller converts waypoints into steering and acceleration commands

---

## Multi-Task Learning

ST-P3 trains three tasks simultaneously rather than training them separately. Why?

### Why Predict Segmentation + Occupancy + Planning Together?

**1. Shared representations are richer**
- The BEV features must be good enough for segmentation (understanding the current scene), occupancy prediction (predicting the future), AND planning (deciding actions)
- This forces the network to learn features that are maximally informative

**2. Auxiliary tasks act as regularization**
- Segmentation loss ensures the BEV features actually "understand" the scene
- Without it, the model might learn shortcut features that only work for planning on the training set but do not generalize

**3. Gradient flow from multiple supervisory signals**
```
Planning Loss  ──────────────────────────┐
                                         │
Future Occupancy Loss  ──────────────┐   │
                                     │   │
BEV Segmentation Loss  ─────────┐   │   │
                                │   │   │
                                ▼   ▼   ▼
                          [Shared BEV Encoder]
                                │
                          [LSS + Backbone]
```
- Multiple losses guide the backbone to extract better features
- Planning loss alone is sparse (just 6 waypoints) -- not enough signal to train a large network

**4. Information flow between tasks**
- The occupancy prediction head learns "where will things be"
- The same features inform the planner "where should I NOT go"
- Segmentation features tell the planner "where CAN I go"

### The Loss Function

```
L_total = w_seg * L_segmentation + w_occ * L_occupancy + w_plan * L_planning

Where:
  L_segmentation = Cross-entropy loss per BEV cell
  L_occupancy    = Binary cross-entropy for future occupied/free
  L_planning     = L2 distance between predicted and expert waypoints
```

---

## Our Implementation

This directory contains a **simplified educational implementation** of ST-P3 (10.7M parameters) that captures the core architectural ideas while being easy to understand and run on modest hardware.

### What is simplified:

| Component | Original ST-P3 | Our Implementation |
|-----------|----------------|-------------------|
| Backbone | EfficientNet-B4 (pretrained) | Simple 4-layer CNN |
| LSS | Full depth prediction + frustum + pillar pooling | Learned direct projection to BEV |
| BEV resolution | 200x200 (0.5m/pixel) | Configurable (default 200x200 or 100x100) |
| Temporal model | ConvGRU/ConvLSTM | ConvGRU (faithful to paper) |
| Planning | Dual-pathway (semantic + geometric) | Single-pathway GRU decoder |
| Training | Multi-GPU, nuScenes pipeline | Demo-ready single-GPU |

### What is faithfully preserved:

- The overall architecture flow (cameras -> backbone -> LSS -> BEV -> temporal GRU -> multi-task heads)
- ConvGRU cell implementation (reset gate, update gate, candidate, blend)
- Multi-task output structure (BEV segmentation + future occupancy + waypoint trajectory)
- GRU-based autoregressive planning (decode one waypoint at a time)
- Temporal processing (sequential BEV frames fed through recurrence)

---

## Running the Code

### Prerequisites

```bash
pip install torch  # PyTorch (CPU or GPU)
```

### Run the Demo

```bash
cd two_step_e2e/ST-P3
python model.py
```

### Expected Output

```
ST-P3: Spatial Temporal Feature Learning Demo
==================================================
Parameters: 10,731,XXX

Input: torch.Size([2, 4, 6, 3, 128, 256]) (B, T, cameras, C, H, W)

Outputs:
  BEV segmentation: torch.Size([2, 4, 100, 100])
  Future occupancy: torch.Size([2, 5, 100, 100])
  Planned trajectory: torch.Size([2, 6, 2])
```

### Understanding the Shapes

| Tensor | Shape | Meaning |
|--------|-------|---------|
| Input images | (B, 4, 6, 3, 128, 256) | Batch, 4 timesteps, 6 cameras, RGB, height, width |
| BEV segmentation | (B, 4, 100, 100) | 4 classes (road/vehicle/lane/bg) over 100x100 BEV grid |
| Future occupancy | (B, 5, 100, 100) | Binary occupancy at 5 future timesteps over BEV grid |
| Trajectory | (B, 6, 2) | 6 waypoints, each with (x, y) position offset |

---

## Results

### Key Results from the Paper (nuScenes)

| Metric | ST-P3 | NMP (CVPR 2020) | Improvement |
|--------|:-----:|:---------------:|:-----------:|
| Planning L2 @ 1s | 0.62m | 0.73m | 15% better |
| Planning L2 @ 2s | 1.27m | 1.46m | 13% better |
| Planning L2 @ 3s | 2.13m | 2.31m | 8% better |
| Collision Rate @ 3s | 1.27% | 1.92% | 34% fewer collisions |

### Ablation: Why Temporal Matters

The paper shows the impact of removing temporal aggregation:

| Model Variant | L2 @ 3s | Collision Rate |
|---------------|:-------:|:--------------:|
| ST-P3 (full, with temporal) | 2.13m | 1.27% |
| Without temporal (single frame) | 2.51m | 1.87% |
| Without prediction task | 2.34m | 1.65% |

Temporal aggregation alone reduces planning error by ~15% and collision rate by ~32%.

---

## Historical Significance

### Why ST-P3 Matters in the E2E Autonomous Driving Timeline

```
Timeline of Two-Step E2E Models:
─────────────────────────────────────────────────────────────
2020 │  NMP (Zeng et al.) - early neural motion planner
     │
2022 │  ST-P3 (ECCV) ← YOU ARE HERE
     │    First to explicitly use spatial-temporal BEV features
     │    for joint perception-prediction-planning
     │
2023 │  UniAD (CVPR Best Paper) - full-stack unified model
     │  VAD (ICCV) - vectorized representation
     │
2024 │  Many follow-ups building on these foundations
─────────────────────────────────────────────────────────────
```

**ST-P3's contributions to the field:**

1. **Established the spatial-temporal BEV paradigm.** Before ST-P3, most E2E models either ignored temporal information or handled it simplistically. ST-P3 showed that explicit temporal aggregation in BEV space dramatically improves planning. Nearly all subsequent models (UniAD, VAD, etc.) use some form of temporal BEV aggregation.

2. **Demonstrated the P3 framework (Perception-Prediction-Planning).** ST-P3 formalized the idea that an E2E driving model should explicitly address all three tasks. This P3 structure became the template for later models.

3. **Showed multi-task learning benefits planning.** The paper provided clear ablation evidence that training segmentation and occupancy prediction alongside planning makes the planner better -- a finding that influenced how all subsequent models are trained.

4. **Proved BEV as the unifying representation.** By showing that BEV features can simultaneously serve perception, prediction, and planning, ST-P3 helped establish BEV as the dominant intermediate representation in autonomous driving (which it remains today).

5. **Came from OpenDriveLab.** The same lab later produced UniAD (CVPR 2023 Best Paper), making ST-P3 an important stepping stone in the most influential line of E2E driving research.

### Limitations (addressed by later work)

| Limitation | How later models addressed it |
|-----------|-------------------------------|
| No explicit object detection | UniAD adds TrackFormer for detection + tracking |
| ConvGRU has limited memory | Transformer-based temporal attention (BEVFormer) |
| Fixed temporal window (4 frames) | Adaptive attention over longer histories |
| No agent interaction modeling | MotionFormer in UniAD models agent-agent interactions |
| CNN backbone | Later models use Vision Transformers for richer features |

---

## References

1. **ST-P3 Paper:** Hu, S., Chen, L., Wu, P., Li, H., Yan, J., & Tao, D. (2022). "ST-P3: End-to-end Vision-based Autonomous Driving via Spatial Temporal Feature Learning." ECCV 2022. [arXiv:2207.07601](https://arxiv.org/abs/2207.07601)

2. **Lift-Splat-Shoot (LSS):** Philion, J., & Fidler, S. (2020). "Lift, Splat, Shoot: Encoding Images From Arbitrary Camera Rigs by Implicitly Unprojecting to 3D." ECCV 2020. [arXiv:2008.05711](https://arxiv.org/abs/2008.05711)

3. **ConvGRU:** Ballas, N., et al. (2016). "Delving Deeper into Convolutional Networks for Learning Video Representations." ICLR 2016.

4. **nuScenes Dataset:** Caesar, H., et al. (2020). "nuScenes: A multimodal dataset for autonomous driving." CVPR 2020.

5. **NMP (Neural Motion Planner):** Zeng, W., et al. (2019). "End-to-end Interpretable Neural Motion Planner." CVPR 2019.

6. **UniAD (successor):** Hu, Y., et al. (2023). "Planning-oriented Autonomous Driving." CVPR 2023 (Best Paper). [arXiv:2212.10156](https://arxiv.org/abs/2212.10156)

---

## Files

```
ST-P3/
├── README.md      # This file (beginner-friendly guide)
└── model.py       # Simplified ST-P3 implementation (10.7M params)
```
