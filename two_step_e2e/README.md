# Two-Step End-to-End Models for Autonomous Driving

> A beginner-friendly guide to the two-step E2E paradigm — models that connect perception and planning networks directly, enabling gradient flow while preserving interpretability.

---

## Table of Contents

- [What is Two-Step E2E?](#what-is-two-step-e2e)
- [How It Compares to Other Approaches](#how-it-compares-to-other-approaches)
- [Advantages and Disadvantages](#advantages-and-disadvantages)
- [Models in This Category](#models-in-this-category)
- [Comparison Table](#comparison-table)
- [Key Concepts Shared Across All Models](#key-concepts-shared-across-all-models)
- [Historical Context](#historical-context)
- [Getting Started](#getting-started)
- [References](#references)

---

## What is Two-Step E2E?

Two-step E2E is an approach to autonomous driving where **two distinct neural networks** (perception and planning) are connected **directly** through learned feature representations, with no hand-crafted post-processing in between. Both networks are trained **jointly** — meaning the planning loss (how good the driving decision is) flows backward through the perception network, teaching it to produce features that are useful for planning, not just for detection.

### The Key Insight

In a traditional pipeline, the perception module outputs discrete objects (bounding boxes, tracking IDs) that are post-processed (NMS, thresholding) before being consumed by the planner. This creates an **information bottleneck** — rich continuous features are reduced to a sparse set of discrete outputs.

Two-step E2E keeps the perception output as **learned continuous features** (like BEV embeddings, object queries, or vector representations). The planner consumes these directly.

```
What "Two-Step" means:

  Step 1: Perception Network
           Input:  Camera images (typically 6 surround cameras)
           Output: Learned features (BEV map, object queries, map vectors)

  Step 2: Planning Network
           Input:  Learned features from Step 1 (NOT post-processed boxes!)
           Output: Future ego-vehicle trajectory (sequence of waypoints)

The two steps are connected by a differentiable bridge:
  - Features flow FORWARD (perception -> planning)
  - Gradients flow BACKWARD (planning loss -> perception weights)
```

### Diagram: How Two-Step E2E Works

```
                         TWO-STEP E2E MODEL
  =====================================================================

  6 Camera Images          Learned Feature Space           Planned Trajectory
  (front, rear,            (continuous, rich)              (future waypoints)
   left, right,
   front-left,            +-------------------+
   front-right)           | BEV Features      |
        |                 | Object Queries    |            [wp1]->[wp2]->[wp3]->...
        |                 | Map Vectors       |                    |
        v                 | Motion States     |                    |
  +-------------+         +--------+----------+         +----------v--------+
  | Perception  |                  |                    |    Planning        |
  | Network     |---> features --->|------features----->|    Network         |
  | (backbone + |                  |                    |    (transformer    |
  |  BEV encoder|    NO post-      |                   |     decoder or     |
  |  + decoders)|    processing!   |                    |     GRU head)      |
  +------+------+                  |                    +----------+---------+
         ^                         |                               |
         |                         |                               |
         |     GRADIENT FLOW (backward during training)            |
         +<-------------------<----+-----<-------------------------+
                                                           Planning Loss
                                                        (L2 to expert trajectory)
```

---

## How It Compares to Other Approaches

### ASCII Architecture Comparison

```
  =====================================================================
  TRADITIONAL MODULAR PIPELINE (NOT end-to-end)
  =====================================================================

  Cameras --> [Perception] --> Boxes --> [NMS] --> [Tracking] --> Track IDs
                                                                     |
                                                                     v
  Control <-- [PID] <-- trajectory <-- [Planning] <-- predictions <-- [Prediction]

  Problems:
    * Each arrow (-->) is an information bottleneck
    * No gradient flow between modules (trained separately)
    * Errors accumulate: perception error --> wrong prediction --> bad plan
    * NMS/tracking are hand-crafted, lose uncertainty information

  =====================================================================
  TWO-STEP E2E (this approach)
  =====================================================================

  Cameras --> [Perception Net] ====features====> [Planning Net] --> Trajectory
                    ^                                    |
                    |           GRADIENT FLOW            |
                    +<----------------------------------+

  Key differences from traditional:
    * Feature connection (====) is LEARNED, not hand-crafted
    * Gradients flow backward (planning improves perception)
    * Rich intermediate features preserved (no information loss)
    * Still interpretable! You can visualize the BEV/objects

  =====================================================================
  ONE-STEP E2E (alternative approach)
  =====================================================================

  Cameras --> [====== Single Big Network ======] --> Trajectory

  Key differences from two-step:
    * No explicit intermediate representation
    * Maximum information preservation
    * Less interpretable (internal features are opaque)
    * Network decides its own internal structure
```

### Detailed Comparison Table

| Aspect | Traditional Modular | Two-Step E2E | One-Step E2E |
|--------|:---:|:---:|:---:|
| **Number of networks** | 4+ (detect, track, predict, plan) | 2 (perception + planning) | 1 (single network) |
| **Training** | Each module separately | Joint, end-to-end | Single network, end-to-end |
| **Interface between modules** | Discrete (boxes, IDs) | Continuous (learned features) | No interface (one network) |
| **Gradient flow** | Blocked at each boundary | Through both networks | Naturally through single net |
| **Interpretability** | High (each output visible) | Medium-high (BEV visible) | Low (internal only) |
| **Information loss** | Significant (at each step) | Minimal | None |
| **Error accumulation** | Yes (cascading failures) | Reduced (joint training) | N/A (one step) |
| **Debugging ease** | Easy (check each module) | Medium (check BEV/queries) | Hard (black box) |
| **Industry adoption** | Current standard | Emerging (research → production) | Emerging (CARLA mostly) |
| **Example** | Apollo, Autoware | UniAD, VAD | TransFuser, DriveVLM |

---

## Advantages and Disadvantages

### Advantages

1. **End-to-end gradient flow**: The planning loss directly improves perception. If the planner needs better lane detection to plan turns, the perception network learns to detect lanes better — automatically.

2. **Interpretable intermediates**: Unlike one-step models, you can visualize what the perception network "sees." This is critical for safety validation and debugging in the real world.

3. **Reduced error accumulation**: Since both networks are trained together, the perception network learns to output features that the planner can use effectively, rather than generic detection outputs.

4. **Modularity for development**: Engineers can still work on perception and planning somewhat independently. You can freeze perception and iterate on planning, or vice versa.

5. **Regulatory compliance**: For safety-critical systems, regulators often require explanations. Two-step models can show "I detected this car, this lane, and planned to avoid it" — one-step models cannot.

### Disadvantages

1. **Computational cost**: Running explicit perception (BEV encoding, object detection, map segmentation) is expensive. One-step models that skip this can be faster.

2. **Bottleneck risk**: Despite using continuous features (not discrete boxes), the intermediate representation still constrains what information reaches the planner. If the perception module doesn't encode something, the planner can never use it.

3. **Architecture complexity**: Designing the interface between perception and planning requires careful engineering. Too few features and you lose information; too many and training is unstable.

4. **Training instability**: Joint training of two large networks can be difficult. Balancing perception losses (detection, segmentation) with planning loss requires careful tuning of loss weights.

5. **Not truly optimal**: The explicit perception step imposes a human-designed structure on the representation. A one-step model might learn a better internal representation that humans would never design.

---

## Models in This Category

### UniAD (CVPR 2023 Best Paper)

**Full name:** Planning-Oriented Autonomous Driving — Unified Autonomous Driving

UniAD is the most comprehensive two-step E2E model. It unifies ALL perception tasks (detection, tracking, mapping, motion prediction, occupancy prediction) into a single framework, then feeds all of these into a planning module.

**Key innovation:** Every task is connected — detection helps tracking, tracking helps prediction, prediction helps planning. All tasks share features and all are improved by the planning loss.

**Architecture highlights:**
- BEVFormer backbone creates bird's-eye-view features from 6 cameras
- Transformer decoders for each task (detection, tracking, mapping, prediction)
- Planning module receives query features from ALL previous tasks
- Trained jointly with task-specific losses + planning loss

**Performance:** 1.03m L2 error at 3s, 0.31% collision rate (nuScenes)

```
See: UniAD/README.md and UniAD/model.py for full implementation
```

---

### VAD (ICCV 2023)

**Full name:** Vectorized Autonomous Driving

VAD improves upon UniAD by using **vectorized representations** instead of dense BEV maps. Instead of encoding the entire scene as a dense 2D grid, VAD represents objects and map elements as vectors (sequences of points). This is more memory-efficient and faster.

**Key innovation:** Replace dense BEV features with sparse vector representations for both agents and map elements. The planner operates on vectors, not grids.

**Architecture highlights:**
- Vectorized agent representation (center points + velocity + heading)
- Vectorized map representation (polylines: lane boundaries, crosswalks)
- Transformer decoder for planning that attends to agent/map vectors
- Scene constraint mechanism that regularizes planning based on map vectors

**Performance:** 0.97m L2 error at 3s, 0.25% collision rate (nuScenes) — better than UniAD and faster

```
See: VAD/README.md and VAD/model.py for full implementation
```

---

### ST-P3 (ECCV 2022)

**Full name:** Spatial-Temporal Feature Learning for End-to-End Planning

ST-P3 is the earliest model in this category and is simpler than UniAD/VAD. It focuses on learning good **spatial-temporal BEV features** that capture how the scene changes over time, then uses a GRU (recurrent network) for planning.

**Key innovation:** Explicitly model temporal changes in BEV features using 3D convolutions and temporal attention. The planner is a simple GRU that autoregressively generates waypoints.

**Architecture highlights:**
- Spatial BEV encoder (lift-splat-shoot or BEVFormer)
- Temporal fusion with 3D convolutions (captures motion over time)
- GRU-based planning head (generates waypoints one by one)
- Auxiliary losses for segmentation and depth to improve BEV quality

**Performance:** 2.13m L2 error at 3s, 1.27% collision rate (nuScenes)

```
See: ST-P3/README.md and ST-P3/model.py for full implementation
```

---

## Comparison Table

| Feature | UniAD | VAD | ST-P3 |
|---------|:---:|:---:|:---:|
| **Year / Venue** | 2023 / CVPR (Best Paper) | 2023 / ICCV | 2022 / ECCV |
| **Scene Representation** | Dense BEV + queries | Vectorized (sparse) | Dense BEV |
| **Perception Tasks** | Detect + Track + Map + Predict + Occupancy | Detect + Map + Predict | Segmentation + Depth |
| **Planning Method** | Transformer decoder | Vector-conditioned transformer | GRU (recurrent) |
| **Planning L2 (3s)** | 1.03m | 0.97m | 2.13m |
| **Collision Rate** | 0.31% | 0.25% | 1.27% |
| **Inference Speed** | ~2 FPS | ~5 FPS | ~8 FPS |
| **Model Size** | ~300M params | ~150M params | ~100M params |
| **Key Strength** | Comprehensive (all tasks unified) | Efficient (vectorized) | Simple, easy to understand |
| **Key Weakness** | Slow, complex to train | Less interpretable than UniAD | Weaker performance |
| **Dataset** | nuScenes | nuScenes | nuScenes |
| **Suggested for** | Understanding full-stack E2E | Production-oriented research | Learning the basics |

---

## Key Concepts Shared Across All Models

### 1. Bird's Eye View (BEV) Representation

All three models transform camera images into a top-down (bird's eye view) representation. This is essential because planning happens in 2D ground plane coordinates (where should the car drive?), but cameras capture 3D perspective views.

```
Camera view (what the camera sees):         BEV (what the planner needs):

    /\                                        +--+--+--+--+--+
   /  \  <-- car appears small               |  |  |  |  |  |
  /    \     (far away)                       |  |XX|  |  |  |  <-- car at (3,2)
 / car  \                                    |  |  |  |  |  |
/________\                                    |  |  |  |  |  |
  road                                        |  |  |EG|  |  |  <-- ego at (3,4)
                                              +--+--+--+--+--+
                                                 Top-down view
```

**How it works (simplified):**
1. Extract image features with a backbone (ResNet, Swin Transformer)
2. Use depth estimation or geometry to "lift" 2D features into 3D space
3. Project 3D features down onto the ground plane --> BEV features
4. Popular methods: Lift-Splat-Shoot, BEVFormer, BEVDet

### 2. Transformer Decoders (Query-based detection)

All three models use transformer decoders (from the DETR paradigm) to convert BEV features into structured outputs. The key idea is **learned queries**:

```
How transformer decoders work:

  BEV Features (what the scene looks like from above)
       |
       v
  [Transformer Decoder]
       |
       |--- Query 1 -----> "I found a car at position (5, 3), heading east, speed 10 m/s"
       |--- Query 2 -----> "I found a pedestrian at (2, 7), walking north, speed 1.5 m/s"
       |--- Query 3 -----> "I found a lane boundary from (0,0) to (50,0)"
       |--- ...
       |--- Query N -----> "Nothing here" (no object at this query)

  Each query "asks" the BEV features: "Is there something at my learned position?"
  Through attention, each query gathers information from relevant spatial locations.
```

**Why queries matter for two-step E2E:** The query outputs ARE the learned features that flow into the planning module. They are continuous vectors (not discrete boxes), enabling gradient flow.

### 3. Planning Loss and End-to-End Training

The planning loss is what makes these models truly "end-to-end." Without it, the perception and planning networks would just be two separate models running in sequence.

```
Planning Loss (how we train):

  Expert trajectory (what a human driver did):   [x1,y1] -> [x2,y2] -> [x3,y3] -> ...
  Predicted trajectory (what the model plans):   [x1',y1'] -> [x2',y2'] -> [x3',y3'] -> ...

  L2 Loss = sum of squared distances between predicted and expert waypoints
           = (x1-x1')^2 + (y1-y1')^2 + (x2-x2')^2 + (y2-y2')^2 + ...

  This loss flows backward through:
    Planning Network --> Perception Features --> Perception Network --> Image Backbone

  The entire model is optimized so that the FINAL trajectory matches the expert.
```

**Total loss** = Planning Loss + (weighted sum of auxiliary perception losses)

```
L_total = L_planning + w1*L_detection + w2*L_tracking + w3*L_mapping + w4*L_prediction
```

The auxiliary losses ensure that intermediate representations remain interpretable and don't collapse into unstructured features.

### 4. Temporal Fusion (Multi-frame Input)

All models use multiple frames (typically 2-4 seconds of history) to understand motion:

```
Time t-3  Time t-2  Time t-1  Time t (now)
  |         |         |         |
  v         v         v         v
[BEV_t-3] [BEV_t-2] [BEV_t-1] [BEV_t]
  |         |         |         |
  +---------+---------+---------+---> Temporal Fusion ---> Fused BEV
                                      (captures motion,
                                       speed, acceleration)
```

This is critical because a single frame cannot tell you if a car is stopped, moving, or accelerating.

---

## Historical Context

### The Evolution to Two-Step E2E

```
Timeline:
=========

2015-2019: Traditional Modular Pipelines dominate
           (PointPillars, CenterPoint for detection; rule-based planners)
           Problem: error accumulation, information loss at interfaces

2020-2021: Early attempts at connecting modules
           (PnPNet, NMP connect prediction to planning)
           Problem: still hand-crafted interfaces between some modules

2022:      ST-P3 — First clean demonstration of spatial-temporal BEV + planning
           Shows that joint training works and improves both perception and planning

2023:      UniAD — Unifies ALL tasks into one framework (CVPR Best Paper)
           Proves that more tasks connected = better planning performance
           
2023:      VAD — Shows vectorized representations are more efficient
           Makes two-step E2E practical (fast enough for real-time consideration)

2024+:     Industry adoption begins
           Tesla, Waymo, Huawei explore two-step E2E in production
           Focus shifts to scaling, safety guarantees, and real-time performance
```

### Why Two-Step E2E Emerged

The key driver was the observation that **planning performance plateaued** in traditional pipelines even as perception improved. Better object detection did not translate to better driving because:

1. **Information loss**: Converting rich features to boxes loses uncertainty, occluded object hints, and scene context.
2. **No feedback**: Perception never knew what the planner needed. It was optimized for detection metrics (mAP), not driving quality.
3. **Error cascading**: A missed detection could not be recovered by downstream modules.

Two-step E2E solved these by connecting modules through differentiable features while retaining the structured decomposition that engineers understand.

---

## Getting Started

### Suggested Reading Order

**If you're new to autonomous driving E2E models:**

1. **Start with ST-P3** (`ST-P3/README.md`) — It is the simplest model. You will learn:
   - How BEV features are created from cameras
   - How temporal information is fused
   - How a GRU-based planner generates waypoints
   - The basic training loop with planning loss

2. **Then read UniAD** (`UniAD/README.md`) — The most comprehensive model. You will learn:
   - How multiple perception tasks are unified
   - How query-based detection works
   - How all tasks feed into planning
   - Why CVPR gave it Best Paper

3. **Finally read VAD** (`VAD/README.md`) — The most practical model. You will learn:
   - Why vectorized representations are better than dense BEV
   - How to make two-step E2E efficient
   - Scene constraint mechanisms
   - Trade-offs between complexity and performance

### Running the Code

```bash
# Each model has a standalone demo you can run:
python two_step_e2e/ST-P3/model.py       # Simplest — start here
python two_step_e2e/UniAD/model.py       # Most comprehensive
python two_step_e2e/VAD/model.py         # Most efficient

# Each demo will:
# 1. Create a model with the correct architecture
# 2. Generate synthetic input data (simulating camera features)
# 3. Run a forward pass
# 4. Print the output trajectory and intermediate representations
```

### Prerequisites

```bash
pip install torch numpy
```

No GPU required for running demos (they use small synthetic inputs). Training on real data requires a GPU with 24+ GB memory.

---

## References

### Papers

- **UniAD**: Hu et al., "Planning-oriented Autonomous Driving," CVPR 2023. [arXiv:2212.10156](https://arxiv.org/abs/2212.10156)
- **VAD**: Jiang et al., "VAD: Vectorized Scene Representation for Efficient Autonomous Driving," ICCV 2023. [arXiv:2303.12077](https://arxiv.org/abs/2303.12077)
- **ST-P3**: Hu et al., "ST-P3: End-to-end Vision-based Autonomous Driving via Spatial-Temporal Feature Learning," ECCV 2022. [arXiv:2207.07601](https://arxiv.org/abs/2207.07601)

### Related Foundational Works

- **BEVFormer**: Li et al., "BEVFormer: Learning Bird's-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers," ECCV 2022. [arXiv:2203.17270](https://arxiv.org/abs/2203.17270)
- **DETR**: Carion et al., "End-to-End Object Detection with Transformers," ECCV 2020. [arXiv:2005.12872](https://arxiv.org/abs/2005.12872)
- **Lift-Splat-Shoot**: Philion & Fidler, "Lift, Splat, Shoot: Encoding Images From Arbitrary Camera Rigs," ECCV 2020. [arXiv:2008.05711](https://arxiv.org/abs/2008.05711)

### Official Code Repositories

- UniAD: https://github.com/OpenDriveLab/UniAD
- VAD: https://github.com/hustvl/VAD
- ST-P3: https://github.com/OpenDriveLab/ST-P3
