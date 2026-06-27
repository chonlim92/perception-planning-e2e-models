# One-Step End-to-End Models for Autonomous Driving

> A beginner-friendly guide to the one-step E2E paradigm — models that map directly from raw sensor data to driving decisions in a single network, with no explicit intermediate perception output.

---

## Table of Contents

- [What is One-Step E2E?](#what-is-one-step-e2e)
- [Two Sub-Categories](#two-sub-categories)
- [Sub-Category 1: Traditional Deep Learning](#sub-category-1-traditional-deep-learning)
- [Sub-Category 2: Foundation Models](#sub-category-2-foundation-models-the-new-frontier)
- [Why Foundation Models Are the New Frontier](#why-foundation-models-are-the-new-frontier)
- [Comparison Table: All 6 Models](#comparison-table-all-6-models)
- [Advantages and Disadvantages](#advantages-and-disadvantages)
- [Key Concepts](#key-concepts)
- [Getting Started](#getting-started)
- [References](#references)

---

## What is One-Step E2E?

One-step E2E models take raw sensor data (camera images, sometimes LiDAR) and produce driving decisions (trajectory waypoints or steering/throttle/brake commands) in a **single forward pass** through one unified network. There is NO explicit perception output — no bounding boxes, no BEV segmentation map, no detected lanes that you can inspect.

```
One-Step E2E (conceptual):

  Raw Sensors -----> [ Single Neural Network ] -----> Driving Decision
  (6 cameras,        (internal features are           (trajectory: x,y
   LiDAR point       NOT interpretable —              at future times,
   cloud,            network learns its own           OR steering angle
   GPS/IMU)          representation)                  + throttle + brake)
```

### How is This Different from Two-Step E2E?

In **two-step** E2E, you have two clear sub-networks (perception and planning) with a visible intermediate representation (BEV features, detected objects) that you can inspect and debug.

In **one-step** E2E, the network is a single black box. Internally it may learn perception-like features, but you cannot directly inspect them. The network decides for itself what internal representation is best for driving — not for human-interpretable detection metrics.

```
Two-Step:  Cameras --> [Perception Net] --> BEV/Objects --> [Planning Net] --> Trajectory
                                            ^ visible!

One-Step:  Cameras --> [===================Network===================] --> Trajectory
                       (internal features are learned and opaque)
```

---

## Two Sub-Categories

One-step E2E models are divided into two groups based on their training paradigm and architecture scale:

```
                          ONE-STEP E2E MODELS
                                 |
                +----------------+----------------+
                |                                 |
    Traditional Deep Learning            Foundation Model Approaches
    (CNN + Transformer, ~50M params)     (VLM/World Model/Diffusion, 500M-9B params)
                |                                 |
    +-----+-----+-----+             +------+------+------+
    |     |           |             |      |             |
TransFuser InterFuser TCP       DriveVLM GAIA-1       GenAD
 (2022)    (2022)    (2022)     (2024)   (2023)      (2024)
```

### Key Differences Between the Sub-Categories

| Aspect | Traditional DL | Foundation Models |
|--------|:---:|:---:|
| **Training data** | CARLA simulator (expert demos) | Massive real-world driving video |
| **Model size** | 20-80M parameters | 500M - 9B parameters |
| **Training paradigm** | Supervised (imitation learning) | Pre-train + Fine-tune + RL |
| **Output** | Waypoints or control signals | Waypoints + text explanations |
| **Interpretability** | Low (black box) | Medium (chain-of-thought, video generation) |
| **Generalization** | Limited to training distribution | Strong (broad pre-training) |
| **Real-time capable** | Yes (10-30 FPS) | Not yet (1-5 FPS) |
| **Multi-modal output** | Usually single trajectory | Naturally diverse (many options) |

---

## Sub-Category 1: Traditional Deep Learning

These models use standard deep learning architectures (CNNs, Transformers, GRUs) to directly map sensor inputs to driving outputs. They are trained with **imitation learning** — copying what an expert driver did in the same situation.

### Architecture Pattern

```
  TRADITIONAL ONE-STEP E2E (TransFuser / InterFuser / TCP)
  ================================================================

  Camera Image(s)          LiDAR BEV              Route/Navigation
       |                      |                        |
       v                      v                        |
  +-----------+         +-----------+                  |
  | Image     |         | LiDAR     |                  |
  | Backbone  |         | Backbone  |                  |
  | (ResNet/  |         | (PointNet/|                  |
  |  ViT)     |         |  ResNet)  |                  |
  +-----+-----+         +-----+-----+                  |
        |                      |                        |
        v                      v                        |
  +--------------------------------------------+        |
  |         FUSION MODULE                      |        |
  |  (transformer attention / concatenation /  |<-------+
  |   cross-attention between image & LiDAR)   |
  +---------------------+----------------------+
                        |
                        v
  +--------------------------------------------+
  |           PREDICTION HEADS                 |
  |                                            |
  |  +-----------+  +-----------+  +--------+  |
  |  | Waypoint  |  | Control   |  | Safety |  |
  |  | Head      |  | Head      |  | Head   |  |
  |  | (x,y at   |  | (steer,   |  | (stop, |  |
  |  |  T steps) |  |  gas,     |  |  slow) |  |
  |  |           |  |  brake)   |  |        |  |
  |  +-----------+  +-----------+  +--------+  |
  +--------------------------------------------+
```

### Model 1: TransFuser (CVPR 2022 / PAMI 2023)

**Full name:** Multi-Modal Fusion Transformer for End-to-End Driving

TransFuser was one of the first models to show that **transformers can fuse multi-modal inputs** (camera + LiDAR) effectively for driving. It uses self-attention between image and LiDAR features at multiple spatial scales.

**Key innovation:** Multi-scale transformer fusion. Instead of fusing at a single resolution, TransFuser applies transformer attention at 4 different scales (from low-resolution global context to high-resolution local detail).

**How it works:**
1. Process camera images through a ResNet backbone at 4 scales
2. Process LiDAR BEV through a separate ResNet at 4 scales
3. At each scale, apply transformer self-attention between image and LiDAR tokens
4. Predict waypoints with a GRU decoder
5. Convert waypoints to control signals with a PID controller

**CARLA Driving Score:** 54.52

---

### Model 2: InterFuser (CoRL 2022)

**Full name:** Safety-Enhanced Autonomous Driving Using Interpretable Sensor Fusion Transformer

InterFuser improves upon TransFuser by adding **safety-oriented auxiliary outputs** — density maps that predict where objects are likely to be. This gives the model an implicit understanding of risk.

**Key innovation:** Multi-view multi-sensor attention + safety density maps. The model outputs not just waypoints but also a spatial density map showing where obstacles are, giving some interpretability to an otherwise black-box model.

**How it works:**
1. Process multiple camera views (front, left, right) + LiDAR separately
2. A transformer encoder fuses all views through cross-attention
3. Produce safety density maps (auxiliary output showing obstacle locations)
4. Predict waypoints conditioned on the safety-aware features
5. Use density maps for collision-aware waypoint refinement

**CARLA Driving Score:** 68.31

---

### Model 3: TCP (NeurIPS 2022)

**Full name:** Trajectory-guided Control Prediction

TCP is the best-performing traditional one-step model. Its key insight is that **trajectory planning and direct control prediction are complementary** — trajectory gives long-horizon planning, while direct control gives precise short-horizon execution.

**Key innovation:** Dual-branch architecture where trajectory prediction guides control prediction through attention. The trajectory branch provides a "plan" that the control branch uses to compute precise steering/throttle/brake.

**How it works:**
1. Camera-only input (no LiDAR needed!)
2. Branch A: Predict trajectory waypoints (long-horizon plan)
3. Branch B: Predict control signals (steer, throttle, brake)
4. Guidance: Trajectory features attend to control features via cross-attention
5. Final output blends both branches for robust driving

**CARLA Driving Score:** 75.14

---

## Sub-Category 2: Foundation Models (The New Frontier)

Foundation models apply the same paradigm that created GPT/ChatGPT to autonomous driving:

```
  FOUNDATION MODEL TRAINING PARADIGM
  ================================================================

  GPT/ChatGPT                          Autonomous Driving
  ===============                      ====================

  Stage 1: Pre-train on               Stage 1: Pre-train on
           internet text                        millions of driving hours
           (learn language)                     (learn visual world model)

  Stage 2: Fine-tune on               Stage 2: Fine-tune on
           Q&A dialogues                        driving decisions
           (learn to follow                     (learn to drive)
            instructions)

  Stage 3: RLHF                        Stage 3: RL from driving rewards
           (human feedback                      (safety, comfort, efficiency
            improves quality)                    improve driving quality)

  Result:  ChatGPT reasons             Result:  Model drives with
           about language                        reasoning about the scene
```

### Architecture Pattern for Foundation Models

```
  FOUNDATION MODEL ONE-STEP E2E (DriveVLM / GAIA-1 / GenAD)
  ================================================================

  Camera Images     Text Command          Driving History
  (6 surround)     ("turn left at        (past 3 seconds
       |            next light")          of ego motion)
       |                 |                     |
       v                 v                     v
  +----------------------------------------------------------+
  |              LARGE PRE-TRAINED BACKBONE                   |
  |  (Vision-Language Model / World Model / Diffusion Model)  |
  |                                                          |
  |  [Billions of parameters, pre-trained on massive data]   |
  |                                                          |
  |  Internal reasoning:                                     |
  |    "I see a red light ahead..."                         |
  |    "There's a pedestrian crossing..."                    |
  |    "If I turn left now, I'll enter oncoming traffic..."  |
  +---------------------------+------------------------------+
                              |
                              v
  +----------------------------------------------------------+
  |                    OUTPUT GENERATION                      |
  |                                                          |
  |  DriveVLM:  Trajectory + text explanation                |
  |  GAIA-1:    Future video prediction + trajectory         |
  |  GenAD:     Multiple diverse trajectory proposals        |
  +----------------------------------------------------------+
```

### Model 4: DriveVLM (2024)

**Full name:** Drive with Vision-Language Model

DriveVLM uses a **Vision-Language Model** (like GPT-4V / LLaVA) that can both see images and reason about them in natural language. It drives by first describing the scene in words, then reasoning about what to do, then outputting a trajectory.

**Key innovation:** Chain-of-thought reasoning for driving. The model doesn't just output a trajectory — it first generates a textual explanation of its reasoning. This provides unprecedented interpretability for a one-step model.

**How it works:**
1. Input: surround camera images + navigation command
2. Scene description: "There is a stopped bus on the right lane, pedestrians are crossing..."
3. Reasoning: "I should slow down and wait for pedestrians, then merge left around the bus..."
4. Action: Output trajectory waypoints based on the reasoning
5. The chain-of-thought reasoning is generated autoregressively (token by token, like ChatGPT)

**Parameters:** ~7B | **Speed:** 1-2 FPS (too slow for real-time deployment without optimization)

---

### Model 5: GAIA-1 (Wayve, 2023)

**Full name:** Generative AI for Autonomy — A Generative World Model

GAIA-1 is a **world model** — it learns to predict what the world will look like in the future given different actions. It plans by mentally simulating "what happens if I do X?" and choosing the action whose simulated future looks best.

**Key innovation:** Generative world modeling for planning. Instead of directly predicting the best action, GAIA-1 imagines multiple futures (as video frames) and evaluates which future is safest/best.

**How it works:**
1. Input: past camera frames + past ego actions + proposed future action
2. The model generates a video of the predicted future (what will happen)
3. Planning: try multiple actions, generate future video for each, score the futures
4. Select the action whose predicted future is safest and most comfortable
5. Architecture: video VQ-VAE (tokenizes video) + transformer (predicts next token)

**Parameters:** ~9B | **Speed:** ~5 FPS

```
GAIA-1 Planning Process:

  Current scene: [car ahead slowing down]

  Action A: "accelerate"  --> Predicted future: [CRASH into car ahead] --> Score: BAD
  Action B: "brake"       --> Predicted future: [safe stop behind car] --> Score: GOOD
  Action C: "lane change" --> Predicted future: [merge left safely]    --> Score: GOOD

  Selected: Action C (best score considering progress + safety)
```

---

### Model 6: GenAD (2024)

**Full name:** Generative End-to-End Autonomous Driving

GenAD uses a **diffusion model** (the same type of model behind image generators like DALL-E and Stable Diffusion) to generate driving trajectories. Instead of predicting a single "best" trajectory, it generates many diverse trajectory proposals and then scores them to select the best one.

**Key innovation:** Diffusion-based multi-modal trajectory generation. Naturally handles the multi-modal planning problem (multiple correct answers) by generating a diverse set of proposals.

**How it works:**
1. Encode the driving scene (cameras + map + agents)
2. Use a diffusion model to generate K candidate trajectories (K = 64 typically)
3. Each trajectory is different — some go left, some go right, some slow down
4. Score all trajectories using a learned scorer (see `planner_scorer/` module)
5. Select the best-scoring trajectory for execution

**Parameters:** ~500M | **Speed:** Near real-time (depends on diffusion steps, typically 5-10 steps)

```
GenAD Diffusion Process:

  Step 1: Start with random noise     [~~~~~random~~~~~]
  Step 2: Denoise conditioned on scene [~~somewhat shaped~~]
  Step 3: Continue denoising           [looks like a path]
  Step 4: Final trajectory             [smooth, drivable trajectory]

  Repeat 64 times --> 64 diverse proposals --> Score --> Select best
```

---

## Why Foundation Models Are the New Frontier

### The Limitations of Traditional One-Step Models

Traditional models (TransFuser, InterFuser, TCP) are trained by **imitation learning** — they copy what an expert did. This has fundamental limits:

1. **Distribution shift**: The model only knows situations it was trained on. Novel scenarios (construction zones, unusual obstacles) cause failure.
2. **Single-answer problem**: They predict ONE trajectory. But driving often has multiple valid options. Averaging between "go left" and "go right" produces "go straight into the obstacle."
3. **No reasoning**: They cannot explain their decisions. If something goes wrong, there is no way to understand why.
4. **Simulator gap**: Most are trained in CARLA simulator, which does not capture the full complexity of the real world.

### How Foundation Models Solve These

| Problem | Traditional DL Solution | Foundation Model Solution |
|---------|:---:|:---:|
| Novel scenarios | Fail (out of distribution) | Generalize (broad pre-training) |
| Multiple valid actions | Predict one (or average) | Generate diverse options (diffusion/VLM) |
| No explanation | Black box | Chain-of-thought reasoning (VLM) |
| Simulator gap | Train in CARLA | Pre-train on real-world video |
| Performance ceiling | Bounded by expert | Go beyond expert (RL fine-tuning) |

### The Three Foundation Model Approaches Compared

```
  Which Foundation Model approach to use?

  +-----------+     "I want my car to EXPLAIN its decisions"
  | DriveVLM  |     --> Use a Vision-Language Model
  +-----------+     Good for: debugging, safety validation, user trust
                    Bad for:  real-time (slow inference)

  +-----------+     "I want my car to IMAGINE what happens next"
  | GAIA-1    |     --> Use a World Model
  +-----------+     Good for: long-horizon planning, causal understanding
                    Bad for:  real-time (slow generation), hallucination

  +-----------+     "I want my car to consider MANY options"
  | GenAD     |     --> Use a Generative/Diffusion Model
  +-----------+     Good for: multi-modal planning, trajectory diversity
                    Bad for:  needs a separate scorer, generation quality
```

---

## Comparison Table: All 6 Models

| Feature | TransFuser | InterFuser | TCP | DriveVLM | GAIA-1 | GenAD |
|---------|:---:|:---:|:---:|:---:|:---:|:---:|
| **Year** | 2022 | 2022 | 2022 | 2024 | 2023 | 2024 |
| **Venue** | CVPR/PAMI | CoRL | NeurIPS | arXiv | arXiv | arXiv |
| **Sub-category** | Traditional DL | Traditional DL | Traditional DL | Foundation | Foundation | Foundation |
| **Input sensors** | Camera + LiDAR | Camera + LiDAR | Camera only | Camera + Text | Camera + Action | Camera + Map |
| **Parameters** | ~50M | ~60M | ~30M | ~7B | ~9B | ~500M |
| **Output type** | Waypoints + Control | Waypoints + Density | Waypoints + Control | Waypoints + Text | Future video + Traj | K trajectories |
| **Training data** | CARLA expert | CARLA expert | CARLA expert | Real driving video | Real driving video | Real driving video |
| **Training paradigm** | Imitation learning | Imitation learning | Imitation learning | Pretrain + Finetune | Pretrain + Finetune | Pretrain + Diffusion |
| **Driving Score (CARLA)** | 54.52 | 68.31 | 75.14 | N/A (real-world) | N/A (real-world) | N/A (real-world) |
| **Real-time capable** | Yes (15+ FPS) | Yes (10+ FPS) | Yes (20+ FPS) | No (1-2 FPS) | No (~5 FPS) | Near (8-15 FPS) |
| **Interpretability** | Low | Medium (density maps) | Low | High (text explanation) | Medium (video prediction) | Low (needs scorer) |
| **Multi-modal output** | No (single traj) | No (single traj) | No (single traj) | Partially (reasoning) | Yes (imagined futures) | Yes (K proposals) |
| **Generalization** | Low (CARLA only) | Low (CARLA only) | Low (CARLA only) | High (pre-trained) | High (pre-trained) | High (pre-trained) |
| **Key strength** | Simple, fast | Safety-aware | Best CARLA score | Explainable driving | Causal reasoning | Diverse trajectories |
| **Key weakness** | Limited scenarios | Complex architecture | Camera-only limits | Very slow | Hallucination risk | Needs good scorer |

---

## Advantages and Disadvantages

### Advantages of One-Step E2E (General)

1. **Maximum information preservation**: No intermediate bottleneck. The network has access to all raw sensor data throughout the entire computation.

2. **Optimal internal representations**: The network learns whatever internal structure is best for driving — not constrained by human-designed perception outputs.

3. **Simpler architecture**: No need to design interfaces between perception and planning. The network figures it out.

4. **Potentially faster**: No explicit perception stage means fewer sequential computations (especially for traditional DL models).

### Disadvantages of One-Step E2E (General)

1. **Poor interpretability**: You cannot see what the model "perceives." If it makes a bad decision, you cannot tell if it failed to detect an obstacle or failed to plan around it.

2. **Harder to validate for safety**: Regulators and safety engineers cannot verify perception is working correctly because there is no perception output to check.

3. **Data hungry**: Without structured intermediate tasks, the network needs more data to learn good internal representations.

4. **Debugging difficulty**: When the car does something wrong, you cannot isolate which internal component failed.

### Advantages Specific to Foundation Models

1. **Generalization**: Pre-training on massive real-world data gives understanding of rare scenarios that simulator training cannot provide.

2. **Beyond imitation**: RL fine-tuning allows the model to discover driving strategies better than the human expert it was trained on.

3. **Multi-modal planning**: Generative models naturally produce diverse options, solving the "averaging" problem of traditional approaches.

4. **Explainability (VLM)**: Chain-of-thought reasoning provides natural language explanations for driving decisions.

### Disadvantages Specific to Foundation Models

1. **Computational cost**: 7-9 billion parameters require significant hardware. Real-time inference is not yet practical for most.

2. **Hallucination**: Like ChatGPT making up facts, world models can imagine physically impossible futures. Safety requires external verification.

3. **Training cost**: Pre-training requires thousands of GPU-hours and petabytes of data.

4. **Latency**: Even if fast enough on average, worst-case latency for autoregressive generation can be unacceptable for safety-critical driving.

---

## Key Concepts

### 1. Imitation Learning (Behavioral Cloning)

The training paradigm for traditional one-step models. The network learns to copy what an expert driver did:

```
Training data:
  Situation 1: [camera images at time t] --> Expert action: [steer=0.1, throttle=0.6]
  Situation 2: [camera images at time t] --> Expert action: [steer=-0.3, brake=0.8]
  ...

Loss function:
  L = ||predicted_action - expert_action||^2

Problem (compounding error):
  At test time, the model makes a small error -> enters a state it never saw in training
  -> makes a bigger error -> enters an even stranger state -> crashes
```

### 2. Sensor Fusion (Image + LiDAR)

TransFuser and InterFuser use both cameras and LiDAR. The key challenge is fusing these very different data types:

```
Camera: Dense 2D color image (H x W x 3)
        - Rich appearance (color, texture)
        - No direct depth information

LiDAR:  Sparse 3D point cloud (N x 3)
        - Precise depth/distance
        - No color/texture
        - Sparse (many empty areas)

Fusion strategies:
  1. Early fusion:  Concatenate features at input level
  2. Late fusion:   Process separately, combine at output
  3. Attention fusion: Cross-attention between modalities (TransFuser approach)
```

### 3. Multi-Modal Planning (The Averaging Problem)

A critical challenge in one-step models (solved by foundation models):

```
Scenario: Obstacle ahead. Two equally valid options: go left OR go right.

  Expert dataset contains:
    50% of experts went LEFT:  trajectory = [(-1,1), (-2,2), (-3,3)]
    50% of experts went RIGHT: trajectory = [(1,1), (2,2), (3,3)]

  A model trained with L2 loss will AVERAGE these:
    predicted = [(0,1), (0,2), (0,3)]  <-- goes STRAIGHT into the obstacle!

  Solution (GenAD): Generate 64 diverse trajectories, THEN score and select.
  Solution (GAIA-1): Imagine both futures, THEN pick the better one.
  Solution (DriveVLM): Reason in language about which option is better.
```

### 4. PID Controller (Waypoints to Control)

Most models output waypoints (x, y positions at future times). These must be converted to actual steering/throttle/brake:

```
PID Controller:
  Input:  Waypoints [wp1, wp2, wp3, ...]  (where to go)
  Output: Steering angle, throttle, brake  (how to get there)

  Steering = Kp * lateral_error + Kd * d(lateral_error)/dt
  Throttle = Kp * (desired_speed - current_speed)

  (PID = Proportional-Integral-Derivative controller)
```

### 5. VQ-VAE (Vector Quantized Variational Autoencoder)

Used by GAIA-1 to convert video frames into discrete tokens that a transformer can process:

```
VQ-VAE Process:
  Video frame (256x256 RGB) --> Encoder --> Continuous features --> Codebook lookup
                                                                         |
                                              Match to nearest code      v
                                              in a learned codebook      [token_42]
                                                                         
  Token sequence: [42, 17, 88, 3, 55, ...] (like words in a sentence)
  
  A transformer then predicts: given past tokens, what comes next?
  (Same as GPT predicting the next word, but for video frames)
```

### 6. Diffusion Models (GenAD)

Used by GenAD to generate diverse trajectory proposals:

```
Diffusion training:
  Forward process: Add noise to expert trajectory step by step
    Expert traj --> slightly noisy --> more noisy --> ... --> pure random noise

  Reverse process (learned): Remove noise step by step
    Pure noise --> less noisy --> ... --> clean trajectory (generated)

At inference:
  1. Start with random noise (shape = trajectory)
  2. Condition on the scene (camera features, map, agents)
  3. Iteratively denoise --> produces a plausible trajectory
  4. Repeat with different random noise --> different trajectory each time
  5. Now you have K diverse, plausible trajectories to choose from!
```

---

## Getting Started

### Suggested Reading Order

**For understanding the basics (traditional DL):**

1. **TransFuser** (`TransFuser/README.md`) — Start here. Learn how multi-modal fusion works and how transformers bridge camera and LiDAR. The simplest architecture to understand.

2. **InterFuser** (`InterFuser/README.md`) — Next, see how safety considerations are added to the architecture. Introduces density maps as a safety mechanism.

3. **TCP** (`TCP/README.md`) — Finally, understand the dual-branch (trajectory + control) design that achieves the best performance. Shows why trajectory guidance helps control prediction.

**For understanding the frontier (foundation models):**

4. **DriveVLM** (`DriveVLM/README.md`) — Start with the most intuitive foundation model. If you understand ChatGPT, you can understand DriveVLM. The chain-of-thought reasoning is accessible.

5. **GenAD** (`GenAD/README.md`) — Next, understand diffusion-based trajectory generation. This is the most practical foundation model approach (smallest, fastest). Connects to the `planner_scorer/` module for trajectory selection.

6. **GAIA-1** (`GAIA-1/README.md`) — Finally, explore world models. This is the most ambitious approach — a model that learns the physics of the world by predicting video futures.

### Running the Code

```bash
# Traditional DL models (simpler, faster)
python one_step_e2e/TransFuser/model.py    # Multi-modal fusion demo
python one_step_e2e/InterFuser/model.py    # Safety-enhanced fusion demo
python one_step_e2e/TCP/model.py           # Dual-branch control demo

# Foundation Model approaches (more complex)
python one_step_e2e/DriveVLM/model.py      # Vision-Language driving demo
python one_step_e2e/GAIA-1/model.py        # World model / future prediction demo
python one_step_e2e/GenAD/model.py         # Diffusion trajectory generation demo
```

Each demo will:
1. Create the model architecture
2. Generate synthetic input data (simulating sensor inputs)
3. Run a forward pass
4. Print/visualize the output (trajectory, control signals, or generated text)

### Prerequisites

```bash
pip install torch numpy scipy matplotlib
```

No GPU required for running demos (synthetic small-scale inputs). Training foundation models on real data requires multiple GPUs with 80+ GB memory each.

### What to Look For in the Code

When reading the model files, pay attention to:

- **Forward method**: This shows the full data flow from input to output
- **Fusion mechanism**: How different sensor modalities are combined
- **Output heads**: How the final driving decision is produced
- **Loss computation**: What objective the model is optimized for

---

## References

### Traditional DL Papers

- **TransFuser**: Chitta et al., "TransFuser: Imitation with Transformer-Based Sensor Fusion for Autonomous Driving," PAMI 2023. [arXiv:2205.15997](https://arxiv.org/abs/2205.15997)
- **InterFuser**: Shao et al., "Safety-Enhanced Autonomous Driving Using Interpretable Sensor Fusion Transformer," CoRL 2022. [arXiv:2207.14024](https://arxiv.org/abs/2207.14024)
- **TCP**: Wu et al., "Trajectory-guided Control Prediction for End-to-end Autonomous Driving: A Simple yet Strong Baseline," NeurIPS 2022. [arXiv:2206.08129](https://arxiv.org/abs/2206.08129)

### Foundation Model Papers

- **DriveVLM**: Tian et al., "DriveVLM: The Convergence of Autonomous Driving and Large Vision-Language Models," 2024. [arXiv:2402.12289](https://arxiv.org/abs/2402.12289)
- **GAIA-1**: Hu et al., "GAIA-1: A Generative World Model for Autonomous Driving," Wayve 2023. [arXiv:2309.17080](https://arxiv.org/abs/2309.17080)
- **GenAD**: Zheng et al., "GenAD: Generative End-to-End Autonomous Driving," 2024. [arXiv:2402.11502](https://arxiv.org/abs/2402.11502)

### Official Code Repositories

- TransFuser: https://github.com/autonomousvision/transfuser
- InterFuser: https://github.com/opendilab/InterFuser
- TCP: https://github.com/OpenDriveLab/TCP

### Background Reading

- **CARLA Simulator**: https://carla.org — The simulator used by traditional DL models
- **Diffusion Models**: Ho et al., "Denoising Diffusion Probabilistic Models," NeurIPS 2020
- **VQ-VAE**: van den Oord et al., "Neural Discrete Representation Learning," NeurIPS 2017
- **LLaVA**: Liu et al., "Visual Instruction Tuning," NeurIPS 2023
