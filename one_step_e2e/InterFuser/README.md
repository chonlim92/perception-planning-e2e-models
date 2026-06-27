# InterFuser

**Safety-Enhanced Autonomous Driving Using Interpretable Sensor Fusion Transformer**

A one-step end-to-end driving model that takes raw sensor inputs and directly outputs driving actions, while also producing human-interpretable safety maps for debugging and verification.

---

## What is InterFuser?

InterFuser (Interpretable Sensor Fusion Transformer) is a one-step end-to-end autonomous driving model published at CoRL 2022. "One-step" means the model takes raw sensor data (cameras + LiDAR) as input and directly produces driving actions (waypoints) as output -- no separate perception or planning modules in between.

What makes InterFuser special is its emphasis on **safety** and **interpretability**. Most end-to-end models are black boxes: you feed in sensor data, driving commands come out, and you have no idea why the model made a particular decision. InterFuser solves this by producing interpretable intermediate outputs -- density maps, waypoint heatmaps, and a safety score -- that let engineers understand what the model "sees" and whether it considers the current situation safe.

**Key facts:**
- **Paper:** "Safety-Enhanced Autonomous Driving Using Interpretable Sensor Fusion Transformer"
- **Venue:** CoRL 2022
- **arXiv:** [2207.14024](https://arxiv.org/abs/2207.14024)
- **Official code:** [github.com/opendilab/InterFuser](https://github.com/opendilab/InterFuser)
- **Our simplified implementation:** ~9M parameters

---

## Why Safety Matters

Traditional end-to-end driving models optimize for a single objective: follow the route. They produce waypoints or steering commands, but they never explicitly reason about whether the current plan is **safe**. This creates a fundamental problem:

1. **Silent failures.** A model might output waypoints that lead straight into an obstacle because nothing in the architecture forces it to reason about obstacle locations.
2. **Undebuggable decisions.** When the model makes a mistake, engineers cannot inspect what went wrong because the internal representations are opaque.
3. **No early warning.** There is no mechanism to detect that the model is uncertain or that the situation is dangerous -- it just outputs waypoints regardless.

InterFuser addresses all three problems:

- It predicts a **traffic density map** that explicitly represents where obstacles are, so you can verify the model's perception.
- It predicts a **waypoint heatmap** showing where the model thinks it should drive, so you can verify the model's planning logic.
- It outputs a **safety score** (0 to 1) that represents how confident the model is in the safety of its current plan. A low safety score can trigger conservative fallback behaviors (e.g., slow down, stop, hand over to a human).

This "safety-first" design philosophy means InterFuser is not just trying to drive well -- it is trying to drive safely and tell you when it cannot.

---

## Architecture

InterFuser processes multi-view camera images and LiDAR data through a shared transformer, then produces multiple output heads including interpretable maps and a safety score.

```
 INPUTS
 ======
 Front Camera (RGB)    Left Camera (RGB)    Right Camera (RGB)    LiDAR BEV
      |                      |                      |                  |
      v                      v                      v                  v
 +-----------+         +-----------+         +-----------+      +-----------+
 | CNN       |         | CNN       |         | CNN       |      | CNN       |
 | Encoder   |         | Encoder   |         | Encoder   |      | Encoder   |
 +-----------+         +-----------+         +-----------+      +-----------+
      |                      |                      |                  |
      v                      v                      v                  v
  [Tokens]              [Tokens]              [Tokens]            [Tokens]
      |                      |                      |                  |
      +----------+-----------+----------+----------+
                 |
                 v
     +---------------------------+
     |   JOINT TRANSFORMER       |
     |   ENCODER (6 layers)      |
     |                           |
     |   All tokens attend to    |
     |   each other: cameras     |
     |   attend to LiDAR,        |
     |   LiDAR attends to        |
     |   cameras, etc.           |
     +---------------------------+
                 |
                 v
         [Fused Features]
                 |
        +--------+--------+--------+
        |        |        |        |
        v        v        v        v
  +---------+ +-------+ +------+ +----------+
  | Density | | WP    | |Safety| | Waypoint |
  | Map     | | Heat  | |Score | | Decoder  |
  | Head    | | Map   | |Head  | | (X-Attn) |
  +---------+ +-------+ +------+ +----------+
        |        |        |        |
        v        v        v        v
   32x32 map  32x32   [0..1]   4 waypoints
   (obstacles) (path)  scalar   (x,y pairs)

 OUTPUTS
 =======
 - waypoints:        Where to drive (sequence of x,y coordinates)
 - density_map:      Where obstacles are (32x32 bird's-eye view)
 - waypoint_heatmap: Where the model thinks it should drive (32x32 BEV)
 - safety_score:     How safe is the current plan (0=dangerous, 1=safe)
```

---

## Key Concepts

### Tokenization

Each sensor input (front camera, left camera, right camera, LiDAR) is first processed by its own CNN encoder to extract spatial features. These feature maps are then "flattened" into sequences of tokens -- just like how words become tokens in language models. Each token represents a small spatial patch of the original input.

For example, a camera image encoded to a 4x8 feature map produces 32 tokens, each representing a different region of the image.

### Joint Transformer

The core innovation is the **joint transformer encoder**. All tokens from all modalities are concatenated into a single sequence and processed together. This means:

- Camera tokens can attend to LiDAR tokens (learning "this camera region corresponds to that LiDAR region")
- LiDAR tokens can attend to camera tokens (learning "this depth point corresponds to that visual object")
- Tokens from different camera views can attend to each other (learning spatial consistency across views)

This joint attention across all modalities is what makes the fusion "interpretable" -- the model must learn explicit correspondences between sensors rather than relying on opaque feature mixing.

### Density Map

The density map is a 32x32 bird's-eye-view grid that predicts where obstacles (vehicles, pedestrians, etc.) are located around the ego vehicle. Values close to 1.0 indicate high obstacle density; values close to 0.0 indicate free space.

Think of it as an implicit occupancy grid -- the model learns to predict this as a supervised auxiliary task, which forces the internal representations to encode spatial understanding of the scene.

### Safety Score

The safety score is a scalar between 0 and 1 that represents the model's confidence in its current driving plan being safe. It is produced by a small MLP head that reads the global fused features.

- **High safety score (close to 1.0):** The model is confident its plan avoids obstacles and follows traffic rules.
- **Low safety score (close to 0.0):** The model detects potential danger -- this can trigger defensive driving behaviors.

### Interpretability

"Interpretability" here means that engineers can inspect the model's intermediate outputs to understand its decision-making:

- If the model is about to hit something, the density map should show an obstacle in the path.
- If the model is turning left, the waypoint heatmap should show high values to the left.
- If neither map looks correct, you know exactly where the failure occurred.

This is fundamentally different from debugging a black-box model where you can only see inputs and outputs.

---

## Interpretable Outputs

InterFuser's interpretable outputs make it uniquely debuggable compared to other end-to-end models.

### Why This Matters for Engineering

When a self-driving car makes a mistake, the first question is always: "Why did the model do that?" With a typical end-to-end model, the answer is buried in millions of neural network weights. With InterFuser, you can systematically diagnose failures:

| Failure Mode | Density Map | Waypoint Heatmap | Safety Score | Diagnosis |
|---|---|---|---|---|
| Hits a car | Missing obstacle | Points at obstacle | Low | Perception failure |
| Hits a car | Shows obstacle | Points at obstacle | High | Planning failure |
| Misses a turn | Correct | Points straight | High | Map/route failure |
| Sudden stop | Empty | Empty/noisy | Low | Sensor input failure |

This structured debugging workflow is impossible with black-box models. The interpretable outputs essentially give you a built-in "explanation" for every driving decision.

### Visualizing the Outputs

In practice, engineers overlay the density map and waypoint heatmap on a bird's-eye view of the scene during evaluation. The density map appears as a red-to-blue heatmap showing obstacle locations, while the waypoint heatmap shows the planned driving corridor in green. Combined with the safety score displayed as a gauge, you get a complete picture of the model's reasoning at every timestep.

---

## How It Works Step by Step

Here is the complete forward pass, from raw sensor data to driving commands:

**Step 1: Capture sensor inputs**
- Front camera captures a 256x512 RGB image
- Left camera captures a 256x512 RGB image
- Right camera captures a 256x512 RGB image
- LiDAR point cloud is projected to a 256x256 bird's-eye-view image (2 channels: height + intensity)

**Step 2: Encode each modality into tokens**
- Each camera image passes through its own 3-layer CNN encoder (Conv -> ReLU -> Conv -> ReLU -> Conv -> ReLU), producing a feature map of shape (256, h, w)
- The LiDAR BEV passes through a separate CNN encoder with the same structure
- Each feature map is flattened spatially and transposed to become a sequence of 256-dimensional tokens

**Step 3: Concatenate all tokens**
- Tokens from all four modalities (3 cameras + 1 LiDAR) are concatenated into a single long sequence
- This sequence contains all spatial information from all sensors in a unified format

**Step 4: Joint transformer processing**
- The concatenated token sequence passes through 6 transformer encoder layers
- Each layer applies multi-head self-attention (8 heads) across ALL tokens
- After 6 layers, every token has been updated with information from every other modality and spatial location

**Step 5: Extract global features**
- Mean-pool across all fused tokens to get a single 256-dimensional global feature vector
- This vector summarizes the entire scene understanding

**Step 6: Produce interpretable outputs**
- **Density map:** MLP maps global features to 32x32 grid, sigmoid activation bounds values to [0,1]
- **Waypoint heatmap:** Same architecture, separate weights, predicts drivable areas
- **Safety score:** Small MLP maps global features to a single [0,1] value

**Step 7: Decode waypoints**
- Learnable waypoint queries (4 queries for 4 future waypoints) are cross-attended against the fused token sequence using a 2-layer transformer decoder
- Each query decodes into an (x, y) coordinate representing a future position
- Result: 4 waypoints forming the planned trajectory

**Step 8: Execute (downstream)**
- Waypoints are passed to a PID controller that converts them to steering, throttle, and brake
- Safety score can modulate execution (e.g., reduce speed if safety < threshold)

---

## Our Implementation

This is a simplified educational implementation (~9M parameters) that captures InterFuser's core ideas while being easy to understand and modify.

### What We Kept (Core Ideas)

- Multi-view camera + LiDAR input (4 modalities)
- Separate CNN encoders per modality, producing tokens
- Joint transformer encoder for cross-modal fusion
- Multiple interpretable output heads (density map, waypoint heatmap, safety score)
- Transformer decoder with learned queries for waypoint prediction

### What We Simplified

| Aspect | Original InterFuser | Our Implementation |
|---|---|---|
| Image backbone | ResNet-50 or EfficientNet | 3-layer CNN |
| Parameters | ~60M+ | ~9M |
| Transformer layers | 6+ with positional encodings | 6 standard layers |
| Waypoint decoder | GRU-based | Transformer decoder with cross-attention |
| Traffic light head | Yes (separate classifier) | Omitted |
| Training pipeline | CARLA data collection + multi-loss | Not included (model only) |
| Safety inference | Fallback controller logic | Safety score output only |

### Design Choices

- **Why 256-dim?** Balances expressiveness with computational cost for educational use.
- **Why 32x32 BEV maps?** Sufficient resolution to demonstrate the concept while keeping the model small.
- **Why transformer decoder for waypoints?** More modern approach than GRU; cross-attention explicitly queries the fused representation.

---

## Running the Code

### Prerequisites

```bash
pip install torch
```

### Run the Demo

```bash
cd one_step_e2e/InterFuser
python model.py
```

### Expected Output

```
InterFuser Demo
========================================
Parameters: 9,XXX,XXX
 
Waypoints: torch.Size([2, 4, 2])
Density map: torch.Size([2, 1, 32, 32])
Waypoint heatmap: torch.Size([2, 1, 32, 32])
Safety score: 0.XXX
```

### Understanding the Output Shapes

- `waypoints (B, 4, 2)`: Batch of 4 future waypoints, each with (x, y) coordinates
- `density_map (B, 1, 32, 32)`: Batch of 32x32 obstacle density maps (single channel)
- `waypoint_heatmap (B, 1, 32, 32)`: Batch of 32x32 drivable-area heatmaps
- `safety_score (B, 1)`: Batch of safety confidence values in [0, 1]

### Using the Model in Your Code

```python
import torch
from model import InterFuser

# Create model
model = InterFuser(d_model=256, n_heads=8, num_layers=6, num_waypoints=4, bev_size=32)

# Prepare inputs (batch_size=1)
front_img = torch.randn(1, 3, 256, 512)   # Front camera RGB
left_img = torch.randn(1, 3, 256, 512)    # Left camera RGB
right_img = torch.randn(1, 3, 256, 512)   # Right camera RGB
lidar_bev = torch.randn(1, 2, 256, 256)   # LiDAR bird's-eye view

# Forward pass
output = model(front_img, left_img, right_img, lidar_bev)

# Access outputs
waypoints = output['waypoints']           # (1, 4, 2) - planned trajectory
density = output['density_map']           # (1, 1, 32, 32) - obstacle map
heatmap = output['waypoint_heatmap']      # (1, 1, 32, 32) - drivable areas
safety = output['safety_score']           # (1, 1) - safety confidence

# Safety-based decision making
if safety.item() < 0.3:
    print("Low safety score! Consider emergency braking.")
```

---

## Results

InterFuser was evaluated on the CARLA Longest6 benchmark, a challenging urban driving scenario with dense traffic, intersections, and pedestrians.

### CARLA Leaderboard Results

| Method | Driving Score | Route Completion | Infraction Score |
|---|:---:|:---:|:---:|
| CILRS | 7.47 | 13.40 | 0.75 |
| LBC | 30.97 | 55.01 | 0.66 |
| TransFuser | 54.52 | 78.41 | 0.76 |
| **InterFuser** | **68.31** | **95.02** | **0.72** |
| TCP | 75.14 | 93.64 | 0.81 |

**Key observations:**
- InterFuser achieves **95.02% route completion** -- highest among the methods shown -- meaning it successfully navigates nearly all routes.
- The driving score of 68.31 reflects route completion weighted by infractions.
- The infraction score of 0.72 (lower is better) shows room for improvement in safety, which later models like TCP addressed.
- The strong route completion with interpretable outputs makes InterFuser particularly valuable for development and debugging.

### Why These Numbers Matter

- **Route completion** tells you how often the model finishes the assigned route (higher = better navigation).
- **Infraction score** penalizes collisions, red light violations, etc. (lower = fewer infractions).
- **Driving score** = route completion x infraction multiplier (higher = better overall).

InterFuser's high route completion with interpretable outputs means engineers can actually diagnose why the remaining failures happen -- something purely black-box models with higher scores cannot offer.

---

## References

1. **InterFuser Paper:** Hao Shao, Letian Wang, RuoBing Chen, Hongsheng Li, Yu Liu. "Safety-Enhanced Autonomous Driving Using Interpretable Sensor Fusion Transformer." CoRL 2022. [arXiv:2207.14024](https://arxiv.org/abs/2207.14024)

2. **Official Implementation:** [github.com/opendilab/InterFuser](https://github.com/opendilab/InterFuser)

3. **TransFuser** (predecessor): Prakash et al. "Multi-Modal Fusion Transformer for End-to-End Autonomous Driving." CVPR 2021.

4. **TCP** (successor): Wu et al. "Trajectory-guided Control Prediction for End-to-end Autonomous Driving: A Simple yet Strong Baseline." NeurIPS 2022.

5. **CARLA Simulator:** Dosovitskiy et al. "CARLA: An Open Urban Driving Simulator." CoRL 2017. [carla.org](https://carla.org)

---

---

## Training

### Quick Start

```bash
python train.py --epochs 5 --batch_size 2
```

Runs with synthetic CARLA-like data (no simulator needed).

### Loss Functions

| Loss | Source | Weight | Purpose |
|:---|:---:|:---:|:---|
| Waypoint L1 | `[FROM PAPER]` | 1.0 | Future waypoint regression |
| Density Map Focal | `[FROM PAPER]` | 1.0 | Object density heatmap (safety) |
| Safety Score BCE | `[FROM PAPER]` | 2.0 | Binary collision risk classifier |
| Traffic Light CE | `[FROM PAPER]` | 0.5 | Traffic light state classification |
| Junction CE | `[SELF-IMPLEMENTED]` | 0.3 | At-junction binary classifier |

### Key Arguments

```bash
python train.py \
    --epochs 50 \
    --batch_size 4 \
    --lr 1e-4 \
    --density_map_size 50 \
    --num_waypoints 4 \
    --num_samples 300 \
    --resume checkpoint.pth
```

### What the Training Script Includes

- **Multi-view camera fusion** (front + left + right + rear) `[FROM PAPER]`
- **Interpretable density maps** for safety-aware driving `[FROM PAPER]`
- **Focal loss** for density map (handles class imbalance) `[FROM PAPER]`
- **Safety score** as explicit collision risk output `[FROM PAPER]`
- **Traffic light state** classification head `[FROM PAPER]`
- **Validation metrics:** waypoint L1, density map MSE, safety AUC, TL accuracy
- **Mixed precision + gradient clipping** `[SELF-IMPLEMENTED]`
- **Cosine annealing LR with warmup** `[SELF-IMPLEMENTED]`

## File Structure

```
InterFuser/
├── README.md    # This file (beginner-friendly guide)
├── model.py     # InterFuser implementation (~9M params)
└── train.py     # Complete training pipeline (920+ lines)
```
