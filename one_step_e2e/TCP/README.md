# TCP: Trajectory-guided Control Prediction for End-to-End Autonomous Driving

> A one-step end-to-end model that fuses trajectory planning with direct control prediction via a dual-branch architecture and adaptive fusion gate.

**Paper:** "TCP: Trajectory-guided Control Prediction for End-to-End Autonomous Driving"  
**Authors:** Penghao Wu, Xiaosong Jia, Li Chen, Junchi Yan, Hongyang Li, Yu Qiao  
**Venue:** NeurIPS 2022  
**arXiv:** [2206.08129](https://arxiv.org/abs/2206.08129)  
**Official Code:** [OpenDriveLab/TCP](https://github.com/OpenDriveLab/TCP)

---

## What is TCP?

TCP (Trajectory-guided Control Prediction) is a **one-step end-to-end** autonomous driving model. "One-step" means it goes directly from raw sensor inputs (camera images + LiDAR) to vehicle control commands (steering, throttle, brake) in a single forward pass -- no separate perception or planning modules.

The key innovation is its **dual-branch architecture**: one branch predicts a trajectory (future waypoints), while the other predicts control signals directly. Crucially, the trajectory branch **guides** the control branch through cross-attention, and an **adaptive fusion gate** intelligently blends both outputs based on the driving situation.

---

## The Problem TCP Solves

Before TCP, end-to-end driving models typically chose one of two approaches. Both have fundamental problems:

### Approach 1: Trajectory-Only (Predict Waypoints, then PID Controller)

```
Sensors --> Neural Network --> Waypoints --> PID Controller --> Steer/Gas/Brake
```

**The problem:** The PID controller converts waypoints into control commands using a simple formula. Even small errors in predicted waypoints get amplified by PID, causing oscillation or dangerous overshoot. There is also inherent latency -- the PID reacts to waypoint errors after they happen, not proactively.

### Approach 2: Control-Only (Directly Predict Control)

```
Sensors --> Neural Network --> Steer/Gas/Brake
```

**The problem:** Without the intermediate waypoint representation, the model loses interpretability and tends to produce jerky outputs. It is also prone to "mode collapse" -- averaging over multiple valid driving behaviors produces outputs that match none of them well (e.g., averaging "turn left" and "turn right" produces "go straight" into a wall).

### TCP's Solution: Why Not Both?

TCP recognizes that trajectories and direct control have **complementary strengths**. Instead of choosing one, it predicts both and uses the trajectory to inform the control prediction:

```
Sensors --> Neural Network --> BOTH trajectory AND control
                              (trajectory GUIDES control via attention)
                              --> Adaptive fusion picks the best blend
```

---

## Architecture

```
        ┌────────────┐    ┌──────────────┐
        │ Camera RGB │    │  LiDAR BEV   │
        │ (B,3,H,W)  │    │  (B,2,H,W)  │
        └─────┬──────┘    └──────┬───────┘
              │                   │
              ▼                   ▼
     ┌────────────────┐  ┌──────────────────┐
     │  Image Encoder │  │  LiDAR Encoder   │
     │  (CNN → 512-d) │  │  (CNN → 256-d)   │
     └────────┬───────┘  └──────┬───────────┘
              │                   │
              └─────────┬─────────┘
                        ▼
              ┌────────────────────┐
              │  Feature Fusion    │     ┌──────────────┐
              │  (MLP: 768 → 512) │ ◄───│ Speed Embed  │
              └────────┬───────────┘     │ (1 → 512)   │
                       │                 └──────────────┘
                       │
          ┌────────────┼────────────────────────┐
          │            │                        │
          ▼            │                        ▼
┌──────────────────┐   │          ┌──────────────────────────┐
│ TRAJECTORY BRANCH│   │          │    CONTROL BRANCH        │
│                  │   │          │                          │
│ GRU Decoder      │   │          │  Features attend to      │
│ (4 steps)        │   │          │  trajectory via          │
│      │           │   │          │  Cross-Attention         │
│      ▼           │───┼─────────►│  (MultiheadAttention)    │
│ 4 Waypoints      │ trajectory   │      │                   │
│ (x, y) each      │ guidance     │      ▼                   │
└────────┬─────────┘              │  Control MLP             │
         │                        │  → (steer, throttle,     │
         │                        │     brake)               │
         │                        └────────────┬─────────────┘
         │                                     │
         │     ┌───────────────────────┐       │
         └────►│   ADAPTIVE FUSION     │◄──────┘
               │   GATE (sigma)        │
               │                       │
               │  gate * traj_steer    │
               │  + (1-gate) * ctrl    │
               └───────────┬───────────┘
                           │
                           ▼
                  Final Steering Output
```

---

## Key Concepts

### 1. Trajectory Branch (GRU Decoder)

The trajectory branch predicts **where the car should go** in the near future as a sequence of waypoints (x, y coordinates relative to the ego vehicle).

- Uses a **GRU (Gated Recurrent Unit)** to autoregressively decode waypoints one at a time
- Each step: the GRU takes the fused features as input and its previous hidden state, then a linear head projects the hidden state to (dx, dy)
- Produces `num_waypoints` (default: 4) future positions
- These waypoints are interpretable -- you can visualize them to understand what the model is "planning"

### 2. Control Branch (Trajectory-Guided)

The control branch predicts **what the car should do right now** -- concrete steer, throttle, and brake values.

- Does NOT work in isolation; it is guided by the trajectory branch
- The predicted trajectory is projected into a feature representation
- A **cross-attention** mechanism lets the driving features "attend to" trajectory information
- The attended features are concatenated with the original features and fed to a control MLP
- Outputs: steer in [-1, 1] (tanh), throttle in [0, 1] (sigmoid), brake in [0, 1] (sigmoid)

### 3. Cross-Attention Guidance

This is how the trajectory branch **informs** the control branch:

1. The 4 predicted waypoints are flattened: (B, 4, 2) becomes (B, 8)
2. A linear projection maps them to the hidden dimension: (B, 8) becomes (B, hidden_dim)
3. Multi-head attention: the fused sensor features (query) attend to the trajectory features (key/value)
4. The attended output captures "what does the trajectory tell us about the right control action?"

Think of it this way: the trajectory branch says "we need to turn right in 2 seconds," and cross-attention lets the control branch incorporate that foresight into its immediate control decision.

### 4. Adaptive Fusion Gate

After both branches produce their outputs, a learned gate decides how much to trust each one. See the next section for details.

---

## The Adaptive Fusion Gate

The adaptive fusion gate is perhaps the most elegant part of TCP. It is a small neural network that dynamically decides: **"Should I trust the trajectory-derived steering, or the direct control prediction?"**

### How It Works

```
gate_input = [features, control_output]    # What the model sees + what control branch predicted
gate = sigmoid(Linear(gate_input))         # Output in [0, 1]

final_steer = gate * trajectory_steer + (1 - gate) * direct_steer
```

- **gate close to 1.0**: Trust the trajectory branch (convert waypoints to steering via atan2)
- **gate close to 0.0**: Trust the direct control branch
- **gate around 0.5**: Blend both equally

### Why This Matters

The gate is **learned**, not hand-tuned. During training, the model discovers:

| Situation | Gate Behavior | Reasoning |
|-----------|---------------|-----------|
| Straight highway | gate -> high | Trajectory is smooth and reliable; PID works fine for gentle curves |
| Sharp turns | gate -> low | Trajectory errors amplify in tight turns; direct control is more responsive |
| Ambiguous scenarios | gate -> medium | Neither branch is confident; blend for safety |

This adaptive behavior means TCP automatically adjusts its strategy based on the driving context -- no manual tuning of when to use which approach.

### The Trajectory-to-Steer Conversion

Before fusion, the trajectory branch's waypoints must be converted to a steering angle for comparison:

```python
aim = waypoints[:, 0]                          # First predicted waypoint (x, y)
traj_steer = atan2(y, x + epsilon) / 1.57      # Convert to normalized steering [-1, 1]
```

This is a simplified "PID" -- point toward the first waypoint. The angle is divided by pi/2 (1.57) to normalize to [-1, 1].

---

## How It Works Step by Step

Here is the complete forward pass, from sensors to control output:

**Step 1: Feature Extraction**
- Camera image passes through a CNN encoder producing a 512-d vector
- LiDAR bird's-eye-view passes through a separate CNN producing a 256-d vector
- Both vectors are concatenated (768-d) and fused via MLP to 512-d
- Current vehicle speed is embedded and added to the fused features

**Step 2: Trajectory Branch**
- The GRU decoder iteratively produces 4 waypoints
- Each iteration: GRU(features, previous_hidden) produces new hidden state
- A linear head maps hidden state to (dx, dy) -- one waypoint
- Result: 4 waypoints describing the predicted future path

**Step 3: Cross-Attention Guidance**
- The 4 waypoints are flattened and projected to the hidden dimension
- Multi-head attention (4 heads): sensor features query the trajectory representation
- Output: trajectory-informed features capturing planning context

**Step 4: Control Prediction**
- Original features and attention output are concatenated
- An MLP predicts raw (steer, throttle, brake)
- Activations: tanh for steer, sigmoid for throttle and brake

**Step 5: Adaptive Fusion**
- Trajectory's first waypoint is converted to a steering angle
- The fusion gate (MLP + sigmoid) evaluates how much to trust each branch
- Final steering = gate * trajectory_steer + (1 - gate) * direct_steer
- Throttle and brake come directly from the control branch

**Step 6: Output**
- Returns waypoints (for visualization/interpretability), direct control, fused steering, and the gate value

---

## Our Implementation

This is a **simplified educational implementation** of TCP that preserves the core architectural ideas while being easy to read and modify.

| Aspect | Official TCP | Our Implementation |
|--------|-------------|-------------------|
| Image encoder | ResNet-34 pretrained | Lightweight 4-layer CNN |
| LiDAR encoder | ResNet-18 | Lightweight 3-layer CNN |
| Hidden dimension | 512 | 256 (configurable) |
| Waypoints | 4 | 4 |
| Parameters | ~25M | **~3M** |
| Training data | CARLA expert demonstrations | Not included (bring your own) |
| Fusion | Full dual-branch + ensemble | Simplified gate on steering |

### File Structure

```
TCP/
├── README.md      # This file -- architecture explanation and guide
└── model.py       # Complete TCP model implementation (~200 lines)
```

### Key Design Choices

- **Dual encoders**: Separate CNNs for camera and LiDAR preserve modality-specific features
- **Speed conditioning**: Speed is embedded and added (not concatenated) to features, following the paper's approach
- **GRU for trajectory**: Autoregressive decoding naturally captures sequential waypoint dependencies
- **Multi-head attention**: 4 attention heads allow the control branch to attend to different aspects of the trajectory

---

## Running the Code

### Prerequisites

```bash
pip install torch  # PyTorch (CPU or GPU)
```

### Quick Demo

```bash
cd one_step_e2e/TCP
python model.py
```

Expected output:
```
TCP: Trajectory-guided Control Prediction Demo
==================================================
Parameters: 3,XXX,XXX
Waypoints: torch.Size([4, 4, 2])
Control: torch.Size([4, 3])
Fused steer: 0.XXX
Fusion gate: 0.XXX
  (gate=1 -> trust trajectory, gate=0 -> trust direct control)
```

### Using in Your Code

```python
import torch
from model import TCP, compute_tcp_loss

# Create model
model = TCP(num_waypoints=4, hidden_dim=256)

# Prepare inputs
image = torch.randn(1, 3, 256, 512)       # Front camera
lidar_bev = torch.randn(1, 2, 256, 256)   # LiDAR bird's-eye view
speed = torch.tensor([[5.0]])              # Current speed (m/s)

# Forward pass
output = model(image, lidar_bev, speed)

# Access outputs
print(output['waypoints'].shape)     # (1, 4, 2) -- predicted trajectory
print(output['control'])             # (1, 3) -- [steer, throttle, brake]
print(output['fused_steer'])         # (1, 1) -- final fused steering
print(output['fusion_gate'])         # (1, 1) -- gate value [0, 1]
```

### Training Loop (Skeleton)

```python
model = TCP(num_waypoints=4, hidden_dim=256).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for image, lidar, speed, gt_waypoints, gt_control in dataloader:
    output = model(image, lidar, speed)
    losses = compute_tcp_loss(output, gt_waypoints, gt_control)

    optimizer.zero_grad()
    losses['total'].backward()
    optimizer.step()

    # losses['trajectory'] -- waypoint L1 loss
    # losses['control']    -- control L1 loss
    # losses['fused']      -- fused steering L1 loss
```

---

## Results

TCP achieved **state-of-the-art** performance on the CARLA Longest6 benchmark at the time of publication (2022):

| Method | Driving Score | Route Completion | Infraction Score |
|--------|:---:|:---:|:---:|
| CILRS | 7.47 | 13.40 | 0.75 |
| LBC | 30.97 | 55.01 | 0.73 |
| TransFuser | 54.52 | 78.41 | 0.86 |
| InterFuser | 68.31 | 95.02 | 0.77 |
| **TCP** | **75.14** | **93.64** | **0.87** |

**Driving Score** = Route Completion x Infraction Score. It penalizes collisions, red-light violations, and other infractions while rewarding route progress. A score of 75.14 means TCP reliably completes most routes with minimal safety violations.

---

## Why TCP Outperforms

### 1. Complementary Strengths

The dual-branch design captures two fundamentally different aspects of driving:

- **Trajectory branch** excels at smooth, long-horizon planning (thinking ahead)
- **Control branch** excels at immediate, precise reactions (responding now)

By combining them, TCP gets the planning horizon of trajectory methods AND the responsiveness of direct control.

### 2. Trajectory Guidance Prevents Mode Collapse

The control branch does not predict blindly -- it is informed by the trajectory prediction. This provides a strong inductive bias: "the control should be consistent with where I plan to go." This prevents the control branch from producing contradictory outputs.

### 3. Adaptive Fusion Handles Edge Cases

The learned gate automatically shifts trust based on the situation:
- When trajectory prediction is confident and smooth: trust trajectory (highway driving)
- When the situation is complex and needs fast reactions: trust direct control (intersections, obstacles)
- This means TCP gracefully handles scenarios that would trip up either branch alone.

### 4. Multi-Task Learning Regularization

Training both branches simultaneously provides mutual regularization:
- The trajectory loss forces the shared encoder to learn spatial understanding
- The control loss forces the encoder to learn action-relevant features
- Together, they produce richer, more generalizable representations than either alone

---

## References

1. **TCP Paper:** Wu, P., Jia, X., Chen, L., Yan, J., Li, H., & Qiao, Y. (2022). "Trajectory-guided Control Prediction for End-to-end Autonomous Driving: A Simple yet Strong Baseline." *NeurIPS 2022*. [arXiv:2206.08129](https://arxiv.org/abs/2206.08129)

2. **Official Implementation:** [https://github.com/OpenDriveLab/TCP](https://github.com/OpenDriveLab/TCP)

3. **CARLA Simulator:** Dosovitskiy, A., et al. (2017). "CARLA: An Open Urban Driving Simulator." *CoRL 2017*. [http://carla.org](http://carla.org)

4. **TransFuser:** Prakash, A., et al. (2021). "Multi-Modal Fusion Transformer for End-to-End Autonomous Driving." *CVPR 2021*.

5. **InterFuser:** Shao, H., et al. (2023). "Safety-Enhanced Autonomous Driving Using Interpretable Sensor Fusion Transformer." *CoRL 2022*.

---

---

## Training

### Quick Start

```bash
python train.py --epochs 1 --batch-size 4
```

Note: uses `--batch-size` (hyphen, not underscore). Runs with synthetic data.

### Loss Functions

| Loss | Source | Weight | Purpose |
|:---|:---:|:---:|:---|
| Trajectory L1 | `[FROM PAPER]` | 1.0 | Waypoint prediction (trajectory branch) |
| Control L1 | `[FROM PAPER]` | 1.0 | Steer/throttle/brake (control branch) |
| Fused Steer L1 | `[FROM PAPER]` | 0.5 | Adaptively fused steering output |
| Speed L1 | `[FROM PAPER]` | 0.05 | Speed prediction regularizer |
| Feature Alignment | `[SELF-IMPLEMENTED]` | 0.1 | Align trajectory & control features |

### Key Arguments

```bash
python train.py \
    --epochs 50 \
    --batch-size 8 \
    --lr 1e-4 \
    --num-waypoints 4 \
    --num-samples 500 \
    --resume checkpoint.pth
```

### What the Training Script Includes

- **Dual-branch architecture** (trajectory + direct control) `[FROM PAPER]`
- **Adaptive fusion** with learned confidence weights `[FROM PAPER]`
- **Multi-task training** of both branches simultaneously `[FROM PAPER]`
- **Speed-conditioned fusion** weighting `[FROM PAPER]`
- **Validation metrics:** trajectory L1, control MAE, fused steer error
- **Mixed precision + gradient clipping** `[SELF-IMPLEMENTED]`
- **Cosine annealing LR** `[SELF-IMPLEMENTED]`

## Files

```
TCP/
├── README.md    # This documentation
├── model.py     # TCP model implementation
└── train.py     # Complete training pipeline (820+ lines)
```

---

*This implementation is part of the [perception-planning-e2e-models](../../README.md) educational repository exploring end-to-end autonomous driving architectures.*
