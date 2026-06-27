# GAIA-1: A Generative World Model for Autonomous Driving

**Paper:** "GAIA-1: A Generative World Model for Autonomous Driving"  
**Authors:** Anthony Hu, Lloyd Russell, Hudson Yeo, Zak Murez, et al.  
**Organization:** Wayve  
**arXiv:** https://arxiv.org/abs/2309.17080  
**Year:** 2023

## Overview

GAIA-1 is a generative world model that learns to generate realistic driving videos conditioned on text, action, and video inputs. It represents the "world model" paradigm for autonomous driving — learning a model of how the world works, then using it for planning.

## The World Model Paradigm

```
Traditional E2E:    Sensor → [Model] → Action
World Model:        Sensor → [World Model] → Imagined Futures → [Planner] → Action
```

A world model learns to predict what happens next. For driving:
- Given current observation + proposed action → predict future observation
- Try many possible actions → pick the one with best imagined outcome
- This is essentially "mental simulation" for planning

### Analogy to Human Driving
Humans plan by imagining: "If I turn left, I'll enter the intersection when that car is still far away — safe." GAIA-1 does this computationally.

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                   GAIA-1 Architecture                     │
└──────────────────────────────────────────────────────────┘

    Video Frames         Action (steer, gas)      Text Description
         │                      │                       │
         ▼                      ▼                       ▼
  [Video Tokenizer]      [Action Encoder]       [Text Encoder]
  (VQ-VAE → discrete     (MLP → tokens)         (T5/CLIP →
   image tokens)                                   tokens)
         │                      │                       │
         └──────────────────────┼───────────────────────┘
                                │
                                ▼
                ┌───────────────────────────────┐
                │    Autoregressive Transformer  │
                │    (World Model)               │
                │                                │
                │    Predicts next video tokens  │
                │    conditioned on:             │
                │    • Past video tokens         │
                │    • Actions                   │
                │    • Text descriptions         │
                └───────────────┬───────────────┘
                                │
                                ▼
                    Predicted Future Video Tokens
                                │
                                ▼
                      [Video Decoder (VQ-VAE)]
                                │
                                ▼
                    Generated Future Video Frames
```

## Training

### Stage 1: Video Tokenizer (VQ-VAE)
- Trains discrete tokenizer on driving videos
- Compresses 256×256 frames to 16×16 discrete tokens
- Codebook size: 8192 codes
- Learns to reconstruct video frames from discrete codes

### Stage 2: World Model (Autoregressive Transformer)
- Trains transformer to predict next frame tokens given:
  - Past frame tokens (context)
  - Action tokens (what the car did)
  - Text tokens (scene description)
- Architecture: 6.5B parameter transformer
- Trained on massive driving video dataset

### Using World Model for Planning
```
For each candidate action sequence:
    1. Feed current observation + candidate actions to world model
    2. Generate imagined future frames (rollout)
    3. Evaluate imagined future (did we crash? make progress?)
    4. Score this action sequence
Select action sequence with best imagined outcome
```

## Key Features

- **9B parameters** (video tokenizer + world model)
- Generates realistic 25fps driving videos
- Controllable via text ("rainy weather", "busy intersection")
- Controllable via actions (steering, acceleration)
- Captures complex dynamics (other vehicles, pedestrians, weather)
- Emergent understanding of 3D geometry and physics

## Relationship to LLM Paradigm

| GPT | GAIA-1 |
|-----|--------|
| Text tokens | Video tokens + action tokens |
| Next token prediction | Next frame prediction |
| Language understanding | World understanding |
| Text generation | Video generation |
| In-context learning | Simulation/planning |

## Implementation Notes

- Full model: 9B parameters (not publicly released)
- Video tokenizer: separate VQ-VAE (~200M parameters)
- Requires massive compute for training (1000s of GPU-hours)
- Inference: generates at ~5 fps on 8×A100

## Simplified Implementation

Our implementation demonstrates the core concepts:
- Simplified video tokenizer (smaller VQ-VAE)
- Small-scale autoregressive world model
- Action-conditioned future prediction
- Planning via imagined rollouts

## Files

```
GAIA-1/
├── README.md       # This file
├── model.py        # World model implementation
├── tokenizer.py    # Video tokenizer (VQ-VAE)
└── planning.py     # Planning via world model rollouts
```
