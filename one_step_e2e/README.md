# One-Step End-to-End Models

## Definition

One-step E2E models map **directly** from raw sensor input to planning/control output with NO explicitly exposed intermediate perception representation.

```
Sensors → [Single Neural Network] → Trajectory / Control Signals
```

## Key Characteristics

1. **Single network** from sensors to actions
2. **No intermediate perception output** (internal features exist but are not interpretable)
3. **Potentially optimal** internal representations (learned for planning, not perception)
4. **Less interpretable** but can be more efficient

## Two Sub-Categories

### Traditional Deep Learning (CNN/Transformer)
Models that use standard architectures (ResNets, Transformers, GRUs) to directly map sensor inputs to driving outputs via imitation learning.

| Model | Venue | Key Innovation |
|-------|-------|----------------|
| [TransFuser](TransFuser/) | CVPR 2022 / PAMI 2023 | Multi-scale transformer fusion (image + LiDAR) |
| [InterFuser](InterFuser/) | CoRL 2022 | Safety maps + multi-modal transformer |
| [TCP](TCP/) | NeurIPS 2022 | Trajectory-guided control (dual branch) |

### Foundation Model / LLM-like Approaches (NEW PARADIGM)
Models that apply the foundation model training paradigm (pretrain → fine-tune → RL) to driving, often using vision-language models or generative world models.

| Model | Year | Paradigm |
|-------|------|----------|
| [DriveVLM](DriveVLM/) | 2024 | Vision-Language Model + chain-of-thought reasoning |
| [GAIA-1](GAIA-1/) | 2023 | Generative world model (imagine futures for planning) |
| [GenAD](GenAD/) | 2024 | Diffusion model for diverse trajectory generation |

## The Foundation Model Paradigm Explained

Traditional E2E training:
```
Data → [Model] → Supervised Learning → Done
```

Foundation model training (like GPT):
```
Stage 1: Pre-training     → Learn general visual/world understanding
Stage 2: Fine-tuning      → Adapt to driving tasks (trajectory prediction)
Stage 3: Reinforcement    → Improve via driving reward (safety, comfort)
         Learning
```

### Key Ideas

- **World Models (GAIA-1):** Learn to predict "what happens next" given actions. Plan by imagining outcomes of different actions.
- **Vision-Language Models (DriveVLM):** Use VLM reasoning for driving decisions. Chain-of-thought explains decisions.
- **Generative Models (GenAD):** Generate DIVERSE trajectory proposals (not just one), naturally handling multi-modal driving.
- **Foundation + RL:** Pre-train on data, then improve online via reinforcement learning (analogous to RLHF for LLMs).

## Comparison of All One-Step Approaches

| Aspect | Traditional DL | Foundation Model |
|--------|---------------|-----------------|
| Training data | CARLA expert | Massive real-world video |
| Architecture | CNN + Transformer | VLM (7B+ params) |
| Output | Waypoints / Control | Waypoints + Explanation |
| Interpretability | Low | Medium (chain-of-thought) |
| Generalization | Limited | Strong (pre-trained knowledge) |
| Inference speed | Fast (10+ FPS) | Slow (1-2 FPS) |
| Multi-modal output | Usually single | Naturally diverse |
