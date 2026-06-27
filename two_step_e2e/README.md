# Two-Step End-to-End Models

## Definition

In two-step E2E models, perception and planning are **distinct sub-networks** trained jointly. Perception features flow **directly** into the planning module without hand-crafted post-processing (unlike traditional modular pipelines where each component's output is post-processed before passing to the next).

```
Sensors → [Perception Network] →→→ learned features →→→ [Planning Network] → Trajectory
                                    (no post-processing)
```

## Key Characteristics

1. **Two distinct sub-networks** (perception + planning)
2. **Direct feature passing** between them (no NMS, no hand-crafted thresholds)
3. **End-to-end gradient flow** from planning loss back to perception
4. **Interpretable intermediate representations** (BEV maps, object queries, map vectors)
5. **Joint training** improves both perception and planning

## Models in This Directory

| Model | Venue | Key Innovation | Dataset |
|-------|-------|----------------|---------|
| [UniAD](UniAD/) | CVPR 2023 (Best Paper) | Unified full-stack (det+track+map+pred+plan) | nuScenes |
| [VAD](VAD/) | ICCV 2023 | Vectorized representation (efficient) | nuScenes |
| [ST-P3](ST-P3/) | ECCV 2022 | Spatial-temporal BEV + temporal GRU | nuScenes |

## How They Differ from One-Step Models

| Aspect | Two-Step | One-Step |
|--------|----------|----------|
| Intermediate output | Visible (BEV, objects, map) | Hidden (internal features) |
| Interpretability | High | Low |
| Modularity | Can inspect/debug perception | Black box |
| Efficiency | Generally slower (explicit perception) | Can be faster |
| Gradient flow | End-to-end through both sub-networks | Single network |

## How They Differ from Traditional Modular Pipelines

| Aspect | Two-Step E2E | Traditional Modular |
|--------|-------------|-------------------|
| Interface | Learned features (continuous) | Post-processed outputs (discrete) |
| Training | Joint end-to-end | Each module independent |
| Gradient flow | Through all modules | Blocked at boundaries |
| Error propagation | Mitigated by joint training | Accumulates |
| Information loss | Minimal | Significant at interfaces |
