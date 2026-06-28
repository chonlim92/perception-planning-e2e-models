"""
Comprehensive Edge Case / Stress Tests for all 13 E2E Perception-Planning Models.

Tests:
1. Batch size 1 (B=1)
2. Large batch (B=8)
3. Single timestep (T=1) for temporal models
4. Maximum sequence length for GAIA-1
5. All-zero input
6. All-same mask (all-True and all-False)
7. Single candidate (GenAD num_samples=1, VAD K=1)
8. Very large/small input values
9. Half precision (fp16)
"""

import sys
import os
import traceback

# Add model paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'one_step_e2e', 'GAIA-1'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'one_step_e2e', 'GenAD'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'one_step_e2e', 'DriveVLM'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'one_step_e2e', 'InterFuser'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'one_step_e2e', 'TCP'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'one_step_e2e', 'TransFuser'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'two_step_e2e', 'ST-P3'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'two_step_e2e', 'UniAD'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'two_step_e2e', 'VAD'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'planner_scorer', 'learned'))

import torch
import torch.nn as nn

# Results tracking
results = []

def record(test_name, model_name, status, detail=""):
    results.append((test_name, model_name, status, detail))
    symbol = {"PASS": "PASS", "WARN": "WARN", "FAIL": "FAIL"}[status]
    print(f"  [{symbol}] {model_name}: {detail}" if detail else f"  [{symbol}] {model_name}")


def check_output_health(output, model_name, test_name):
    """Check if output contains NaN or Inf."""
    if isinstance(output, dict):
        for k, v in output.items():
            if isinstance(v, torch.Tensor):
                if torch.isnan(v).any():
                    record(test_name, model_name, "WARN", f"NaN in output['{k}']")
                    return False
                if torch.isinf(v).any():
                    record(test_name, model_name, "WARN", f"Inf in output['{k}']")
                    return False
            elif isinstance(v, dict):
                if not check_output_health(v, model_name, test_name):
                    return False
    elif isinstance(output, torch.Tensor):
        if torch.isnan(output).any():
            record(test_name, model_name, "WARN", f"NaN in output tensor")
            return False
        if torch.isinf(output).any():
            record(test_name, model_name, "WARN", f"Inf in output tensor")
            return False
    return True


# ============================================================
# Model factory functions (small configs for testing)
# ============================================================

def make_gaia1_models(device):
    import importlib
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'one_step_e2e', 'GAIA-1'))
    if 'model' in sys.modules:
        del sys.modules['model']
    import model as gaia1_model
    importlib.reload(gaia1_model)
    tokenizer = gaia1_model.VideoTokenizer(latent_dim=32, num_codes=256).to(device)
    world_model = gaia1_model.WorldModelTransformer(
        num_codes=256, d_model=128, n_heads=4,
        num_layers=2, tokens_per_frame=16, max_frames=16).to(device)
    return tokenizer, world_model


def make_genad(device):
    # Need to re-import from correct path
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'one_step_e2e', 'GenAD'))
    import importlib
    if 'model' in sys.modules:
        del sys.modules['model']
    import model as genad_model
    importlib.reload(genad_model)
    m = genad_model.GenAD(scene_dim=64, hidden_dim=128,
                          num_waypoints=6, num_diffusion_steps=10).to(device)
    return m


def make_drivevlm(device):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'one_step_e2e', 'DriveVLM'))
    import importlib
    if 'model' in sys.modules:
        del sys.modules['model']
    import model as dvlm_model
    importlib.reload(dvlm_model)
    m = dvlm_model.DriveVLM(
        visual_dim=192, llm_dim=256,
        num_query_tokens=16, vocab_size=100).to(device)
    return m


def make_interfuser(device):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'one_step_e2e', 'InterFuser'))
    import importlib
    if 'model' in sys.modules:
        del sys.modules['model']
    import model as if_model
    importlib.reload(if_model)
    m = if_model.InterFuser(d_model=128, n_heads=4,
                            num_layers=2, num_waypoints=4, bev_size=16).to(device)
    return m


def make_tcp(device):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'one_step_e2e', 'TCP'))
    import importlib
    if 'model' in sys.modules:
        del sys.modules['model']
    import model as tcp_model
    importlib.reload(tcp_model)
    m = tcp_model.TCP(num_waypoints=4, hidden_dim=128).to(device)
    return m


def make_transfuser(device):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'one_step_e2e', 'TransFuser'))
    import importlib
    if 'model' in sys.modules:
        del sys.modules['model']
    import model as tf_model
    importlib.reload(tf_model)
    m = tf_model.TransFuser(num_waypoints=4, hidden_dim=256).to(device)
    return m


def make_stp3(device):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'two_step_e2e', 'ST-P3'))
    import importlib
    if 'model' in sys.modules:
        del sys.modules['model']
    import model as stp3_model
    importlib.reload(stp3_model)
    m = stp3_model.STP3(bev_channels=32, bev_h=50, bev_w=50,
                         num_cameras=6, num_waypoints=6, temporal_frames=4).to(device)
    return m


def make_uniad(device):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'two_step_e2e', 'UniAD'))
    import importlib
    if 'model' in sys.modules:
        del sys.modules['model']
    if 'config' in sys.modules:
        del sys.modules['config']
    import config as uniad_config
    importlib.reload(uniad_config)
    import model as uniad_model
    importlib.reload(uniad_model)
    cfg = uniad_config.UniADConfig()
    # Override to smaller sizes for testing
    cfg.bev.bev_h = 50
    cfg.bev.bev_w = 50
    cfg.track.num_queries = 50
    cfg.track.num_layers = 2
    cfg.track.ffn_dim = 512
    cfg.map.num_queries = 20
    cfg.map.num_layers = 2
    cfg.motion.future_steps = 6
    cfg.planner.num_future_steps = 6
    m = uniad_model.UniAD(cfg).to(device)
    return m


def make_vad(device):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'two_step_e2e', 'VAD'))
    import importlib
    if 'model' in sys.modules:
        del sys.modules['model']
    import model as vad_model
    importlib.reload(vad_model)
    m = vad_model.VAD(embed_dim=128, bev_h=50, bev_w=50,
                      num_cameras=6, num_ego_queries=6, num_waypoints=6).to(device)
    return m


def make_mlp_scorer(device):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'planner_scorer', 'learned'))
    import importlib
    if 'mlp_scorer' in sys.modules:
        del sys.modules['mlp_scorer']
    import mlp_scorer as ms
    importlib.reload(ms)
    m = ms.MLPScorer(traj_points=16, traj_dim=4, hidden_dim=128).to(device)
    return m


def make_transformer_scorer(device):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'planner_scorer', 'learned'))
    import importlib
    if 'transformer_scorer' in sys.modules:
        del sys.modules['transformer_scorer']
    import transformer_scorer as ts
    importlib.reload(ts)
    m = ts.TransformerScorer(d_model=128, n_heads=4).to(device)
    return m


# ============================================================
# TEST 1: Batch size 1
# ============================================================
def test_batch_size_1():
    print("\n" + "="*70)
    print("TEST 1: Batch Size B=1")
    print("="*70)
    device = 'cpu'

    # GAIA-1 Tokenizer
    try:
        tok, wm = make_gaia1_models(device)
        tok.eval(); wm.eval()
        frame = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            out = tok(frame)
        if check_output_health(out, "GAIA-1 Tokenizer", "B=1"):
            record("B=1", "GAIA-1 Tokenizer", "PASS")
    except Exception as e:
        record("B=1", "GAIA-1 Tokenizer", "FAIL", str(e))

    # GAIA-1 World Model
    try:
        frames = torch.randint(0, 256, (1, 4, 16))
        actions = torch.randn(1, 4, 3)
        with torch.no_grad():
            out = wm(frames, actions)
        if check_output_health(out, "GAIA-1 WorldModel", "B=1"):
            record("B=1", "GAIA-1 WorldModel", "PASS")
    except Exception as e:
        record("B=1", "GAIA-1 WorldModel", "FAIL", str(e))

    # GenAD
    try:
        model = make_genad(device)
        model.eval()
        imgs = torch.randn(1, 3, 64, 128)
        with torch.no_grad():
            out = model(imgs, num_samples=4)
        if check_output_health(out, "GenAD", "B=1"):
            record("B=1", "GenAD", "PASS")
    except Exception as e:
        record("B=1", "GenAD", "FAIL", str(e))

    # DriveVLM
    try:
        model = make_drivevlm(device)
        model.eval()
        imgs = torch.randn(1, 6, 3, 224, 224)
        with torch.no_grad():
            out = model(imgs)
        if check_output_health(out, "DriveVLM", "B=1"):
            record("B=1", "DriveVLM", "PASS")
    except Exception as e:
        record("B=1", "DriveVLM", "FAIL", str(e))

    # InterFuser
    try:
        model = make_interfuser(device)
        model.eval()
        f = torch.randn(1, 3, 128, 256)
        l = torch.randn(1, 3, 128, 256)
        r = torch.randn(1, 3, 128, 256)
        lid = torch.randn(1, 2, 128, 128)
        with torch.no_grad():
            out = model(f, l, r, lid)
        if check_output_health(out, "InterFuser", "B=1"):
            record("B=1", "InterFuser", "PASS")
    except Exception as e:
        record("B=1", "InterFuser", "FAIL", str(e))

    # TCP
    try:
        model = make_tcp(device)
        model.eval()
        img = torch.randn(1, 3, 128, 256)
        lid = torch.randn(1, 2, 128, 128)
        spd = torch.tensor([[5.0]])
        with torch.no_grad():
            out = model(img, lid, spd)
        if check_output_health(out, "TCP", "B=1"):
            record("B=1", "TCP", "PASS")
    except Exception as e:
        record("B=1", "TCP", "FAIL", str(e))

    # TransFuser
    try:
        model = make_transfuser(device)
        model.eval()
        img = torch.randn(1, 3, 256, 512)
        lid = torch.randn(1, 2, 256, 256)
        spd = torch.tensor([[5.0]])
        with torch.no_grad():
            out = model(img, lid, spd)
        if check_output_health(out, "TransFuser", "B=1"):
            record("B=1", "TransFuser", "PASS")
    except Exception as e:
        record("B=1", "TransFuser", "FAIL", str(e))

    # ST-P3
    try:
        model = make_stp3(device)
        model.eval()
        imgs = torch.randn(1, 4, 6, 3, 64, 128)
        with torch.no_grad():
            out = model(imgs)
        if check_output_health(out, "ST-P3", "B=1"):
            record("B=1", "ST-P3", "PASS")
    except Exception as e:
        record("B=1", "ST-P3", "FAIL", str(e))

    # UniAD
    try:
        model = make_uniad(device)
        model.eval()
        imgs = torch.randn(1, 6, 3, 128, 200)
        with torch.no_grad():
            out = model(imgs)
        if check_output_health(out, "UniAD", "B=1"):
            record("B=1", "UniAD", "PASS")
    except Exception as e:
        record("B=1", "UniAD", "FAIL", str(e))

    # VAD
    try:
        model = make_vad(device)
        model.eval()
        imgs = torch.randn(1, 6, 3, 128, 200)
        with torch.no_grad():
            out = model(imgs)
        if check_output_health(out, "VAD", "B=1"):
            record("B=1", "VAD", "PASS")
    except Exception as e:
        record("B=1", "VAD", "FAIL", str(e))

    # MLP Scorer
    try:
        model = make_mlp_scorer(device)
        model.eval()
        traj = torch.randn(1, 16, 4)
        agents = torch.randn(1, 32, 7)
        mask = torch.ones(1, 32, dtype=torch.bool)
        mapf = torch.randn(1, 64, 5)
        with torch.no_grad():
            out = model(traj, agents, mask, mapf)
        if check_output_health(out, "MLP Scorer", "B=1"):
            record("B=1", "MLP Scorer", "PASS")
    except Exception as e:
        record("B=1", "MLP Scorer", "FAIL", str(e))

    # Transformer Scorer
    try:
        model = make_transformer_scorer(device)
        model.eval()
        traj = torch.randn(1, 16, 4)
        agents = torch.randn(1, 32, 7)
        amask = torch.ones(1, 32, dtype=torch.bool)
        mapf = torch.randn(1, 64, 5)
        mmask = torch.ones(1, 64, dtype=torch.bool)
        with torch.no_grad():
            out = model(traj, agents, amask, mapf, mmask)
        if check_output_health(out, "Transformer Scorer", "B=1"):
            record("B=1", "Transformer Scorer", "PASS")
    except Exception as e:
        record("B=1", "Transformer Scorer", "FAIL", str(e))


# ============================================================
# TEST 2: Large batch B=8
# ============================================================
def test_large_batch():
    print("\n" + "="*70)
    print("TEST 2: Large Batch B=8")
    print("="*70)
    device = 'cpu'
    B = 8

    # GAIA-1 Tokenizer
    try:
        tok, wm = make_gaia1_models(device)
        tok.eval(); wm.eval()
        frame = torch.randn(B, 3, 64, 64)
        with torch.no_grad():
            out = tok(frame)
        if check_output_health(out, "GAIA-1 Tokenizer", "B=8"):
            record("B=8", "GAIA-1 Tokenizer", "PASS")
    except Exception as e:
        record("B=8", "GAIA-1 Tokenizer", "FAIL", str(e))

    # GAIA-1 World Model
    try:
        frames = torch.randint(0, 256, (B, 4, 16))
        actions = torch.randn(B, 4, 3)
        with torch.no_grad():
            out = wm(frames, actions)
        if check_output_health(out, "GAIA-1 WorldModel", "B=8"):
            record("B=8", "GAIA-1 WorldModel", "PASS")
    except Exception as e:
        record("B=8", "GAIA-1 WorldModel", "FAIL", str(e))

    # GenAD (eval mode with sampling)
    try:
        model = make_genad(device)
        model.eval()
        imgs = torch.randn(B, 3, 64, 128)
        with torch.no_grad():
            out = model(imgs, num_samples=4)
        if check_output_health(out, "GenAD", "B=8"):
            record("B=8", "GenAD", "PASS")
    except Exception as e:
        record("B=8", "GenAD", "FAIL", str(e))

    # DriveVLM
    try:
        model = make_drivevlm(device)
        model.eval()
        imgs = torch.randn(B, 6, 3, 224, 224)
        with torch.no_grad():
            out = model(imgs)
        if check_output_health(out, "DriveVLM", "B=8"):
            record("B=8", "DriveVLM", "PASS")
    except Exception as e:
        record("B=8", "DriveVLM", "FAIL", str(e))

    # InterFuser
    try:
        model = make_interfuser(device)
        model.eval()
        f = torch.randn(B, 3, 128, 256)
        l = torch.randn(B, 3, 128, 256)
        r = torch.randn(B, 3, 128, 256)
        lid = torch.randn(B, 2, 128, 128)
        with torch.no_grad():
            out = model(f, l, r, lid)
        if check_output_health(out, "InterFuser", "B=8"):
            record("B=8", "InterFuser", "PASS")
    except Exception as e:
        record("B=8", "InterFuser", "FAIL", str(e))

    # TCP
    try:
        model = make_tcp(device)
        model.eval()
        img = torch.randn(B, 3, 128, 256)
        lid = torch.randn(B, 2, 128, 128)
        spd = torch.rand(B, 1) * 10
        with torch.no_grad():
            out = model(img, lid, spd)
        if check_output_health(out, "TCP", "B=8"):
            record("B=8", "TCP", "PASS")
    except Exception as e:
        record("B=8", "TCP", "FAIL", str(e))

    # TransFuser
    try:
        model = make_transfuser(device)
        model.eval()
        img = torch.randn(B, 3, 256, 512)
        lid = torch.randn(B, 2, 256, 256)
        spd = torch.rand(B, 1) * 10
        with torch.no_grad():
            out = model(img, lid, spd)
        if check_output_health(out, "TransFuser", "B=8"):
            record("B=8", "TransFuser", "PASS")
    except Exception as e:
        record("B=8", "TransFuser", "FAIL", str(e))

    # ST-P3
    try:
        model = make_stp3(device)
        model.eval()
        imgs = torch.randn(B, 4, 6, 3, 64, 128)
        with torch.no_grad():
            out = model(imgs)
        if check_output_health(out, "ST-P3", "B=8"):
            record("B=8", "ST-P3", "PASS")
    except Exception as e:
        record("B=8", "ST-P3", "FAIL", str(e))

    # UniAD
    try:
        model = make_uniad(device)
        model.eval()
        imgs = torch.randn(B, 6, 3, 128, 200)
        with torch.no_grad():
            out = model(imgs)
        if check_output_health(out, "UniAD", "B=8"):
            record("B=8", "UniAD", "PASS")
    except Exception as e:
        record("B=8", "UniAD", "FAIL", str(e))

    # VAD
    try:
        model = make_vad(device)
        model.eval()
        imgs = torch.randn(B, 6, 3, 128, 200)
        with torch.no_grad():
            out = model(imgs)
        if check_output_health(out, "VAD", "B=8"):
            record("B=8", "VAD", "PASS")
    except Exception as e:
        record("B=8", "VAD", "FAIL", str(e))

    # MLP Scorer
    try:
        model = make_mlp_scorer(device)
        model.eval()
        traj = torch.randn(B, 16, 4)
        agents = torch.randn(B, 32, 7)
        mask = torch.ones(B, 32, dtype=torch.bool)
        mapf = torch.randn(B, 64, 5)
        with torch.no_grad():
            out = model(traj, agents, mask, mapf)
        if check_output_health(out, "MLP Scorer", "B=8"):
            record("B=8", "MLP Scorer", "PASS")
    except Exception as e:
        record("B=8", "MLP Scorer", "FAIL", str(e))

    # Transformer Scorer
    try:
        model = make_transformer_scorer(device)
        model.eval()
        traj = torch.randn(B, 16, 4)
        agents = torch.randn(B, 32, 7)
        amask = torch.ones(B, 32, dtype=torch.bool)
        mapf = torch.randn(B, 64, 5)
        mmask = torch.ones(B, 64, dtype=torch.bool)
        with torch.no_grad():
            out = model(traj, agents, amask, mapf, mmask)
        if check_output_health(out, "Transformer Scorer", "B=8"):
            record("B=8", "Transformer Scorer", "PASS")
    except Exception as e:
        record("B=8", "Transformer Scorer", "FAIL", str(e))


# ============================================================
# TEST 3: Single Timestep T=1 (temporal models)
# ============================================================
def test_single_timestep():
    print("\n" + "="*70)
    print("TEST 3: Single Timestep T=1 (temporal models)")
    print("="*70)
    device = 'cpu'

    # ST-P3 with T=1
    try:
        model = make_stp3(device)
        model.eval()
        imgs = torch.randn(2, 1, 6, 3, 64, 128)  # T=1
        with torch.no_grad():
            out = model(imgs)
        if check_output_health(out, "ST-P3 (T=1)", "T=1"):
            record("T=1", "ST-P3", "PASS")
    except Exception as e:
        record("T=1", "ST-P3", "FAIL", str(e))

    # GAIA-1 World Model with T=1 (single frame)
    try:
        _, wm = make_gaia1_models(device)
        wm.eval()
        frames = torch.randint(0, 256, (2, 1, 16))  # T=1
        actions = torch.randn(2, 1, 3)
        with torch.no_grad():
            out = wm(frames, actions)
        if check_output_health(out, "GAIA-1 WorldModel (T=1)", "T=1"):
            record("T=1", "GAIA-1 WorldModel", "PASS")
    except Exception as e:
        record("T=1", "GAIA-1 WorldModel", "FAIL", str(e))


# ============================================================
# TEST 4: Maximum sequence length for GAIA-1
# ============================================================
def test_max_sequence():
    print("\n" + "="*70)
    print("TEST 4: Maximum Sequence Length (GAIA-1 T=max_frames)")
    print("="*70)
    device = 'cpu'

    try:
        _, wm = make_gaia1_models(device)
        wm.eval()
        # max_frames=16, tokens_per_frame=16, so max seq = 16*(16+1)=272
        # pos_embed size = max_frames * (tokens_per_frame + 1) = 16*17 = 272
        # The actual sequence length = T*(N+1) where T=frames, N=tokens_per_frame
        # With T=16, N=16: seq_len = 16*(16+1) = 272 which is exactly pos_embed size
        frames = torch.randint(0, 256, (2, 16, 16))  # max T=16
        actions = torch.randn(2, 16, 3)
        with torch.no_grad():
            out = wm(frames, actions)
        if check_output_health(out, "GAIA-1 WorldModel (T=16 max)", "T=max"):
            record("T=max", "GAIA-1 WorldModel", "PASS")
    except Exception as e:
        record("T=max", "GAIA-1 WorldModel", "FAIL", str(e))

    # Test exceeding max
    try:
        frames = torch.randint(0, 256, (1, 17, 16))  # T=17 > max_frames=16
        actions = torch.randn(1, 17, 3)
        with torch.no_grad():
            out = wm(frames, actions)
        record("T=max+1", "GAIA-1 WorldModel", "WARN", "Exceeding max_frames did not raise error (pos_embed overflow?)")
    except Exception as e:
        record("T=max+1", "GAIA-1 WorldModel", "PASS", f"Correctly errors on overflow: {type(e).__name__}")


# ============================================================
# TEST 5: All-zero input
# ============================================================
def test_all_zero_input():
    print("\n" + "="*70)
    print("TEST 5: All-Zero Input")
    print("="*70)
    device = 'cpu'

    # GAIA-1 Tokenizer
    try:
        tok, wm = make_gaia1_models(device)
        tok.eval(); wm.eval()
        frame = torch.zeros(2, 3, 64, 64)
        with torch.no_grad():
            out = tok(frame)
        if check_output_health(out, "GAIA-1 Tokenizer", "zeros"):
            record("zeros", "GAIA-1 Tokenizer", "PASS")
    except Exception as e:
        record("zeros", "GAIA-1 Tokenizer", "FAIL", str(e))

    # GAIA-1 World Model (zero tokens = index 0, zero actions)
    try:
        frames = torch.zeros(2, 4, 16, dtype=torch.long)
        actions = torch.zeros(2, 4, 3)
        with torch.no_grad():
            out = wm(frames, actions)
        if check_output_health(out, "GAIA-1 WorldModel", "zeros"):
            record("zeros", "GAIA-1 WorldModel", "PASS")
    except Exception as e:
        record("zeros", "GAIA-1 WorldModel", "FAIL", str(e))

    # GenAD
    try:
        model = make_genad(device)
        model.eval()
        imgs = torch.zeros(2, 3, 64, 128)
        with torch.no_grad():
            out = model(imgs, num_samples=4)
        if check_output_health(out, "GenAD", "zeros"):
            record("zeros", "GenAD", "PASS")
    except Exception as e:
        record("zeros", "GenAD", "FAIL", str(e))

    # DriveVLM
    try:
        model = make_drivevlm(device)
        model.eval()
        imgs = torch.zeros(2, 6, 3, 224, 224)
        with torch.no_grad():
            out = model(imgs)
        if check_output_health(out, "DriveVLM", "zeros"):
            record("zeros", "DriveVLM", "PASS")
    except Exception as e:
        record("zeros", "DriveVLM", "FAIL", str(e))

    # InterFuser
    try:
        model = make_interfuser(device)
        model.eval()
        z = torch.zeros(2, 3, 128, 256)
        lid = torch.zeros(2, 2, 128, 128)
        with torch.no_grad():
            out = model(z, z, z, lid)
        if check_output_health(out, "InterFuser", "zeros"):
            record("zeros", "InterFuser", "PASS")
    except Exception as e:
        record("zeros", "InterFuser", "FAIL", str(e))

    # TCP
    try:
        model = make_tcp(device)
        model.eval()
        img = torch.zeros(2, 3, 128, 256)
        lid = torch.zeros(2, 2, 128, 128)
        spd = torch.zeros(2, 1)
        with torch.no_grad():
            out = model(img, lid, spd)
        if check_output_health(out, "TCP", "zeros"):
            record("zeros", "TCP", "PASS")
    except Exception as e:
        record("zeros", "TCP", "FAIL", str(e))

    # TransFuser
    try:
        model = make_transfuser(device)
        model.eval()
        img = torch.zeros(2, 3, 256, 512)
        lid = torch.zeros(2, 2, 256, 256)
        spd = torch.zeros(2, 1)
        with torch.no_grad():
            out = model(img, lid, spd)
        if check_output_health(out, "TransFuser", "zeros"):
            record("zeros", "TransFuser", "PASS")
    except Exception as e:
        record("zeros", "TransFuser", "FAIL", str(e))

    # ST-P3
    try:
        model = make_stp3(device)
        model.eval()
        imgs = torch.zeros(2, 4, 6, 3, 64, 128)
        with torch.no_grad():
            out = model(imgs)
        if check_output_health(out, "ST-P3", "zeros"):
            record("zeros", "ST-P3", "PASS")
    except Exception as e:
        record("zeros", "ST-P3", "FAIL", str(e))

    # UniAD
    try:
        model = make_uniad(device)
        model.eval()
        imgs = torch.zeros(2, 6, 3, 128, 200)
        with torch.no_grad():
            out = model(imgs)
        if check_output_health(out, "UniAD", "zeros"):
            record("zeros", "UniAD", "PASS")
    except Exception as e:
        record("zeros", "UniAD", "FAIL", str(e))

    # VAD
    try:
        model = make_vad(device)
        model.eval()
        imgs = torch.zeros(2, 6, 3, 128, 200)
        with torch.no_grad():
            out = model(imgs)
        if check_output_health(out, "VAD", "zeros"):
            record("zeros", "VAD", "PASS")
    except Exception as e:
        record("zeros", "VAD", "FAIL", str(e))

    # MLP Scorer with zeros
    try:
        model = make_mlp_scorer(device)
        model.eval()
        traj = torch.zeros(2, 16, 4)
        agents = torch.zeros(2, 32, 7)
        mask = torch.ones(2, 32, dtype=torch.bool)
        mapf = torch.zeros(2, 64, 5)
        with torch.no_grad():
            out = model(traj, agents, mask, mapf)
        if check_output_health(out, "MLP Scorer", "zeros"):
            record("zeros", "MLP Scorer", "PASS")
    except Exception as e:
        record("zeros", "MLP Scorer", "FAIL", str(e))

    # Transformer Scorer with zeros
    try:
        model = make_transformer_scorer(device)
        model.eval()
        traj = torch.zeros(2, 16, 4)
        agents = torch.zeros(2, 32, 7)
        amask = torch.ones(2, 32, dtype=torch.bool)
        mapf = torch.zeros(2, 64, 5)
        mmask = torch.ones(2, 64, dtype=torch.bool)
        with torch.no_grad():
            out = model(traj, agents, amask, mapf, mmask)
        if check_output_health(out, "Transformer Scorer", "zeros"):
            record("zeros", "Transformer Scorer", "PASS")
    except Exception as e:
        record("zeros", "Transformer Scorer", "FAIL", str(e))


# ============================================================
# TEST 6: All-same mask (all-True and all-False)
# ============================================================
def test_mask_edge_cases():
    print("\n" + "="*70)
    print("TEST 6: Mask Edge Cases (all-True, all-False)")
    print("="*70)
    device = 'cpu'

    # MLP Scorer - all True mask
    try:
        model = make_mlp_scorer(device)
        model.eval()
        traj = torch.randn(2, 16, 4)
        agents = torch.randn(2, 32, 7)
        mask_all_true = torch.ones(2, 32, dtype=torch.bool)
        mapf = torch.randn(2, 64, 5)
        with torch.no_grad():
            out = model(traj, agents, mask_all_true, mapf)
        if check_output_health(out, "MLP Scorer (all-True mask)", "mask"):
            record("mask", "MLP Scorer (all-True)", "PASS")
    except Exception as e:
        record("mask", "MLP Scorer (all-True)", "FAIL", str(e))

    # MLP Scorer - all False mask
    try:
        mask_all_false = torch.zeros(2, 32, dtype=torch.bool)
        with torch.no_grad():
            out = model(traj, agents, mask_all_false, mapf)
        # With all-False mask, masked_fill sets all to -inf, max gives -inf
        has_nan = torch.isnan(out).any().item()
        has_inf = torch.isinf(out).any().item()
        if has_nan:
            record("mask", "MLP Scorer (all-False)", "WARN", "NaN in output with all-False mask (expected: -inf from max pool)")
        elif has_inf:
            record("mask", "MLP Scorer (all-False)", "WARN", "Inf in output with all-False mask (from -inf max pool propagating)")
        else:
            record("mask", "MLP Scorer (all-False)", "PASS")
    except Exception as e:
        record("mask", "MLP Scorer (all-False)", "FAIL", str(e))

    # Transformer Scorer - all True mask
    try:
        model = make_transformer_scorer(device)
        model.eval()
        traj = torch.randn(2, 16, 4)
        agents = torch.randn(2, 32, 7)
        amask_all_true = torch.ones(2, 32, dtype=torch.bool)
        mapf = torch.randn(2, 64, 5)
        mmask_all_true = torch.ones(2, 64, dtype=torch.bool)
        with torch.no_grad():
            out = model(traj, agents, amask_all_true, mapf, mmask_all_true)
        if check_output_health(out, "Transformer Scorer (all-True)", "mask"):
            record("mask", "Transformer Scorer (all-True)", "PASS")
    except Exception as e:
        record("mask", "Transformer Scorer (all-True)", "FAIL", str(e))

    # Transformer Scorer - all False mask (all tokens masked out)
    try:
        amask_all_false = torch.zeros(2, 32, dtype=torch.bool)
        mmask_all_false = torch.zeros(2, 64, dtype=torch.bool)
        with torch.no_grad():
            out = model(traj, agents, amask_all_false, mapf, mmask_all_false)
        has_nan = torch.isnan(out).any().item()
        has_inf = torch.isinf(out).any().item()
        if has_nan:
            record("mask", "Transformer Scorer (all-False)", "WARN", "NaN with all-False mask (all scene tokens masked -> softmax over -inf)")
        elif has_inf:
            record("mask", "Transformer Scorer (all-False)", "WARN", "Inf with all-False mask")
        else:
            record("mask", "Transformer Scorer (all-False)", "PASS")
    except Exception as e:
        record("mask", "Transformer Scorer (all-False)", "FAIL", str(e))


# ============================================================
# TEST 7: Single candidate (GenAD num_samples=1, VAD K=1)
# ============================================================
def test_single_candidate():
    print("\n" + "="*70)
    print("TEST 7: Single Candidate")
    print("="*70)
    device = 'cpu'

    # GenAD with num_samples=1
    try:
        model = make_genad(device)
        model.eval()
        imgs = torch.randn(2, 3, 64, 128)
        with torch.no_grad():
            out = model(imgs, num_samples=1)
        if check_output_health(out, "GenAD (num_samples=1)", "single_cand"):
            record("single_cand", "GenAD (num_samples=1)", "PASS")
    except Exception as e:
        record("single_cand", "GenAD (num_samples=1)", "FAIL", str(e))

    # VAD with K=1 ego queries
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'two_step_e2e', 'VAD'))
        import importlib
        import model as vad_model
        importlib.reload(vad_model)
        model = vad_model.VAD(embed_dim=128, bev_h=50, bev_w=50,
                              num_cameras=6, num_ego_queries=1, num_waypoints=6).to(device)
        model.eval()
        imgs = torch.randn(2, 6, 3, 128, 200)
        with torch.no_grad():
            out = model(imgs)
        if check_output_health(out, "VAD (K=1)", "single_cand"):
            record("single_cand", "VAD (K=1)", "PASS")
    except Exception as e:
        record("single_cand", "VAD (K=1)", "FAIL", str(e))


# ============================================================
# TEST 8: Very large/small values (numerical stability)
# ============================================================
def test_extreme_values():
    print("\n" + "="*70)
    print("TEST 8: Extreme Input Values (scaled by 100 and 0.001)")
    print("="*70)
    device = 'cpu'

    # --- Scaled by 100 (large values) ---
    print("  --- Large values (scale=100) ---")

    # GAIA-1 Tokenizer
    try:
        tok, wm = make_gaia1_models(device)
        tok.eval(); wm.eval()
        frame = torch.randn(2, 3, 64, 64) * 100
        with torch.no_grad():
            out = tok(frame)
        if check_output_health(out, "GAIA-1 Tokenizer", "large"):
            record("large_vals", "GAIA-1 Tokenizer", "PASS")
    except Exception as e:
        record("large_vals", "GAIA-1 Tokenizer", "FAIL", str(e))

    # GenAD
    try:
        model = make_genad(device)
        model.eval()
        imgs = torch.randn(2, 3, 64, 128) * 100
        with torch.no_grad():
            out = model(imgs, num_samples=4)
        if check_output_health(out, "GenAD", "large"):
            record("large_vals", "GenAD", "PASS")
    except Exception as e:
        record("large_vals", "GenAD", "FAIL", str(e))

    # InterFuser
    try:
        model = make_interfuser(device)
        model.eval()
        f = torch.randn(2, 3, 128, 256) * 100
        lid = torch.randn(2, 2, 128, 128) * 100
        with torch.no_grad():
            out = model(f, f, f, lid)
        if check_output_health(out, "InterFuser", "large"):
            record("large_vals", "InterFuser", "PASS")
    except Exception as e:
        record("large_vals", "InterFuser", "FAIL", str(e))

    # TCP
    try:
        model = make_tcp(device)
        model.eval()
        img = torch.randn(2, 3, 128, 256) * 100
        lid = torch.randn(2, 2, 128, 128) * 100
        spd = torch.tensor([[100.0], [100.0]])
        with torch.no_grad():
            out = model(img, lid, spd)
        if check_output_health(out, "TCP", "large"):
            record("large_vals", "TCP", "PASS")
    except Exception as e:
        record("large_vals", "TCP", "FAIL", str(e))

    # TransFuser
    try:
        model = make_transfuser(device)
        model.eval()
        img = torch.randn(2, 3, 256, 512) * 100
        lid = torch.randn(2, 2, 256, 256) * 100
        spd = torch.tensor([[100.0], [100.0]])
        with torch.no_grad():
            out = model(img, lid, spd)
        if check_output_health(out, "TransFuser", "large"):
            record("large_vals", "TransFuser", "PASS")
    except Exception as e:
        record("large_vals", "TransFuser", "FAIL", str(e))

    # UniAD
    try:
        model = make_uniad(device)
        model.eval()
        imgs = torch.randn(2, 6, 3, 128, 200) * 100
        with torch.no_grad():
            out = model(imgs)
        if check_output_health(out, "UniAD", "large"):
            record("large_vals", "UniAD", "PASS")
    except Exception as e:
        record("large_vals", "UniAD", "FAIL", str(e))

    # VAD
    try:
        model = make_vad(device)
        model.eval()
        imgs = torch.randn(2, 6, 3, 128, 200) * 100
        with torch.no_grad():
            out = model(imgs)
        if check_output_health(out, "VAD", "large"):
            record("large_vals", "VAD", "PASS")
    except Exception as e:
        record("large_vals", "VAD", "FAIL", str(e))

    # MLP Scorer
    try:
        model = make_mlp_scorer(device)
        model.eval()
        traj = torch.randn(2, 16, 4) * 100
        agents = torch.randn(2, 32, 7) * 100
        mask = torch.ones(2, 32, dtype=torch.bool)
        mapf = torch.randn(2, 64, 5) * 100
        with torch.no_grad():
            out = model(traj, agents, mask, mapf)
        if check_output_health(out, "MLP Scorer", "large"):
            record("large_vals", "MLP Scorer", "PASS")
    except Exception as e:
        record("large_vals", "MLP Scorer", "FAIL", str(e))

    # Transformer Scorer
    try:
        model = make_transformer_scorer(device)
        model.eval()
        traj = torch.randn(2, 16, 4) * 100
        agents = torch.randn(2, 32, 7) * 100
        amask = torch.ones(2, 32, dtype=torch.bool)
        mapf = torch.randn(2, 64, 5) * 100
        mmask = torch.ones(2, 64, dtype=torch.bool)
        with torch.no_grad():
            out = model(traj, agents, amask, mapf, mmask)
        if check_output_health(out, "Transformer Scorer", "large"):
            record("large_vals", "Transformer Scorer", "PASS")
    except Exception as e:
        record("large_vals", "Transformer Scorer", "FAIL", str(e))

    # --- Scaled by 0.001 (small values) ---
    print("  --- Small values (scale=0.001) ---")

    # GAIA-1 Tokenizer
    try:
        tok, wm = make_gaia1_models(device)
        tok.eval()
        frame = torch.randn(2, 3, 64, 64) * 0.001
        with torch.no_grad():
            out = tok(frame)
        if check_output_health(out, "GAIA-1 Tokenizer", "small"):
            record("small_vals", "GAIA-1 Tokenizer", "PASS")
    except Exception as e:
        record("small_vals", "GAIA-1 Tokenizer", "FAIL", str(e))

    # TCP
    try:
        model = make_tcp(device)
        model.eval()
        img = torch.randn(2, 3, 128, 256) * 0.001
        lid = torch.randn(2, 2, 128, 128) * 0.001
        spd = torch.tensor([[0.001], [0.001]])
        with torch.no_grad():
            out = model(img, lid, spd)
        if check_output_health(out, "TCP", "small"):
            record("small_vals", "TCP", "PASS")
    except Exception as e:
        record("small_vals", "TCP", "FAIL", str(e))

    # TransFuser
    try:
        model = make_transfuser(device)
        model.eval()
        img = torch.randn(2, 3, 256, 512) * 0.001
        lid = torch.randn(2, 2, 256, 256) * 0.001
        spd = torch.tensor([[0.001], [0.001]])
        with torch.no_grad():
            out = model(img, lid, spd)
        if check_output_health(out, "TransFuser", "small"):
            record("small_vals", "TransFuser", "PASS")
    except Exception as e:
        record("small_vals", "TransFuser", "FAIL", str(e))

    # MLP Scorer
    try:
        model = make_mlp_scorer(device)
        model.eval()
        traj = torch.randn(2, 16, 4) * 0.001
        agents = torch.randn(2, 32, 7) * 0.001
        mask = torch.ones(2, 32, dtype=torch.bool)
        mapf = torch.randn(2, 64, 5) * 0.001
        with torch.no_grad():
            out = model(traj, agents, mask, mapf)
        if check_output_health(out, "MLP Scorer", "small"):
            record("small_vals", "MLP Scorer", "PASS")
    except Exception as e:
        record("small_vals", "MLP Scorer", "FAIL", str(e))


# ============================================================
# TEST 9: Half Precision (fp16)
# ============================================================
def test_half_precision():
    print("\n" + "="*70)
    print("TEST 9: Half Precision (float16)")
    print("="*70)
    device = 'cpu'  # fp16 on CPU is limited but tests shape compat

    # GAIA-1 Tokenizer
    try:
        tok, wm = make_gaia1_models(device)
        tok = tok.half().eval()
        frame = torch.randn(2, 3, 64, 64).half()
        with torch.no_grad():
            out = tok(frame)
        if check_output_health(out, "GAIA-1 Tokenizer", "fp16"):
            record("fp16", "GAIA-1 Tokenizer", "PASS")
    except Exception as e:
        record("fp16", "GAIA-1 Tokenizer", "FAIL", str(e))

    # GAIA-1 World Model
    try:
        _, wm = make_gaia1_models(device)
        wm = wm.half().eval()
        frames = torch.randint(0, 256, (2, 4, 16))
        actions = torch.randn(2, 4, 3).half()
        with torch.no_grad():
            out = wm(frames, actions)
        if check_output_health(out, "GAIA-1 WorldModel", "fp16"):
            record("fp16", "GAIA-1 WorldModel", "PASS")
    except Exception as e:
        record("fp16", "GAIA-1 WorldModel", "FAIL", str(e))

    # GenAD (training mode to avoid diffusion sampling complexity in fp16)
    try:
        model = make_genad(device)
        model = model.half()
        model.train()  # skip diffusion sampling
        imgs = torch.randn(2, 3, 64, 128).half()
        with torch.no_grad():
            out = model(imgs)
        if check_output_health(out, "GenAD", "fp16"):
            record("fp16", "GenAD (train mode)", "PASS")
    except Exception as e:
        record("fp16", "GenAD (train mode)", "FAIL", str(e))

    # DriveVLM
    try:
        model = make_drivevlm(device)
        model = model.half().eval()
        imgs = torch.randn(1, 6, 3, 224, 224).half()
        with torch.no_grad():
            out = model(imgs)
        if check_output_health(out, "DriveVLM", "fp16"):
            record("fp16", "DriveVLM", "PASS")
    except Exception as e:
        record("fp16", "DriveVLM", "FAIL", str(e))

    # InterFuser
    try:
        model = make_interfuser(device)
        model = model.half().eval()
        f = torch.randn(2, 3, 128, 256).half()
        lid = torch.randn(2, 2, 128, 128).half()
        with torch.no_grad():
            out = model(f, f, f, lid)
        if check_output_health(out, "InterFuser", "fp16"):
            record("fp16", "InterFuser", "PASS")
    except Exception as e:
        record("fp16", "InterFuser", "FAIL", str(e))

    # TCP
    try:
        model = make_tcp(device)
        model = model.half().eval()
        img = torch.randn(2, 3, 128, 256).half()
        lid = torch.randn(2, 2, 128, 128).half()
        spd = torch.tensor([[5.0], [5.0]]).half()
        with torch.no_grad():
            out = model(img, lid, spd)
        if check_output_health(out, "TCP", "fp16"):
            record("fp16", "TCP", "PASS")
    except Exception as e:
        record("fp16", "TCP", "FAIL", str(e))

    # TransFuser
    try:
        model = make_transfuser(device)
        model = model.half().eval()
        img = torch.randn(2, 3, 256, 512).half()
        lid = torch.randn(2, 2, 256, 256).half()
        spd = torch.tensor([[5.0], [5.0]]).half()
        with torch.no_grad():
            out = model(img, lid, spd)
        if check_output_health(out, "TransFuser", "fp16"):
            record("fp16", "TransFuser", "PASS")
    except Exception as e:
        record("fp16", "TransFuser", "FAIL", str(e))

    # ST-P3
    try:
        model = make_stp3(device)
        model = model.half().eval()
        imgs = torch.randn(1, 4, 6, 3, 64, 128).half()
        with torch.no_grad():
            out = model(imgs)
        if check_output_health(out, "ST-P3", "fp16"):
            record("fp16", "ST-P3", "PASS")
    except Exception as e:
        record("fp16", "ST-P3", "FAIL", str(e))

    # UniAD
    try:
        model = make_uniad(device)
        model = model.half().eval()
        imgs = torch.randn(1, 6, 3, 128, 200).half()
        with torch.no_grad():
            out = model(imgs)
        if check_output_health(out, "UniAD", "fp16"):
            record("fp16", "UniAD", "PASS")
    except Exception as e:
        record("fp16", "UniAD", "FAIL", str(e))

    # VAD
    try:
        model = make_vad(device)
        model = model.half().eval()
        imgs = torch.randn(1, 6, 3, 128, 200).half()
        with torch.no_grad():
            out = model(imgs)
        if check_output_health(out, "VAD", "fp16"):
            record("fp16", "VAD", "PASS")
    except Exception as e:
        record("fp16", "VAD", "FAIL", str(e))

    # MLP Scorer
    try:
        model = make_mlp_scorer(device)
        model = model.half().eval()
        traj = torch.randn(2, 16, 4).half()
        agents = torch.randn(2, 32, 7).half()
        mask = torch.ones(2, 32, dtype=torch.bool)
        mapf = torch.randn(2, 64, 5).half()
        with torch.no_grad():
            out = model(traj, agents, mask, mapf)
        if check_output_health(out, "MLP Scorer", "fp16"):
            record("fp16", "MLP Scorer", "PASS")
    except Exception as e:
        record("fp16", "MLP Scorer", "FAIL", str(e))

    # Transformer Scorer
    try:
        model = make_transformer_scorer(device)
        model = model.half().eval()
        traj = torch.randn(2, 16, 4).half()
        agents = torch.randn(2, 32, 7).half()
        amask = torch.ones(2, 32, dtype=torch.bool)
        mapf = torch.randn(2, 64, 5).half()
        mmask = torch.ones(2, 64, dtype=torch.bool)
        with torch.no_grad():
            out = model(traj, agents, amask, mapf, mmask)
        if check_output_health(out, "Transformer Scorer", "fp16"):
            record("fp16", "Transformer Scorer", "PASS")
    except Exception as e:
        record("fp16", "Transformer Scorer", "FAIL", str(e))


# ============================================================
# RUN ALL TESTS
# ============================================================
if __name__ == '__main__':
    print("="*70)
    print("COMPREHENSIVE EDGE CASE TESTS - 13 E2E PERCEPTION-PLANNING MODELS")
    print("="*70)

    test_batch_size_1()
    test_large_batch()
    test_single_timestep()
    test_max_sequence()
    test_all_zero_input()
    test_mask_edge_cases()
    test_single_candidate()
    test_extreme_values()
    test_half_precision()

    # ============================================================
    # FINAL SUMMARY
    # ============================================================
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)

    pass_count = sum(1 for r in results if r[2] == "PASS")
    warn_count = sum(1 for r in results if r[2] == "WARN")
    fail_count = sum(1 for r in results if r[2] == "FAIL")

    print(f"\nTotal tests: {len(results)}")
    print(f"  PASS: {pass_count}")
    print(f"  WARN: {warn_count}")
    print(f"  FAIL: {fail_count}")

    if warn_count > 0:
        print(f"\n--- WARNINGS ---")
        for test, model, status, detail in results:
            if status == "WARN":
                print(f"  [{test}] {model}: {detail}")

    if fail_count > 0:
        print(f"\n--- FAILURES ---")
        for test, model, status, detail in results:
            if status == "FAIL":
                print(f"  [{test}] {model}: {detail}")

    # Per-model summary
    print(f"\n--- PER-MODEL ROBUSTNESS ---")
    model_names = set(r[1] for r in results)
    for mn in sorted(model_names):
        mr = [r for r in results if r[1] == mn]
        mp = sum(1 for r in mr if r[2] == "PASS")
        mw = sum(1 for r in mr if r[2] == "WARN")
        mf = sum(1 for r in mr if r[2] == "FAIL")
        status = "ROBUST" if mf == 0 and mw == 0 else ("HAS WARNINGS" if mf == 0 else "HAS FAILURES")
        print(f"  {mn}: {mp}P/{mw}W/{mf}F - {status}")
