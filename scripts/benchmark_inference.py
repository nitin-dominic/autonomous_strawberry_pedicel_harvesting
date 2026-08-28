"""
Jetson Orin Inference Speed Benchmark
======================================
Tests FP32, FP16, and torch.compile inference speeds.
Run: python benchmark_inference.py

Edit MODEL_PATH below before running.
"""

import torch
import time
import sys
import os

# ============================================================
# CONFIGURATION — Edit this
# ============================================================
MODEL_PATH = "/media/precag/ARM101/models/pretrained_model_resnet50"
IMG_H      = 640    # must match your training resolution
IMG_W      = 480
N_WARMUP   = 5      # warmup runs before timing
N_BENCH    = 30     # timing runs
# ============================================================

# Add lerobot to path
sys.path.insert(0, os.path.expanduser("~/lerobot/src"))

def separator(title):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")

def benchmark_backbone(backbone, dummy_input, label, n_warmup=5, n_bench=30):
    """Time the backbone forward pass."""
    # Warmup
    for _ in range(n_warmup):
        with torch.inference_mode():
            _ = backbone(dummy_input)
    torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(n_bench):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            _ = backbone(dummy_input)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    mean_ms = sum(times) / len(times) * 1000
    min_ms  = min(times) * 1000
    max_hz  = 1000 / mean_ms

    print(f"  {label:<25} {mean_ms:6.1f}ms avg  |  {min_ms:5.1f}ms min  |  {max_hz:5.0f}Hz theoretical")
    return mean_ms, max_hz


separator("ENVIRONMENT CHECK")

# Check CUDA
if not torch.cuda.is_available():
    print("ERROR: CUDA not available")
    print("Make sure you built PyTorch with CUDA support")
    sys.exit(1)

print(f"  PyTorch     : {torch.__version__}")
print(f"  CUDA        : {torch.version.cuda}")
print(f"  GPU         : {torch.cuda.get_device_name(0)}")
print(f"  Memory      : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"  Model path  : {MODEL_PATH}")
print(f"  Image size  : {IMG_H}x{IMG_W}")


separator("LOADING MODEL")

try:
    from lerobot.policies.act.modeling_act import ACTPolicy
    print("  Loading ACTPolicy...")
    policy_fp32 = ACTPolicy.from_pretrained(MODEL_PATH)
    policy_fp32.eval().cuda()
    backbone_fp32 = policy_fp32.model.backbone
    print(f"  Model loaded successfully")
    print(f"  Backbone type: {type(backbone_fp32).__name__}")
except Exception as e:
    print(f"  ERROR loading model: {e}")
    print(f"  Check MODEL_PATH is correct: {MODEL_PATH}")
    sys.exit(1)


separator("BENCHMARKING")

dummy_fp32 = torch.randn(1, 3, IMG_H, IMG_W).cuda()
results    = {}

# ── Test 1: FP32 baseline ───────────────────────────────────
print("\n  Running FP32 baseline...")
ms, hz = benchmark_backbone(backbone_fp32, dummy_fp32, "FP32 (baseline)")
results["fp32"] = {"ms": ms, "hz": hz}


# ── Test 2: FP16 ────────────────────────────────────────────
print("\n  Running FP16...")
try:
    policy_fp16  = ACTPolicy.from_pretrained(MODEL_PATH)
    backbone_fp16 = policy_fp16.model.backbone.eval().cuda().half()
    dummy_fp16    = dummy_fp32.half()
    ms, hz = benchmark_backbone(backbone_fp16, dummy_fp16, "FP16")
    results["fp16"] = {"ms": ms, "hz": hz}
    print(f"  Speedup vs FP32: {results['fp32']['ms']/ms:.2f}x")
except Exception as e:
    print(f"  FP16 failed: {e}")
    results["fp16"] = None


# ── Test 3: torch.compile (reduce-overhead mode) ────────────
print("\n  Running torch.compile (reduce-overhead)...")
print("  First call takes 30-60s to compile — please wait...")
try:
    policy_compiled  = ACTPolicy.from_pretrained(MODEL_PATH)
    backbone_raw     = policy_compiled.model.backbone.eval().cuda()
    backbone_compiled = torch.compile(
        backbone_raw,
        mode="reduce-overhead",
        fullgraph=False
    )
    # Trigger compilation (first call is slow)
    t_compile_start = time.time()
    with torch.inference_mode():
        _ = backbone_compiled(dummy_fp32)
    torch.cuda.synchronize()
    print(f"  Compilation took: {time.time()-t_compile_start:.1f}s")

    ms, hz = benchmark_backbone(
        backbone_compiled, dummy_fp32, "torch.compile",
        n_warmup=3
    )
    results["compile"] = {"ms": ms, "hz": hz}
    print(f"  Speedup vs FP32: {results['fp32']['ms']/ms:.2f}x")
except Exception as e:
    print(f"  torch.compile failed: {e}")
    results["compile"] = None


# ── Test 4: torch.compile + FP16 ────────────────────────────
print("\n  Running torch.compile + FP16...")
try:
    policy_c16   = ACTPolicy.from_pretrained(MODEL_PATH)
    backbone_c16 = policy_c16.model.backbone.eval().cuda().half()
    backbone_c16 = torch.compile(
        backbone_c16,
        mode="reduce-overhead",
        fullgraph=False
    )
    t_compile_start = time.time()
    with torch.inference_mode():
        _ = backbone_c16(dummy_fp16)
    torch.cuda.synchronize()
    print(f"  Compilation took: {time.time()-t_compile_start:.1f}s")

    ms, hz = benchmark_backbone(
        backbone_c16, dummy_fp16, "compile + FP16",
        n_warmup=3
    )
    results["compile_fp16"] = {"ms": ms, "hz": hz}
    print(f"  Speedup vs FP32: {results['fp32']['ms']/ms:.2f}x")
except Exception as e:
    print(f"  compile+FP16 failed: {e}")
    results["compile_fp16"] = None


# ── Summary ─────────────────────────────────────────────────
separator("RESULTS SUMMARY")

print(f"  {'Method':<25} {'Speed':>10}  {'Hz':>8}  {'vs Baseline':>12}")
print(f"  {'-'*60}")

baseline_ms = results["fp32"]["ms"]
options = [
    ("fp32",         "FP32 (baseline)"),
    ("fp16",         "FP16"),
    ("compile",      "torch.compile"),
    ("compile_fp16", "compile + FP16"),
]

best_hz     = 0
best_method = "fp32"

for key, label in options:
    r = results.get(key)
    if r is None:
        print(f"  {label:<25} {'FAILED':>10}")
        continue
    speedup = baseline_ms / r["ms"]
    marker  = " ← BEST" if r["hz"] > best_hz else ""
    if r["hz"] > best_hz:
        best_hz     = r["hz"]
        best_method = key
    print(f"  {label:<25} {r['ms']:>8.1f}ms  {r['hz']:>6.0f}Hz  {speedup:>10.2f}x{marker}")


separator("RECOMMENDATION")

r = results.get(best_method)
if r:
    print(f"  Best method : {best_method}")
    print(f"  Speed       : {r['ms']:.1f}ms  → {r['hz']:.0f}Hz theoretical")
    print()
    if best_method == "fp32":
        print("  No speedup found. Your GPU may be memory-bandwidth limited.")
        print("  Try reducing camera resolution or use HiPerGator policy server.")

    elif best_method == "fp16":
        print("  Use FP16 for inference. To enable in rollout, the policy")
        print("  needs to be loaded in FP16 mode. Contact for a wrapper script.")

    elif "compile" in best_method:
        print("  torch.compile gives the best speed.")
        print("  NOTE: Compilation happens once at startup (~30-60s delay).")
        print("  After that, every inference uses the compiled kernel.")

print()
print(f"  Current deployment Hz : {results['fp32']['hz']:.0f}Hz (FP32)")
print(f"  With best method      : {best_hz:.0f}Hz")
print(f"  Task requirement      : 15Hz sufficient for strawberry harvesting")
print()
if best_hz >= 20:
    print("  ✅ Improvement is meaningful — worth applying")
elif best_hz > results["fp32"]["hz"] * 1.15:
    print("  ✅ Modest improvement — apply if stability allows")
else:
    print("  → Stick with --fps=15 flag in rollout command")
    print("  → Hardware ceiling reached on this JetPack version")
    print("  → Best long-term fix: flash to JetPack 6.2")

separator("DONE")
