"""
Convert ACT policy to FP16 using safetensors directly.
Bypasses save_pretrained which requires distributed support.
Run: python convert_to_fp16.py
"""

import torch
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")
from lerobot.policies.act.modeling_act import ACTPolicy
from safetensors.torch import save_file, load_file

MODEL_PATH = "/media/precag/ARM101/models/pretrained_model_resnet50"
FP16_PATH  = "/media/precag/ARM101/models/pretrained_model_resnet50_fp16"

# ── Step 1: Load FP32 model ──────────────────────────────────
print("Loading FP32 model...")
policy = ACTPolicy.from_pretrained(MODEL_PATH, local_files_only=True)
policy.eval()
print(f"  Loaded — param dtype: {next(policy.parameters()).dtype}")

# ── Step 2: Convert weights to FP16 ─────────────────────────
print("Converting weights to FP16...")
state_dict_fp16 = {}
for key, tensor in policy.state_dict().items():
    if tensor.is_floating_point():
        state_dict_fp16[key] = tensor.half()
    else:
        state_dict_fp16[key] = tensor   # keep int/bool as-is
print(f"  Converted {len(state_dict_fp16)} tensors")

# ── Step 3: Save FP16 weights ────────────────────────────────
Path(FP16_PATH).mkdir(parents=True, exist_ok=True)
weights_path = Path(FP16_PATH) / "model.safetensors"
save_file(state_dict_fp16, str(weights_path))
print(f"  Saved weights: {weights_path}")
print(f"  File size    : {weights_path.stat().st_size / 1e6:.1f} MB")

# ── Step 4: Copy all config files ────────────────────────────
print("Copying config files...")
for fname in Path(MODEL_PATH).iterdir():
    if fname.suffix in [".json", ".yaml", ".txt"] or fname.name == "model.safetensors":
        dst = Path(FP16_PATH) / fname.name
        if fname.name != "model.safetensors":  # skip FP32 weights
            shutil.copy2(fname, dst)
            print(f"  Copied: {fname.name}")

# ── Step 5: Verify FP16 model loads and runs ─────────────────
print("\nVerifying FP16 model...")
policy_fp16 = ACTPolicy.from_pretrained(FP16_PATH, local_files_only=True)
policy_fp16.eval().cuda().half()

param = next(policy_fp16.parameters())
print(f"  dtype  : {param.dtype}")
print(f"  device : {param.device}")

# Benchmark backbone speed
dummy = torch.randn(1, 3, 640, 480, dtype=torch.float16).cuda()
times = []
for _ in range(5):   # warmup
    with torch.inference_mode():
        _ = policy_fp16.model.backbone(dummy)
torch.cuda.synchronize()
for _ in range(20):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        _ = policy_fp16.model.backbone(dummy)
    torch.cuda.synchronize()
    times.append(time.perf_counter() - t0)

ms = sum(times) / len(times) * 1000
print(f"  Backbone: {ms:.1f}ms → {1000/ms:.0f}Hz")
print(f"\n✅ FP16 model ready at: {FP16_PATH}")
print(f"\nRun rollout with:")
print(f"  --policy.path={FP16_PATH}")
