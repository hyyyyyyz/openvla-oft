#!/usr/bin/env python3
"""Quick test: verify model produces reasonable actions."""
import os, torch, numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTHONPATH"] = "/home/hurricane/VLA/openvla-oft:/home/hurricane/VLA/openvla-oft/LIBERO"

from experiments.robot.openvla_utils import get_vla, get_processor

class Cfg:
    pretrained_checkpoint = "./checkpoints/arm_d_official/final"
    num_images_in_input = 2
    lora_rank = 32
    load_in_8bit = False
    load_in_4bit = False
    use_film = False
    use_l1_regression = True
    use_diffusion = False
    model_family = "openvla"
    use_proprio = True
    center_crop = True
    unnorm_key = "libero_spatial_no_noops"

cfg = Cfg()
print("Loading model...")
vla = get_vla(cfg)
proc = get_processor(cfg)
print("Model loaded!")

# Check norm_stats on both levels
bm = vla.base_model.model
print("Base model.norm_stats keys:", list(bm.norm_stats.keys())[:2], "...")
print("libero_spatial_no_noops in base model:", "libero_spatial_no_noops" in bm.norm_stats)

# Get action bounds
unnorm_key = "libero_spatial_no_noops"
action_stats = bm.norm_stats[unnorm_key]["action"]
print("\nAction stats (first 3 dims):")
for i in range(3):
    print(f"  dim {i}: q01={action_stats['q01'][i]:.3f}, q99={action_stats['q99'][i]:.3f}")

from PIL import Image
# Quick forward pass with a random image
print("\nRunning inference...")
img_pil = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
img_np = np.array(img_pil)
prompt = "In: What action should the robot take to pick up the black bowl?\nOut:"
inputs = proc(prompt, img_pil).to("cuda:0", dtype=torch.bfloat16)

print(f"pixel_values shape: {inputs['pixel_values'].shape}")

action, _ = vla.predict_action(**inputs, unnorm_key=unnorm_key)
print(f"\nAction shape: {action.shape}")
print(f"Action values (first 3 dims): {action[:3]}")
print(f"Action range: [{action.min():.4f}, {action.max():.4f}]")
print(f"Action std: {action.std():.4f}")

# Check if action is reasonable
action_range_ok = (action >= action_stats['q01']).all() and (action <= action_stats['q99']).all()
print(f"\nAction within q01-q99 bounds: {action_range_ok}")

# Compare to using base model WITHOUT LoRA
print("\n--- Comparing with base model (no LoRA) ---")
from transformers import AutoModelForVision2Seq, AutoProcessor
base_vla = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b", torch_dtype=torch.bfloat16, trust_remote_code=True
)
base_vla.eval()
base_vla = base_vla.to("cuda:0", dtype=torch.bfloat16)
base_vla.base_model.model.norm_stats = bm.norm_stats  # inject stats

# Check action dim
base_action_dim = base_vla.base_model.model.get_action_dim(unnorm_key)
print(f"Base model action dim: {base_action_dim}")

# The base model expects single image
inputs_single = proc(prompt, img_pil).to("cuda:0", dtype=torch.bfloat16)
base_action, _ = base_vla.predict_action(**inputs_single, unnorm_key=unnorm_key)
print(f"Base model action (first 3 dims): {base_action[:3]}")
print(f"Base model action range: [{base_action.min():.4f}, {base_action.max():.4f}]")

print("\nDone!")
