#!/usr/bin/env python3
"""Debug: compare base vs checkpoint vision backbone."""
import os, sys, torch
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, "/home/hurricane/VLA/openvla-oft")
sys.path.insert(0, "/home/hurricane/VLA/openvla-oft/LIBERO")

from transformers import AutoModelForVision2Seq, AutoProcessor
import numpy as np
from PIL import Image

# === BASE MODEL (from HuggingFace hub) ===
print("=" * 60)
print("BASE MODEL (openvla/openvla-7b)")
base_vla = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
base_vla = base_vla.cuda().to(torch.bfloat16)
base_vla.eval()
print(f"  num_images_in_input: {base_vla.vision_backbone.num_images_in_input}")
print(f"  use_fused: {base_vla.vision_backbone.use_fused_vision_backbone}")
pw = base_vla.vision_backbone.featurizer.patch_embed.proj.weight
print(f"  featurizer patch_embed.proj: {pw.shape}")
pw_fused = base_vla.vision_backbone.fused_featurizer.patch_embed.proj.weight
print(f"  fused_featurizer patch_embed.proj: {pw_fused.shape}")

# Processor output for single image
proc = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
prompt = "In: What action should the robot take?"
inputs_single = proc(prompt, img)
pv1 = inputs_single["pixel_values"]
print(f"  Processor (1 img): {pv1.shape}")

# Processor output for two images
proc2 = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
img2 = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
inputs_two = proc2(prompt, [img, img2])
pv2 = inputs_two["pixel_values"]
print(f"  Processor (2 imgs): {pv2.shape}")

# Set num_images_in_input=2 on base model
base_vla.vision_backbone.set_num_images_in_input(2)
pw2 = base_vla.vision_backbone.featurizer.patch_embed.proj.weight
print(f"  After set_num_images_in_input(2) featurizer patch_embed: {pw2.shape}")

# === CHECKPOINT MODEL ===
print("=" * 60)
print("CHECKPOINT MODEL (arm_d_official/final)")
vla = AutoModelForVision2Seq.from_pretrained(
    "./checkpoints/arm_d_official/final",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
vla = vla.cuda().to(torch.bfloat16)
vla.eval()
print(f"  num_images_in_input: {vla.vision_backbone.num_images_in_input}")
print(f"  use_fused: {vla.vision_backbone.use_fused_vision_backbone}")
pw = vla.vision_backbone.featurizer.patch_embed.proj.weight
print(f"  featurizer patch_embed.proj: {pw.shape}")
pw_fused = vla.vision_backbone.fused_featurizer.patch_embed.proj.weight
print(f"  fused_featurizer patch_embed.proj: {pw_fused.shape}")

# Set num_images_in_input=2
vla.vision_backbone.set_num_images_in_input(2)
pw2 = vla.vision_backbone.featurizer.patch_embed.proj.weight
print(f"  After set_num_images_in_input(2) featurizer patch_embed: {pw2.shape}")

# Processor output for this model
proc3 = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
inputs_three = proc3(prompt, [img, img2])
pv3 = inputs_three["pixel_values"]
print(f"  Processor (2 imgs): {pv3.shape}")

# Try 12-channel input with checkpoint model
x12 = torch.randn(1, 12, 224, 224).cuda().to(torch.bfloat16)
print(f"  Testing 12ch input...")
try:
    out = vla.vision_backbone(x12)
    print(f"  SUCCESS: output shape = {out.shape}")
except Exception as e:
    print(f"  FAILED: {str(e)[:150]}")
