#!/usr/bin/env python3
"""Verify the multi-image fix works."""
import os, sys, torch
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, "/home/hurricane/VLA/openvla-oft")
sys.path.insert(0, "/home/hurricane/VLA/openvla-oft/LIBERO")

import numpy as np
from PIL import Image
from experiments.robot.openvla_utils import get_vla, get_processor, prepare_images_for_vla

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
print(f"num_images_in_input: {vla.vision_backbone.num_images_in_input}")

# Simulate observation with 2 images
img1 = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
img2 = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
obs = {
    "full_image": img1,
    "wrist_image": img2,
}

# Test prepare_images_for_vla
all_imgs = [obs["full_image"]]
all_imgs.extend([obs[k] for k in obs.keys() if "wrist" in k])
processed = prepare_images_for_vla(all_imgs, cfg)
print(f"prepare_images_for_vla output: {len(processed)} PIL images")
all_inputs = []
for img in processed:
    single = proc("test", img)
    all_inputs.append(single)
    print(f"  Single image pixel_values: {single['pixel_values'].shape}")
concat_pv = torch.cat([inp["pixel_values"] for inp in all_inputs], dim=1)
print(f"Concatenated pixel_values: {concat_pv.shape}")
print(f"Expected shape: [1, 12, 224, 224] for num_images_in_input=2")

# Now test the fixed get_vla_action
task_label = "pick up the black bowl"
print("\nTesting predict_action with 2 images...")
try:
    action, _ = vla.predict_action(
        input_ids=all_inputs[0]["input_ids"].cuda(),
        pixel_values=concat_pv.to(device="cuda", dtype=torch.bfloat16),
        attention_mask=all_inputs[0]["attention_mask"].cuda(),
        unnorm_key="libero_spatial_no_noops",
    )
    print(f"SUCCESS! Action shape: {action.shape}")
    print(f"Action (first 3 dims): {action[0][:3]}")
    print(f"Action range: [{action.min():.4f}, {action.max():.4f}]")
except Exception as e:
    import traceback
    print(f"FAILED: {str(e)[:300]}")
    traceback.print_exc()
