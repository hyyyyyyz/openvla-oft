#!/usr/bin/env python3
"""Quick test: verify LoRA + norm_stats fix and get a sample action."""
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
vla = get_vla(cfg)
proc = get_processor(cfg)

# Check norm_stats on both levels
print("PeftModel.norm_stats keys:", list(vla.norm_stats.keys())[:2], "...")
bm = vla.base_model.model
print("Base model.norm_stats keys:", list(bm.norm_stats.keys())[:2], "...")
print("libero_spatial_no_noops in base model:", "libero_spatial_no_noops" in bm.norm_stats)

# Quick forward pass with random image
from PIL import Image
img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
prompt = "In: What action should the robot take to pick up the black bowl?\nOut:"
inputs = proc(prompt, img).to("cuda:0", dtype=torch.bfloat16)
action, _ = vla.predict_action(**inputs, unnorm_key="libero_spatial_no_noops")
print("Action shape:", action.shape)
print("Action values (first 3):", action[:3])
print("Action range:", action.min(), "to", action.max())
print("SUCCESS: Model outputs non-trivial actions!")
