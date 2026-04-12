#!/usr/bin/env python3
"""Debug: what does predict_action do with actual processor output?"""
import os, sys, torch
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, "/home/hurricane/VLA/openvla-oft")
sys.path.insert(0, "/home/hurricane/VLA/openvla-oft/LIBERO")

from transformers import AutoModelForVision2Seq, AutoProcessor
import numpy as np
from PIL import Image

vla = AutoModelForVision2Seq.from_pretrained(
    "./checkpoints/arm_d_official/final",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
vla = vla.cuda().to(torch.bfloat16)
vla.eval()

proc = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
prompt = "In: What action should the robot take?"
inputs = proc(prompt, img).to("cuda:0", dtype=torch.bfloat16)
pv = inputs["pixel_values"]
print(f"Processor pixel_values shape: {pv.shape}")

# Test with num_images_in_input=2
vla.vision_backbone.set_num_images_in_input(2)
print(f"num_images_in_input: {vla.vision_backbone.num_images_in_input}")

# Show what the vision backbone expects
x = pv  # [1, 3, 224, 224]
print(f"Actual input shape: {x.shape}")
print(f"Split [6, 6] requires input dim=12, got: {x.shape[1]}")

# Try anyway
try:
    action, _ = vla.predict_action(**inputs, unnorm_key="libero_spatial_no_noops")
    print(f"predict_action SUCCESS: shape={action.shape}, values={action[:3]}")
except Exception as e:
    print(f"predict_action FAILED: {str(e)[:200]}")

# Now test num_images_in_input=1
vla.vision_backbone.set_num_images_in_input(1)
try:
    action1, _ = vla.predict_action(**inputs, unnorm_key="libero_spatial_no_noops")
    print(f"predict_action num=1 SUCCESS: shape={action1.shape}")
except Exception as e:
    print(f"predict_action num=1 FAILED: {str(e)[:200]}")
