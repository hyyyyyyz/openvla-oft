#!/usr/bin/env python3
"""Quick: check processor output structure."""
import os, sys, torch
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, "/home/hurricane/VLA/openvla-oft")
sys.path.insert(0, "/home/hurricane/VLA/openvla-oft/LIBERO")

import numpy as np
from PIL import Image
from transformers import AutoProcessor

proc = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
prompt = "In: What action should the robot take?"
result = proc(prompt, img)
print("Processor result type:", type(result))
print("Result keys:", result.keys() if hasattr(result, 'keys') else 'N/A')
print("pixel_values shape:", result["pixel_values"].shape)
print("attention_mask type:", type(result.get("attention_mask", "NOT FOUND")))
am = result.get("attention_mask", None)
if am is not None:
    print("attention_mask shape:", am.shape if hasattr(am, 'shape') else am)
    print("attention_mask value:", am)
