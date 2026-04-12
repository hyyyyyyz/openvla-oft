#!/usr/bin/env python3
"""Check RLDS features for wrist image."""
import json

with open("/home/hurricane/VLA/modified_libero_rlds/libero_spatial_no_noops/1.0.0/features.json") as f:
    features = json.load(f)
print("Top-level features:")
for k in features.keys():
    print(f"  {k}: {features[k]['type']}")
