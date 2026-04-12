#!/usr/bin/env python3
import json
with open('/home/hurricane/VLA/modified_libero_rlds/libero_spatial_no_noops/1.0.0/features.json') as f:
    d = json.load(f)
# Navigate nested structure
d = d.get('featuresDict', d).get('features', d)
d = d.get('steps', {}).get('sequence', {})
d = d.get('feature', d).get('featuresDict', d)
d = d.get('features', d)
obs = d.get('observation', {}).get('featuresDict', {}).get('features', {})
print("Observation features:", list(obs.keys()))
