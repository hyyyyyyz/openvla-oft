"""
generate_pseudo_depth.py

Generate pseudo depth maps from RGB images using Depth Anything V2.
This creates depth data for the RGB-D teacher (Arm D) without requiring
real depth from simulation.
"""

import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
import torch
import cv2
from PIL import Image
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))


def load_depth_anything_model(model_size="small"):
    """Load Depth Anything V2 model."""
    try:
        from depth_anything_v2.dpt import DepthAnythingV2

        # Model configurations
        model_configs = {
            "small": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
            "base": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
            "large": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        }

        config = model_configs[model_size]
        model = DepthAnythingV2(**config)

        # Load pretrained weights
        checkpoint_path = f"depth_anything_v2_{model_size}.pth"
        if not os.path.exists(checkpoint_path):
            print(f"Downloading Depth Anything V2 {model_size} model...")
            import urllib.request
            url = f"https://huggingface.co/depth-anything/Depth-Anything-V2-{model_size}/resolve/main/depth_anything_v2_{model_size}.pth"
            urllib.request.urlretrieve(url, checkpoint_path)

        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        model = model.cuda().eval()

        return model
    except ImportError:
        print("Depth Anything V2 not installed. Using alternative approach...")
        return None


def estimate_depth_simple(rgb_image):
    """Simple depth estimation using grayscale as fallback."""
    # Convert to grayscale and use as pseudo depth
    if len(rgb_image.shape) == 3:
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    else:
        gray = rgb_image

    # Normalize to 0-255
    depth = gray.astype(np.float32)
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8) * 255
    return depth.astype(np.uint8)


def process_tfrecord_with_depth(input_path, output_path, depth_model=None):
    """Process a TFRecord file and add depth images."""
    # Read input TFRecord
    raw_dataset = tf.data.TFRecordDataset(input_path, compression_type="GZIP")

    # Create writer
    writer = tf.io.TFRecordWriter(output_path)

    for raw_record in tqdm(raw_dataset, desc=f"Processing {os.path.basename(input_path)}"):
        # Parse example
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())

        # Get RGB image
        features = example.features.feature

        # Check if image exists
        if "observation/image" in features:
            # Decode image
            img_bytes = features["observation/image"].bytes_list.value[0]
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Generate depth
            if depth_model is not None:
                # Use Depth Anything V2
                with torch.no_grad():
                    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).cuda() / 255.0
                    depth = depth_model(img_tensor)
                    depth = depth.cpu().numpy().squeeze()
                    # Normalize to 0-255
                    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8) * 255
                    depth = depth.astype(np.uint8)
            else:
                # Use simple fallback
                depth = estimate_depth_simple(img_rgb)

            # Encode depth as PNG
            depth_encoded = cv2.imencode(".png", depth)[1].tobytes()

            # Add depth to features
            features["observation/depth"].bytes_list.value[:] = [depth_encoded]

        # Write modified example
        writer.write(example.SerializeToString())

    writer.close()


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate pseudo depth for LIBERO dataset")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to RLDS dataset directory")
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name (e.g., libero_spatial)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for depth-augmented dataset")
    parser.add_argument("--model_size", type=str, default="small", choices=["small", "base", "large"],
                        help="Depth Anything V2 model size")
    parser.add_argument("--use_simple_depth", action="store_true",
                        help="Use simple grayscale depth instead of Depth Anything")

    args = parser.parse_args()

    # Load depth model
    depth_model = None
    if not args.use_simple_depth:
        print(f"Loading Depth Anything V2 ({args.model_size})...")
        depth_model = load_depth_anything_model(args.model_size)
        if depth_model is None:
            print("Falling back to simple depth estimation")

    # Process dataset
    input_dir = Path(args.data_dir) / args.dataset_name / "1.0.0"
    output_dir = Path(args.output_dir) / args.dataset_name / "1.0.0"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy dataset info files
    for file in ["dataset_info.json", "features.json"]:
        src = input_dir / file
        if src.exists():
            import shutil
            shutil.copy(src, output_dir / file)

    # Process TFRecord files
    tfrecord_files = sorted(input_dir.glob("*.tfrecord-*"))
    print(f"Found {len(tfrecord_files)} TFRecord files to process")

    for tfrecord_file in tfrecord_files:
        output_file = output_dir / tfrecord_file.name
        process_tfrecord_with_depth(str(tfrecord_file), str(output_file), depth_model)

    print(f"Depth-augmented dataset saved to {output_dir}")


if __name__ == "__main__":
    main()
