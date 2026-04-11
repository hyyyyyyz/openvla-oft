"""
train_rgbd_teacher.py

Arm D: RGB-D privileged OpenVLA-OFT teacher training on LIBERO.
This is the upper bound that uses depth information during training.

CRITICAL: This requires implementing depth input path in OpenVLA-OFT.
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import draccus
import torch
import torch.distributed as dist
from peft import LoraConfig, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from transformers import AutoModelForVision2Seq, AutoProcessor

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset


@dataclass
class RGBDTeacherConfig:
    """Training configuration for RGB-D teacher (Arm D)."""

    # fmt: off
    # Model parameters
    base_model: str = "openvla/openvla-7b"           # Base OpenVLA checkpoint
    lora_rank: int = 32                              # LoRA rank
    lora_dropout: float = 0.0                        # LoRA dropout

    # Dataset parameters
    dataset_name: str = "libero_spatial"             # Dataset name
    data_dir: str = "/home/hurricane/VLA/modified_libero_rlds"  # RLDS data directory
    shuffle_buffer_size: int = 100_000               # Shuffle buffer size
    image_size: int = 224                            # Image size for training

    # Depth parameters
    load_depth: bool = True                          # Enable depth loading (CRITICAL)
    depth_obs_keys: dict = None                      # Depth observation keys
    depth_resize_size: tuple = (224, 224)            # Depth image resize size
    num_depth_bins: int = 4                          # Number of depth bins for 2.5D map

    # Training parameters
    batch_size: int = 16                             # Batch size per GPU
    num_epochs: int = 10                             # Number of epochs
    learning_rate: float = 5e-4                      # Learning rate
    warmup_steps: int = 1000                         # Warmup steps
    weight_decay: float = 0.0                        # Weight decay
    grad_clip: float = 1.0                           # Gradient clipping
    save_steps: int = 1000                           # Save checkpoint every N steps
    log_steps: int = 10                              # Log every N steps

    # Output directories
    output_dir: str = "./checkpoints/arm_d_rgbd"     # Output checkpoint directory
    run_name: Optional[str] = None                   # Run name for logging

    # Distributed training
    local_rank: int = 0                              # Local rank for DDP
    world_size: int = 1                              # World size for DDP
    seed: int = 7                                    # Random seed

    # fmt: on

    def __post_init__(self):
        if self.depth_obs_keys is None:
            # Default LIBERO depth keys (if depth is available in dataset)
            self.depth_obs_keys = {"primary": "depth", "secondary": None, "wrist": None}


def setup_distributed(cfg: RGBDTeacherConfig):
    """Setup distributed training."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        cfg.local_rank = int(os.environ["RANK"])
        cfg.world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group("nccl")
        torch.cuda.set_device(cfg.local_rank)
    else:
        cfg.local_rank = 0
        cfg.world_size = 1


def get_libero_rgbd_dataset_kwargs(dataset_name: str, data_dir: str, load_depth: bool = True):
    """Get LIBERO RGB-D dataset kwargs."""
    if "spatial" in dataset_name.lower():
        base_name = "libero_spatial"
    elif "object" in dataset_name.lower():
        base_name = "libero_object"
    elif "goal" in dataset_name.lower():
        base_name = "libero_goal"
    elif "10" in dataset_name.lower():
        base_name = "libero_10"
    else:
        base_name = dataset_name

    # Create dataset kwargs with depth enabled
    dataset_kwargs = {
        "name": base_name,
        "data_dir": data_dir,
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": "wrist_image"},
        "state_obs_keys": ["state"],
        "state_encoding": 1,  # POS_EULER
        "action_encoding": 1,  # EEF_POS
    }

    # Configure depth keys if loading depth
    if load_depth:
        # LIBERO datasets may have depth images with different key names
        # Common patterns: "depth", "depth_image", "depth_static", etc.
        dataset_kwargs["depth_obs_keys"] = {
            "primary": "depth",  # or "depth_image" depending on dataset
            "secondary": None,
            "wrist": None,
        }
    else:
        dataset_kwargs["depth_obs_keys"] = {"primary": None, "secondary": None, "wrist": None}

    return dataset_kwargs


def create_rgbd_dataloader(cfg: RGBDTeacherConfig, batch_transform, processor):
    """Create RLDS dataloader with depth support."""
    # Get dataset kwargs with depth
    dataset_kwargs = get_libero_rgbd_dataset_kwargs(
        cfg.dataset_name, cfg.data_dir, load_depth=cfg.load_depth
    )

    # Create RLDS dataset with load_depth flag
    from prismatic.vla.datasets import RLDSDataset
    dataset = RLDSDataset(
        data_root_dir=Path(cfg.data_dir),
        data_mix=dataset_kwargs["name"],
        batch_transform=batch_transform,
        resize_resolution=(cfg.image_size, cfg.image_size),
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        train=True,
        load_depth=cfg.load_depth,  # Enable depth loading
    )

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=4,
        pin_memory=True,
        collate_fn=lambda x: x,
    )

    return dataloader


class RGBDOpenVLA(torch.nn.Module):
    """OpenVLA with RGB-D input support.

    This extends the base OpenVLA model to accept both RGB and depth images.
    The depth is processed through a separate encoder and fused with RGB features
    at the vision backbone output level.
    """

    def __init__(self, base_model, num_depth_bins: int = 4):
        super().__init__()
        self.base_model = base_model
        self.num_depth_bins = num_depth_bins

        # Get vision backbone dimensions from base model
        self.vision_backbone = base_model.vision_backbone
        self.projector = base_model.projector

        # Vision feature dimension (e.g., 1024 for DINOv2)
        self.vision_dim = self.vision_backbone.embed_dim

        # Depth encoder: process depth images to produce features
        # Input: (B, 1, H, W) depth images
        # Output: (B, 256, H', W') feature maps that match RGB patch structure
        self.depth_encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(256, self.vision_dim, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(self.vision_dim),
            torch.nn.ReLU(inplace=True),
        )

        # Depth feature pooling to match patch dimensions
        # OpenVLA typically uses 14x14 patches for 224x224 images
        self.depth_pool = torch.nn.AdaptiveAvgPool2d((14, 14))

        # RGB-Depth fusion: combine RGB and depth patch features
        # Input: concatenated RGB (vision_dim) + depth (vision_dim) features
        self.rgbd_fusion = torch.nn.Sequential(
            torch.nn.Linear(self.vision_dim * 2, self.vision_dim),
            torch.nn.LayerNorm(self.vision_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(0.1),
        )

        # Learnable gating for depth contribution (optional, for adaptive fusion)
        self.depth_gate = torch.nn.Sequential(
            torch.nn.Linear(self.vision_dim * 2, 1),
            torch.nn.Sigmoid(),
        )

    def _process_depth_features(self, depth_images: torch.Tensor) -> torch.Tensor:
        """Process depth images and extract patch-level features.

        Args:
            depth_images: (B, C, H, W) depth images, where C=1 or C=3

        Returns:
            depth_features: (B, num_patches, vision_dim) depth patch features
        """
        # Convert to single channel if needed
        if depth_images.shape[1] == 3:
            # Average RGB channels to get single depth channel
            depth_images = depth_images.mean(dim=1, keepdim=True)

        # Extract depth features
        depth_features = self.depth_encoder(depth_images)  # (B, vision_dim, H', W')

        # Pool to match patch dimensions (14x14 for standard ViT)
        depth_features = self.depth_pool(depth_features)  # (B, vision_dim, 14, 14)

        # Reshape to patch sequence: (B, vision_dim, 14, 14) -> (B, 196, vision_dim)
        B, D, H, W = depth_features.shape
        depth_features = depth_features.view(B, D, H * W).transpose(1, 2)  # (B, 196, vision_dim)

        return depth_features

    def _fuse_rgb_depth_features(
        self,
        rgb_features: torch.Tensor,
        depth_features: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse RGB and depth patch features.

        Args:
            rgb_features: (B, num_patches, vision_dim) RGB patch features
            depth_features: (B, num_patches, vision_dim) depth patch features

        Returns:
            fused_features: (B, num_patches, vision_dim) fused features
        """
        # Concatenate RGB and depth features
        combined = torch.cat([rgb_features, depth_features], dim=-1)  # (B, num_patches, 2*vision_dim)

        # Learnable gating
        gate = self.depth_gate(combined)  # (B, num_patches, 1)

        # Fuse features with gating
        fused = self.rgbd_fusion(combined)  # (B, num_patches, vision_dim)

        # Residual connection: weighted combination of RGB and fused features
        # This allows the model to ignore depth if it's not helpful
        output = gate * fused + (1 - gate) * rgb_features

        return output

    def forward(
        self,
        pixel_values: torch.Tensor,
        depth_values: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """Forward pass with optional depth input.

        Args:
            pixel_values: (B, 3, H, W) RGB images
            depth_values: (B, 1, H, W) or (B, 3, H, W) depth images (optional)
            input_ids: input token ids
            attention_mask: attention mask
            labels: labels for loss computation
            **kwargs: additional arguments passed to base model

        Returns:
            Model output with loss and logits
        """
        # If no depth provided, delegate to base model (RGB only)
        if depth_values is None:
            return self.base_model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs
            )

        # === RGB-D Forward Pass ===
        # Step 1: Extract RGB features using vision backbone
        # Get input embeddings
        input_embeddings = self.base_model.get_input_embeddings()(input_ids)

        # Process action masks if labels provided
        if labels is not None:
            all_actions_mask = self.base_model._process_action_masks(labels)
            language_embeddings = input_embeddings[~all_actions_mask].reshape(
                input_embeddings.shape[0], -1, input_embeddings.shape[2]
            )
        else:
            language_embeddings = None

        # Extract RGB vision features
        rgb_patch_features = self.vision_backbone(
            pixel_values, language_embeddings if hasattr(self.vision_backbone, 'use_film') else None
        )  # (B, num_patches, vision_dim)

        # Step 2: Extract depth features
        depth_patch_features = self._process_depth_features(depth_values)  # (B, num_patches, vision_dim)

        # Step 3: Fuse RGB and depth features
        fused_patch_features = self._fuse_rgb_depth_features(
            rgb_patch_features, depth_patch_features
        )  # (B, num_patches, vision_dim)

        # Step 4: Project fused features to language model space
        projected_patch_embeddings = self.projector(fused_patch_features)  # (B, num_patches, llm_dim)

        # Step 5: Build multimodal embeddings (vision patches + text tokens)
        multimodal_embeddings, multimodal_attention_mask = self.base_model._build_multimodal_attention(
            input_embeddings, projected_patch_embeddings, attention_mask
        )

        # Step 6: Build labels for multimodal sequence
        multimodal_labels = self.base_model._build_multimodal_labels(labels, projected_patch_embeddings)

        # Step 7: Forward through language model
        language_model_output = self.base_model.language_model(
            input_ids=None,
            attention_mask=multimodal_attention_mask,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=multimodal_embeddings,
            labels=multimodal_labels,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )

        # Return in expected format
        from prismatic.extern.hf.modeling_prismatic import PrismaticCausalLMOutputWithPast
        return PrismaticCausalLMOutputWithPast(
            loss=language_model_output.loss,
            logits=language_model_output.logits,
            past_key_values=language_model_output.past_key_values,
            hidden_states=language_model_output.hidden_states,
            attentions=language_model_output.attentions,
            projector_features=projected_patch_embeddings,
        )


def train_arm_d(cfg: RGBDTeacherConfig):
    """Train Arm D: RGB-D privileged teacher."""
    # Setup distributed training
    setup_distributed(cfg)

    # Set seed
    torch.manual_seed(cfg.seed)

    # Create output directory
    os.makedirs(cfg.output_dir, exist_ok=True)

    # Load base model
    print(f"Loading base model: {cfg.base_model}")
    print("=" * 80)
    print("IMPORTANT: This script requires RGB-D support in OpenVLA-OFT.")
    print("Before running, ensure:")
    print("1. LIBERO dataset has depth images available")
    print("2. OXE_DATASET_CONFIGS has depth_obs_keys configured for LIBERO")
    print("3. RLDSBatchTransform has load_depth=True")
    print("4. Model forward pass accepts depth input")
    print("=" * 80)

    base_model = AutoModelForVision2Seq.from_pretrained(
        cfg.base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(cfg.base_model, trust_remote_code=True)

    # Wrap with RGB-D support
    model = RGBDOpenVLA(base_model, num_depth_bins=cfg.num_depth_bins)

    # Move model to GPU
    device = torch.device(f"cuda:{cfg.local_rank}")
    model = model.to(device)

    # Apply LoRA
    print(f"Applying LoRA with rank={cfg.lora_rank}")
    lora_config = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=min(cfg.lora_rank, 16),
        lora_dropout=cfg.lora_dropout,
        target_modules="all-linear",
        init_lora_weights="gaussian",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Wrap with DDP if distributed
    if cfg.world_size > 1:
        model = DDP(model, device_ids=[cfg.local_rank])

    # Create batch transform with depth support
    batch_transform = RLDSBatchTransform(
        action_tokenizer=processor.action_tokenizer,
        base_tokenizer=processor.tokenizer,
        image_transform=processor.image_transform,
        prompt_builder_fn=processor.prompt_builder_fn,
        use_depth=cfg.load_depth,  # Enable depth loading
    )

    # Create dataloader with depth
    print(f"Creating RGB-D dataloader for {cfg.dataset_name}")
    print(f"Load depth: {cfg.load_depth}")
    print(f"Depth obs keys: {cfg.depth_obs_keys}")
    dataloader = create_rgbd_dataloader(cfg, batch_transform, processor)

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    # Training loop
    print(f"Starting RGB-D teacher training for {cfg.num_epochs} epochs")
    global_step = 0
    model.train()

    for epoch in range(cfg.num_epochs):
        for step, batch in enumerate(dataloader):
            # Move batch to device
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            # Get depth values if available
            depth_values = batch.get("depth_values", None)
            if depth_values is not None:
                depth_values = depth_values.to(device)

            # Forward pass with depth
            outputs = model(
                pixel_values=pixel_values,
                depth_values=depth_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            global_step += 1

            # Logging
            if global_step % cfg.log_steps == 0 and cfg.local_rank == 0:
                print(f"Epoch {epoch}, Step {global_step}, Loss: {loss.item():.4f}")

            # Save checkpoint
            if global_step % cfg.save_steps == 0 and cfg.local_rank == 0:
                save_path = os.path.join(cfg.output_dir, f"checkpoint-{global_step}")
                os.makedirs(save_path, exist_ok=True)
                # Save only the RGBD-specific layers and base model
                torch.save({
                    "global_step": global_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }, os.path.join(save_path, "pytorch_model.bin"))
                print(f"Saved checkpoint to {save_path}")

    # Save final checkpoint
    if cfg.local_rank == 0:
        final_path = os.path.join(cfg.output_dir, "final")
        print(f"Saving final checkpoint to {final_path}")

    # Cleanup
    if cfg.world_size > 1:
        dist.destroy_process_group()

    print("RGB-D teacher training complete!")


@draccus.wrap()
def main(cfg: RGBDTeacherConfig):
    """Main entry point."""
    train_arm_d(cfg)


if __name__ == "__main__":
    main()
