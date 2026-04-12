"""
train_multiview_teacher.py

Arm D: Multi-view RGB privileged OpenVLA-OFT teacher training on LIBERO.
Uses static camera (primary) + wrist camera for privileged geometry information.
This is the upper bound that uses multi-view fusion during training.
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import draccus
import torch
import torch.distributed as dist
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from transformers import AutoModelForVision2Seq, AutoProcessor

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset


@dataclass
class MultiViewTeacherConfig:
    """Training configuration for multi-view teacher (Arm D)."""

    # fmt: off
    # Model parameters
    base_model: str = "openvla/openvla-7b"           # Base OpenVLA checkpoint
    lora_rank: int = 32                              # LoRA rank
    lora_dropout: float = 0.0                        # LoRA dropout

    # Dataset parameters
    dataset_name: str = "libero_spatial"             # Dataset name
    data_dir: str = "/home/hurricane/VLA/modified_libero_rlds"  # RLDS data directory
    shuffle_buffer_size: int = 10_000                # Shuffle buffer size (reduce if OOM)
    image_size: int = 224                            # Image size for training

    # Multi-view parameters
    use_wrist_camera: bool = True                    # Enable wrist camera (privileged)
    num_images_in_input: int = 2                     # Number of images (1 static + 1 wrist)

    # Training parameters
    batch_size: int = 1                              # Batch size per GPU (locked at 1 for 24GB)
    num_epochs: int = 100                            # Large enough; actual stop controlled by max_steps
    learning_rate: float = 5e-4                      # Learning rate
    warmup_steps: int = 1000                         # Warmup steps
    weight_decay: float = 0.0                        # Weight decay
    grad_clip: float = 1.0                           # Gradient clipping
    save_steps: int = 1000                           # Save checkpoint every N steps
    log_steps: int = 10                              # Log every N steps
    max_steps: int = 30000                          # Stop training after N steps (0 = no limit)

    # Output directories
    output_dir: str = "./checkpoints/arm_d_official"  # Output checkpoint directory (matches eval)
    run_name: Optional[str] = None                   # Run name for logging

    # Distributed training
    local_rank: int = 0                              # Local rank for DDP
    world_size: int = 1                              # World size for DDP
    seed: int = 7                                    # Random seed

    # fmt: on


def setup_distributed(cfg: MultiViewTeacherConfig):
    """Setup distributed training."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        cfg.local_rank = int(os.environ["RANK"])
        cfg.world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group("nccl")
        torch.cuda.set_device(cfg.local_rank)
    else:
        cfg.local_rank = 0
        cfg.world_size = 1


class MultiViewOpenVLA(nn.Module):
    """OpenVLA with multi-view RGB input support.

    Fuses static camera (primary) and wrist camera features at the vision backbone output level.
    """

    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

        # Get vision backbone dimensions
        self.vision_backbone = base_model.vision_backbone
        self.projector = base_model.projector
        self.vision_dim = self.vision_backbone.embed_dim

        # Multi-view fusion: combine static and wrist features
        self.multiview_fusion = nn.Sequential(
            nn.Linear(self.vision_dim * 2, self.vision_dim),
            nn.LayerNorm(self.vision_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # Learnable gating for wrist camera contribution
        self.wrist_gate = nn.Sequential(
            nn.Linear(self.vision_dim * 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        pixel_values: torch.Tensor,  # (B, num_images, C, H, W) or (B, C, H, W)
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """Forward pass with multi-view input.

        Args:
            pixel_values: (B, num_images, C, H, W) for multi-view or (B, C, H, W) for single-view
            input_ids: input token ids
            attention_mask: attention mask
            labels: labels for loss computation
            **kwargs: additional arguments

        Returns:
            Model output with loss and logits
        """
        # Handle single-view input (fallback to base model)
        if pixel_values.dim() == 4:
            return self.base_model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs
            )

        # Multi-view forward pass
        B, num_images, C, H, W = pixel_values.shape

        # Extract features for each view
        all_patch_features = []
        for i in range(num_images):
            view_pixels = pixel_values[:, i]  # (B, C, H, W)
            # Get vision features from base model (allow gradients for fine-tuning)
            patch_features = self.vision_backbone(view_pixels)  # (B, num_patches, vision_dim)
            all_patch_features.append(patch_features)

        # Stack features: (B, num_views, num_patches, vision_dim)
        stacked_features = torch.stack(all_patch_features, dim=1)

        # Fuse features across views
        # Average pool across patches first to get view-level features
        view_features = stacked_features.mean(dim=2)  # (B, num_views, vision_dim)

        # Concatenate static and wrist features
        static_features = view_features[:, 0]  # (B, vision_dim)
        wrist_features = view_features[:, 1] if num_images > 1 else static_features

        combined = torch.cat([static_features, wrist_features], dim=-1)  # (B, 2*vision_dim)

        # Learnable gating
        gate = self.wrist_gate(combined)  # (B, 1)

        # Fuse features
        fused = self.multiview_fusion(combined)  # (B, vision_dim)

        # Expand back to patch level (broadcast fusion weights to all patches)
        static_patch_features = all_patch_features[0]  # (B, num_patches, vision_dim)
        fused_patches = static_patch_features + gate.unsqueeze(1) * fused.unsqueeze(1)

        # Project to language model space
        projected_patch_embeddings = self.projector(fused_patches)

        # Get input embeddings
        input_embeddings = self.base_model.get_input_embeddings()(input_ids)

        # Build multimodal embeddings
        multimodal_embeddings, multimodal_attention_mask = self.base_model._build_multimodal_attention(
            input_embeddings, projected_patch_embeddings, attention_mask
        )

        # Build labels
        multimodal_labels = self.base_model._build_multimodal_labels(labels, projected_patch_embeddings)

        # Forward through language model
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


def create_dataloader(cfg: MultiViewTeacherConfig, batch_transform, action_tokenizer, processor):
    """Create RLDS dataloader for multi-view training."""
    # Map dataset name to OXE config name
    dataset_name_map = {
        "libero_spatial": "libero_spatial_no_noops",
        "libero_object": "libero_object_no_noops",
        "libero_goal": "libero_goal_no_noops",
        "libero_10": "libero_10_no_noops",
    }

    data_mix = dataset_name_map.get(cfg.dataset_name, cfg.dataset_name)

    # Create RLDS dataset with multi-view support
    dataset = RLDSDataset(
        data_root_dir=Path(cfg.data_dir),
        data_mix=data_mix,
        batch_transform=batch_transform,
        resize_resolution=(cfg.image_size, cfg.image_size),
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        train=True,
        load_depth=False,
    )

    # Create collator
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length,
        processor.tokenizer.pad_token_id,
        padding_side="right",
    )

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=0,  # Important: Set to 0 for RLDS
        pin_memory=True,
        collate_fn=collator,
    )

    return dataloader


def train_arm_d(cfg: MultiViewTeacherConfig):
    """Train Arm D: Multi-view privileged teacher."""
    # Setup distributed training
    setup_distributed(cfg)

    # Set seed
    torch.manual_seed(cfg.seed)

    # Create output directory
    os.makedirs(cfg.output_dir, exist_ok=True)

    # Load base model
    print(f"Loading base model: {cfg.base_model}")
    print(f"Multi-view mode: static + wrist camera")
    base_model = AutoModelForVision2Seq.from_pretrained(
        cfg.base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(cfg.base_model, trust_remote_code=True)

    # Apply LoRA to base model
    print(f"Applying LoRA with rank={cfg.lora_rank}")
    lora_config = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=min(cfg.lora_rank, 16),
        lora_dropout=cfg.lora_dropout,
        target_modules="all-linear",
        init_lora_weights="gaussian",
    )
    base_model = get_peft_model(base_model, lora_config)
    base_model.print_trainable_parameters()

    # Wrap with multi-view support
    model = MultiViewOpenVLA(base_model)

    # Move model to GPU
    device = torch.device(f"cuda:{cfg.local_rank}")
    model = model.to(device)

    # Wrap with DDP if distributed
    if cfg.world_size > 1:
        model = DDP(model, device_ids=[cfg.local_rank])

    # Create action tokenizer
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    # Create batch transform with multi-view support
    batch_transform = RLDSBatchTransform(
        action_tokenizer=action_tokenizer,
        base_tokenizer=processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
        use_depth=False,
    )

    # Create dataloader
    print(f"Creating multi-view dataloader for {cfg.dataset_name}")
    dataloader = create_dataloader(cfg, batch_transform, action_tokenizer, processor)

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    # Training loop
    print(f"Starting multi-view teacher training for {cfg.num_epochs} epochs")
    global_step = 0
    model.train()

    for epoch in range(cfg.num_epochs):
        for step, batch in enumerate(dataloader):
            # Move batch to device and convert to bfloat16
            pixel_values = batch["pixel_values"].to(device, dtype=torch.bfloat16)
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            # Forward pass
            outputs = model(
                pixel_values=pixel_values,
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

            # Stop if max_steps reached
            if cfg.max_steps > 0 and global_step >= cfg.max_steps:
                if cfg.local_rank == 0:
                    final_path = os.path.join(cfg.output_dir, "final")
                    os.makedirs(final_path, exist_ok=True)
                    if cfg.world_size > 1:
                        model.module.base_model.save_pretrained(final_path)
                    else:
                        model.base_model.save_pretrained(final_path)
                    print(f"Reached max_steps={cfg.max_steps}, saved final to {final_path}")
                return

            # Save checkpoint
            if global_step % cfg.save_steps == 0 and cfg.local_rank == 0:
                save_path = os.path.join(cfg.output_dir, f"checkpoint-{global_step}")
                os.makedirs(save_path, exist_ok=True)
                # Save the base model (LoRA weights)
                if cfg.world_size > 1:
                    model.module.base_model.save_pretrained(save_path)
                else:
                    model.base_model.save_pretrained(save_path)
                print(f"Saved checkpoint to {save_path}")

    # Save final checkpoint
    if cfg.local_rank == 0:
        final_path = os.path.join(cfg.output_dir, "final")
        os.makedirs(final_path, exist_ok=True)
        if cfg.world_size > 1:
            model.module.base_model.save_pretrained(final_path)
        else:
            model.base_model.save_pretrained(final_path)
        print(f"Saved final checkpoint to {final_path}")

    # Cleanup
    if cfg.world_size > 1:
        dist.destroy_process_group()

    print("Multi-view teacher training complete!")


@draccus.wrap()
def main(cfg: MultiViewTeacherConfig):
    """Main entry point."""
    train_arm_d(cfg)


if __name__ == "__main__":
    main()
