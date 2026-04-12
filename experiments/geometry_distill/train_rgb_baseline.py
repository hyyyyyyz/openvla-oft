"""
train_rgb_baseline.py

Arm A: RGB-only OpenVLA-OFT baseline training on LIBERO.
This is the lower bound baseline for geometry-critical failure diagnosis.
"""

import os
import sys
# Limit TensorFlow threads to avoid main thread blocking
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import draccus
import torch
import torch.distributed as dist
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
class TrainConfig:
    """Training configuration for RGB baseline (Arm A)."""

    # fmt: off
    # Model parameters
    base_model: str = "openvla/openvla-7b"           # Base OpenVLA checkpoint
    lora_rank: int = 32                              # LoRA rank
    lora_dropout: float = 0.0                        # LoRA dropout

    # Dataset parameters
    dataset_name: str = "libero_spatial"             # Dataset name (libero_spatial, libero_goal, etc.)
    data_dir: str = "/home/hurricane/VLA/modified_libero_rlds"  # RLDS data directory
    shuffle_buffer_size: int = 10_000               # Shuffle buffer size (same as Arm D)
    image_size: int = 224                            # Image size for training

    # Training parameters
    batch_size: int = 1                              # Batch size per GPU (1 for 24GB GPU)
    num_epochs: int = 100                            # Large enough; actual stop controlled by max_steps
    max_steps: int = 30000                           # Stop training after N steps
    learning_rate: float = 5e-4                      # Learning rate
    warmup_steps: int = 1000                         # Warmup steps
    weight_decay: float = 0.0                        # Weight decay
    grad_clip: float = 1.0                           # Gradient clipping
    save_steps: int = 1000                           # Save checkpoint every N steps
    log_steps: int = 50                              # Log every N steps

    # Output directories
    output_dir: str = "./checkpoints/arm_a_rgb"      # Output checkpoint directory
    run_name: Optional[str] = None                   # Run name for logging

    # Distributed training
    local_rank: int = 0                              # Local rank for DDP
    world_size: int = 1                              # World size for DDP
    seed: int = 7                                    # Random seed

    # fmt: on


def setup_distributed(cfg: TrainConfig):
    """Setup distributed training."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        cfg.local_rank = int(os.environ["RANK"])
        cfg.world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group("nccl")
        torch.cuda.set_device(cfg.local_rank)
    else:
        cfg.local_rank = 0
        cfg.world_size = 1


def create_dataloader(cfg: TrainConfig, batch_transform, action_tokenizer, processor):
    """Create RLDS dataloader for LIBERO."""
    # Map dataset name to OXE config name
    dataset_name_map = {
        "libero_spatial": "libero_spatial_no_noops",
        "libero_object": "libero_object_no_noops",
        "libero_goal": "libero_goal_no_noops",
        "libero_10": "libero_10_no_noops",
    }

    # Get the correct dataset name for OXE_DATASET_CONFIGS
    if cfg.dataset_name in dataset_name_map:
        data_mix = dataset_name_map[cfg.dataset_name]
    else:
        data_mix = cfg.dataset_name

    # Create RLDS dataset
    dataset = RLDSDataset(
        data_root_dir=Path(cfg.data_dir),
        data_mix=data_mix,
        batch_transform=batch_transform,
        resize_resolution=(cfg.image_size, cfg.image_size),
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        train=True,
        load_depth=False,  # No depth for RGB baseline
    )

    # Create collator for action prediction
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length,
        processor.tokenizer.pad_token_id,
        padding_side="right",
    )

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=0,  # Important: Set to 0 for RLDS which uses its own parallelism
        pin_memory=False,  # False with num_workers=0 to avoid main thread blocking
        collate_fn=collator,
    )

    return dataloader


def train_arm_a(cfg: TrainConfig):
    """Train Arm A: RGB-only baseline."""
    # Setup distributed training
    setup_distributed(cfg)

    # Set seed
    torch.manual_seed(cfg.seed)

    # Create output directory
    os.makedirs(cfg.output_dir, exist_ok=True)

    # Setup logging to file
    log_file = os.path.join(cfg.output_dir, "training.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Output directory: {cfg.output_dir}")
    logger.info(f"Dataset: {cfg.dataset_name}, Batch size: {cfg.batch_size}, Epochs: {cfg.num_epochs}, Max steps: {cfg.max_steps}")

            # Load base model and processor
    logger.info(f"Loading base model: {cfg.base_model}")
    base_model = AutoModelForVision2Seq.from_pretrained(
        cfg.base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(cfg.base_model, trust_remote_code=True)

    # Apply LoRA to base model
    logger.info(f"Applying LoRA with rank={cfg.lora_rank}")
    lora_config = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=min(cfg.lora_rank, 16),
        lora_dropout=cfg.lora_dropout,
        target_modules="all-linear",
        init_lora_weights="gaussian",
    )
    base_model = get_peft_model(base_model, lora_config)
    base_model.print_trainable_parameters()

    # Move model to GPU
    device = torch.device(f"cuda:{cfg.local_rank}")
    model = base_model.to(device)

    # Wrap with DDP if distributed
    if cfg.world_size > 1:
        model = DDP(model, device_ids=[cfg.local_rank])

    # Create action tokenizer
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    # Create batch transform
    batch_transform = RLDSBatchTransform(
        action_tokenizer=action_tokenizer,
        base_tokenizer=processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
        use_depth=False,  # No depth for RGB baseline
    )

    # Create dataloader
    logger.info(f"Creating dataloader for {cfg.dataset_name}")
    dataloader = create_dataloader(cfg, batch_transform, action_tokenizer, processor)

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    # Training loop
    logger.info(f"Starting training for {cfg.num_epochs} epochs (target max_steps={cfg.max_steps})")
    global_step = 0
    model.train()

    for epoch in range(cfg.num_epochs):
        for step, batch in enumerate(dataloader):
            if global_step >= cfg.max_steps:
                logger.info(f"Reached max_steps={cfg.max_steps}, stopping training")
                break
        if global_step >= cfg.max_steps:
            break
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
                logger.info(f"Epoch {epoch}, Step {global_step}, Loss: {loss.item():.4f}")

            # Save checkpoint
            if global_step % cfg.save_steps == 0 and cfg.local_rank == 0:
                save_path = os.path.join(cfg.output_dir, f"checkpoint-{global_step}")
                os.makedirs(save_path, exist_ok=True)
                # Save the model
                if cfg.world_size > 1:
                    model.module.save_pretrained(save_path)
                else:
                    model.save_pretrained(save_path)
                logger.info(f"Saved checkpoint to {save_path}")

    # Save final checkpoint
    if cfg.local_rank == 0:
        final_path = os.path.join(cfg.output_dir, "final")
        os.makedirs(final_path, exist_ok=True)
        if cfg.world_size > 1:
            model.module.save_pretrained(final_path)
        else:
            model.save_pretrained(final_path)
        logger.info(f"Saved final checkpoint to {final_path}")

    # Cleanup distributed
    if cfg.world_size > 1:
        dist.destroy_process_group()

    logger.info("Training complete!")


@draccus.wrap()
def main(cfg: TrainConfig):
    """Main entry point."""
    train_arm_a(cfg)


if __name__ == "__main__":
    main()
