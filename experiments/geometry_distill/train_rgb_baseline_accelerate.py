"""
train_rgb_baseline_accelerate.py

Arm A: RGB-only OpenVLA-OFT baseline training on LIBERO.
Single-view (static camera only), using Accelerate framework.

This is the lower bound baseline for geometry-critical failure diagnosis.
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import logging

import draccus
import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForVision2Seq, AutoProcessor
from accelerate import Accelerator
from torch.optim import AdamW

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics


@dataclass
class RGBBaselineConfig:
    """Training configuration for RGB baseline (Arm A)."""

    # fmt: off
    # Model parameters
    vla_path: str = "openvla/openvla-7b"              # Base OpenVLA checkpoint
    lora_rank: int = 32                               # LoRA rank
    lora_dropout: float = 0.0                        # LoRA dropout

    # Dataset parameters
    dataset_name: str = "libero_spatial"              # Dataset name (OXE format)
    data_root_dir: str = "/home/hurricane/VLA/modified_libero_rlds"
    shuffle_buffer_size: int = 10_000                 # Shuffle buffer size

    # Training parameters
    batch_size: int = 1                               # Batch size per GPU (1 for 24GB GPU)
    learning_rate: float = 5e-4                       # Learning rate
    max_steps: int = 15_000                          # Max training steps
    grad_accumulation_steps: int = 1                  # Gradient accumulation
    save_freq: int = 1000                             # Save checkpoint every N steps
    log_freq: int = 50                                # Log every N steps

    # Output directories
    run_root_dir: str = "./checkpoints/arm_a_rgb"     # Output directory

    # Resume training
    resume_from_checkpoint: bool = False              # Resume from checkpoint
    resume_step: int = 0                             # Step to resume from

    # Distributed training
    seed: int = 7                                     # Random seed

    # fmt: on


def train_rgb_baseline(cfg: RGBBaselineConfig):
    """Train Arm A: RGB-only baseline using Accelerate."""

    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.grad_accumulation_steps,
        mixed_precision="bf16",
    )

    # Create output directory
    run_dir = Path(cfg.run_root_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging to file
    log_file = run_dir / "training.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)

    # Set seed
    torch.manual_seed(cfg.seed)

    # Map dataset name to OXE config name
    dataset_name_map = {
        "libero_spatial": "libero_spatial_no_noops",
        "libero_object": "libero_object_no_noops",
        "libero_goal": "libero_goal_no_noops",
        "libero_10": "libero_10_no_noops",
    }
    data_mix = dataset_name_map.get(cfg.dataset_name, cfg.dataset_name)
    logger.info(f"Dataset mapping: {cfg.dataset_name} -> {data_mix}")

    # Load processor
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)

    # Load base model
    logger.info(f"Loading base model: {cfg.vla_path}")
    logger.info(f"Single-view mode: static camera only (no wrist)")

    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # Apply LoRA
    logger.info(f"Applying LoRA with rank={cfg.lora_rank}")
    lora_config = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=min(cfg.lora_rank, 16),
        lora_dropout=cfg.lora_dropout,
        target_modules="all-linear",
        init_lora_weights="gaussian",
    )
    vla = get_peft_model(vla, lora_config)
    vla.print_trainable_parameters()

    # Resume from checkpoint if specified
    start_step = 0
    if cfg.resume_from_checkpoint and cfg.resume_step > 0:
        checkpoint_path = run_dir / f"checkpoint-{cfg.resume_step}"
        if checkpoint_path.exists():
            logger.info(f"Resuming from checkpoint: {checkpoint_path}")
            vla.load_adapter(str(checkpoint_path), "default")
            start_step = cfg.resume_step
            logger.info(f"Resumed from step {start_step}")
        else:
            logger.warning(f"Checkpoint {checkpoint_path} not found, starting from scratch")

    # Create action tokenizer
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    # Create batch transform (single-view: no wrist camera)
    batch_transform = RLDSBatchTransform(
        action_tokenizer=action_tokenizer,
        base_tokenizer=processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
        use_wrist_image=False,  # Single-view: static camera only
        use_proprio=False,
        use_depth=False,        # No depth
    )

    # Create dataset
    train_dataset = RLDSDataset(
        data_root_dir=Path(cfg.data_root_dir),
        data_mix=data_mix,
        batch_transform=batch_transform,
        resize_resolution=tuple(vla.config.image_sizes),
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        train=True,
        image_aug=False,
    )

    # Save dataset statistics
    if accelerator.is_main_process:
        save_dataset_statistics(train_dataset.dataset_statistics, run_dir)

    # Create collator
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length,
        processor.tokenizer.pad_token_id,
        padding_side="right",
    )

    # Create dataloader
    dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,  # Important: 0 for RLDS
    )

    # Create optimizer
    optimizer = AdamW(vla.parameters(), lr=cfg.learning_rate)

    # Prepare for distributed training
    vla, optimizer, dataloader = accelerator.prepare(vla, optimizer, dataloader)

    # Training loop
    logger.info(f"Starting RGB baseline training for {cfg.max_steps} steps (starting from step {start_step})")
    vla.train()
    optimizer.zero_grad()

    global_step = 0
    for batch in dataloader:
        # Skip batches until we reach start_step
        if global_step < start_step:
            global_step += 1
            continue

        with accelerator.accumulate(vla):
            # Forward pass
            output = vla(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values=batch["pixel_values"].to(torch.bfloat16),
                labels=batch["labels"],
            )

            loss = output.loss

            # Backward pass
            accelerator.backward(loss)
            torch.nn.utils.clip_grad_norm_(vla.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        global_step += 1

        # Logging
        if global_step % cfg.log_freq == 0 and accelerator.is_main_process:
            logger.info(f"Step {global_step}, Loss: {loss.item():.6e}")

        # Save checkpoint
        if global_step % cfg.save_freq == 0 and accelerator.is_main_process:
            checkpoint_dir = run_dir / f"checkpoint-{global_step}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            accelerator.unwrap_model(vla).save_pretrained(checkpoint_dir)
            logger.info(f"Saved checkpoint to {checkpoint_dir}")

        if global_step >= cfg.max_steps:
            break

    # Save final checkpoint
    if accelerator.is_main_process:
        final_dir = run_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        accelerator.unwrap_model(vla).save_pretrained(final_dir)
        logger.info(f"Saved final checkpoint to {final_dir}")

    logger.info("RGB baseline training complete!")


@draccus.wrap()
def main(cfg: RGBBaselineConfig):
    """Main entry point."""
    train_rgb_baseline(cfg)


if __name__ == "__main__":
    main()
