"""
train_rgb_baseline.py

Arm A: RGB-only OpenVLA-OFT baseline training on LIBERO.
This is the lower bound baseline for geometry-critical failure diagnosis.
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import draccus
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from transformers import AutoModelForVision2Seq, AutoProcessor

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from prismatic.vla.datasets.rlds.oxe import make_oxe_dataset_kwargs
from prismatic.vla.datasets.rlds.oxe.configs import OXE_DATASET_CONFIGS
from prismatic.vla.datasets.rlds.oxe.mixtures import OXE_MIXTURES


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
    data_dir: str = "/path/to/modified_libero_rlds"  # RLDS data directory
    shuffle_buffer_size: int = 100_000               # Shuffle buffer size
    image_size: int = 224                            # Image size for training

    # Training parameters
    batch_size: int = 16                             # Batch size per GPU
    num_epochs: int = 10                             # Number of epochs
    learning_rate: float = 5e-4                      # Learning rate
    warmup_steps: int = 1000                         # Warmup steps
    weight_decay: float = 0.0                        # Weight decay
    grad_clip: float = 1.0                           # Gradient clipping
    save_steps: int = 1000                           # Save checkpoint every N steps
    eval_steps: int = 500                            # Evaluate every N steps
    log_steps: int = 10                              # Log every N steps

    # Output directories
    output_dir: str = "./checkpoints/arm_a_rgb"      # Output checkpoint directory
    run_name: Optional[str] = None                   # Run name for logging

    # Distributed training
    local_rank: int = 0                              # Local rank for DDP
    world_size: int = 1                              # World size for DDP
    seed: int = 7                                    # Random seed

    # LIBERO specific
    task_suite: str = "libero_spatial"               # LIBERO task suite
    num_actions_chunk: int = 10                      # Action chunk size

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


def get_libero_dataset_kwargs(dataset_name: str, data_dir: str):
    """Get LIBERO dataset kwargs for OXE config."""
    # LIBERO datasets use modified_libero_rlds format
    # Map dataset_name to OXE config
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

    # Create dataset kwargs
    dataset_kwargs = {
        "name": base_name,
        "data_dir": data_dir,
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": "wrist_image"},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},  # No depth for Arm A
        "state_obs_keys": ["state"],
        "state_encoding": 1,  # POS_EULER
        "action_encoding": 1,  # EEF_POS
    }
    return dataset_kwargs


def create_dataloader(cfg: TrainConfig, batch_transform):
    """Create RLDS dataloader for LIBERO."""
    # Get dataset kwargs
    dataset_kwargs = get_libero_dataset_kwargs(cfg.dataset_name, cfg.data_dir)

    # Create RLDS dataset
    dataset = RLDSDataset(
        data_dir=cfg.data_dir,
        dataset_kwargs=[dataset_kwargs],
        batch_transform=batch_transform,
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        train=True,
    )

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=4,
        pin_memory=True,
        collate_fn=lambda x: x,  # RLDS handles batching
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

    # Load base model and processor
    print(f"Loading base model: {cfg.base_model}")
    model = AutoModelForVision2Seq.from_pretrained(
        cfg.base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(cfg.base_model, trust_remote_code=True)

    # Move model to GPU
    device = torch.device(f"cuda:{cfg.local_rank}")
    model = model.to(device)

    # Apply LoRA
    print(f"Applying LoRA with rank={cfg.lora_rank}")
    # TODO: Apply LoRA to model (using PEFT or custom implementation)

    # Wrap with DDP if distributed
    if cfg.world_size > 1:
        model = DDP(model, device_ids=[cfg.local_rank])

    # Create batch transform
    batch_transform = RLDSBatchTransform(
        action_tokenizer=processor.action_tokenizer,
        image_transform=processor.image_transform,
        prompt_builder_fn=processor.prompt_builder_fn,
    )

    # Create dataloader
    print(f"Creating dataloader for {cfg.dataset_name}")
    dataloader = create_dataloader(cfg, batch_transform)

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    # Training loop
    print(f"Starting training for {cfg.num_epochs} epochs")
    global_step = 0
    model.train()

    for epoch in range(cfg.num_epochs):
        for step, batch in enumerate(dataloader):
            # Forward pass
            # TODO: Implement forward pass with action prediction loss

            # Backward pass
            optimizer.zero_grad()
            # loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            global_step += 1

            # Logging
            if global_step % cfg.log_steps == 0 and cfg.local_rank == 0:
                print(f"Epoch {epoch}, Step {global_step}, Loss: {0.0:.4f}")

            # Save checkpoint
            if global_step % cfg.save_steps == 0 and cfg.local_rank == 0:
                checkpoint_path = os.path.join(cfg.output_dir, f"checkpoint-{global_step}")
                print(f"Saving checkpoint to {checkpoint_path}")
                # TODO: Save checkpoint

    # Save final checkpoint
    if cfg.local_rank == 0:
        final_path = os.path.join(cfg.output_dir, "final")
        print(f"Saving final checkpoint to {final_path}")
        # TODO: Save final checkpoint

    # Cleanup distributed
    if cfg.world_size > 1:
        dist.destroy_process_group()

    print("Training complete!")


@draccus.wrap()
def main(cfg: TrainConfig):
    """Main entry point."""
    train_arm_a(cfg)


if __name__ == "__main__":
    main()
