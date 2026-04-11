"""
train_rgb_distill.py

Arm B/C: RGB-only student with 2.5D interaction map distillation.
Arm B: Distill from RGB-D teacher (privileged geometry)
Arm C: Distill from RGB teacher (control - teacher smoothing only)
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import draccus
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from transformers import AutoModelForVision2Seq, AutoProcessor

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset


@dataclass
class DistillConfig:
    """Training configuration for RGB distillation (Arm B/C)."""

    # fmt: off
    # Model parameters
    base_model: str = "openvla/openvla-7b"           # Base OpenVLA checkpoint
    lora_rank: int = 32                              # LoRA rank

    # Teacher checkpoint (Arm D for B, Arm A for C)
    teacher_checkpoint: str = ""                     # Path to teacher checkpoint
    teacher_type: str = "rgbd"                       # "rgbd" (Arm D) or "rgb" (Arm A)

    # Dataset parameters
    dataset_name: str = "libero_spatial"             # Dataset name
    data_dir: str = "/path/to/modified_libero_rlds"  # RLDS data directory
    shuffle_buffer_size: int = 100_000               # Shuffle buffer size
    image_size: int = 224                            # Image size

    # Distillation parameters
    lambda_kl: float = 0.5                           # KL distillation loss weight
    num_depth_bins: int = 4                          # Number of depth bins for 2.5D map
    temperature: float = 1.0                         # Temperature for KL divergence

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
    output_dir: str = "./checkpoints/arm_b_distill"  # Output checkpoint directory
    run_name: Optional[str] = None                   # Run name for logging

    # Distributed training
    local_rank: int = 0                              # Local rank for DDP
    world_size: int = 1                              # World size for DDP
    seed: int = 7                                    # Random seed

    # Arm selection
    arm: str = "b"                                   # "b" for Arm B, "c" for Arm C

    # fmt: on


def setup_distributed(cfg: DistillConfig):
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
    """Get LIBERO dataset kwargs."""
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

    return {
        "name": base_name,
        "data_dir": data_dir,
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": "wrist_image"},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},  # Student is RGB-only
        "state_obs_keys": ["state"],
        "state_encoding": 1,
        "action_encoding": 1,
    }


class InteractionMapHead(torch.nn.Module):
    """2.5D Interaction Map prediction head.

    Predicts action-conditioned affordance map with depth binning.
    Output: A_t(u,v,b) - probability that location (u,v) at depth bin b
    is a feasible contact site for the current action step.
    """

    def __init__(self, input_dim: int = 1024, num_depth_bins: int = 4, hidden_dim: int = 512):
        super().__init__()
        self.num_depth_bins = num_depth_bins

        # Project visual features to interaction map
        self.feature_proj = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(0.1),
        )

        # Predict 2.5D interaction map
        # Output: H x W x num_depth_bins (e.g., 14 x 14 x 4)
        self.spatial_size = 14  # Typical ViT patch grid size (224/16)
        self.interaction_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, self.spatial_size * self.spatial_size * num_depth_bins),
        )

    def forward(self, visual_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            visual_features: [B, N, D] visual tokens from backbone
        Returns:
            interaction_map: [B, H, W, num_depth_bins] 2.5D affordance map
        """
        B = visual_features.size(0)

        # Use CLS token or average pool spatial tokens
        if visual_features.size(1) > 1:
            # Average pool all spatial tokens
            features = visual_features.mean(dim=1)
        else:
            features = visual_features.squeeze(1)

        # Project features
        features = self.feature_proj(features)

        # Predict interaction map
        logits = self.interaction_head(features)
        logits = logits.view(B, self.spatial_size, self.spatial_size, self.num_depth_bins)

        # Apply softmax over spatial + depth dimensions
        B, H, W, D = logits.shape
        logits_flat = logits.view(B, H * W * D)
        probs_flat = F.softmax(logits_flat, dim=-1)
        interaction_map = probs_flat.view(B, H, W, D)

        return interaction_map


class DistillationStudent(torch.nn.Module):
    """RGB student with 2.5D interaction map distillation."""

    def __init__(self, base_model, num_depth_bins: int = 4, lambda_kl: float = 0.5):
        super().__init__()
        self.base_model = base_model
        self.num_depth_bins = num_depth_bins
        self.lambda_kl = lambda_kl

        # Get hidden dimension from base model
        # Assuming LLaMA-7B hidden dim = 4096, but visual features are typically smaller
        self.hidden_dim = 1024  # DINOv2 ViT-L/14 feature dim

        # 2.5D interaction map head
        self.interaction_head = InteractionMapHead(
            input_dim=self.hidden_dim,
            num_depth_bins=num_depth_bins,
        )

    def forward(self, images, input_ids, teacher_interaction_map=None, **kwargs):
        """Forward pass with optional distillation target."""
        # TODO: Extract visual features from base model
        # This requires modifying the base model to return intermediate features

        # For now, placeholder forward
        outputs = self.base_model(images=images, input_ids=input_ids, **kwargs)

        # TODO: Extract visual features and predict interaction map
        # visual_features = self.extract_visual_features(images)
        # student_interaction_map = self.interaction_head(visual_features)

        # Compute losses
        losses = {}
        if teacher_interaction_map is not None:
            # KL distillation loss
            # student_map = student_interaction_map
            # kl_loss = F.kl_div(
            #     torch.log(student_map + 1e-8),
            #     teacher_interaction_map,
            #     reduction='batchmean'
            # )
            # losses['kl_loss'] = kl_loss
            pass

        return outputs, losses

    def extract_interaction_map(self, images):
        """Extract 2.5D interaction map for visualization/evaluation."""
        # TODO: Implement feature extraction
        # This would be used to visualize what the model learned
        pass


def compute_distillation_loss(
    student_map: torch.Tensor,
    teacher_map: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Compute KL divergence loss between student and teacher interaction maps.

    Args:
        student_map: [B, H, W, D] student interaction map (probabilities)
        teacher_map: [B, H, W, D] teacher interaction map (probabilities)
        temperature: Temperature for softening distributions

    Returns:
        kl_loss: KL divergence loss
    """
    B, H, W, D = student_map.shape

    # Flatten spatial and depth dimensions
    student_flat = student_map.view(B, H * W * D)
    teacher_flat = teacher_map.view(B, H * W * D)

    # Apply temperature
    student_log_probs = F.log_softmax(torch.log(student_flat + 1e-8) / temperature, dim=-1)
    teacher_probs = F.softmax(torch.log(teacher_flat + 1e-8) / temperature, dim=-1)

    # KL divergence
    kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')

    return kl_loss


def train_distillation(cfg: DistillConfig):
    """Train Arm B or C with 2.5D distillation."""
    # Setup distributed training
    setup_distributed(cfg)

    # Set seed
    torch.manual_seed(cfg.seed)

    # Create output directory
    os.makedirs(cfg.output_dir, exist_ok=True)

    # Load base model
    arm_name = "Arm B" if cfg.arm.lower() == "b" else "Arm C"
    print(f"Training {arm_name}: RGB + 2.5D distillation")
    print(f"Teacher: {cfg.teacher_checkpoint}")
    print(f"Teacher type: {cfg.teacher_type}")
    print(f"Lambda KL: {cfg.lambda_kl}")
    print(f"Num depth bins: {cfg.num_depth_bins}")

    base_model = AutoModelForVision2Seq.from_pretrained(
        cfg.base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(cfg.base_model, trust_remote_code=True)

    # Wrap with distillation
    model = DistillationStudent(
        base_model,
        num_depth_bins=cfg.num_depth_bins,
        lambda_kl=cfg.lambda_kl,
    )

    # Load teacher model
    print(f"Loading teacher from {cfg.teacher_checkpoint}")
    # TODO: Load teacher checkpoint
    # teacher_model = ...

    # Move to GPU
    device = torch.device(f"cuda:{cfg.local_rank}")
    model = model.to(device)

    # Wrap with DDP if distributed
    if cfg.world_size > 1:
        model = DDP(model, device_ids=[cfg.local_rank])

    # Create dataloader
    batch_transform = RLDSBatchTransform(
        action_tokenizer=processor.action_tokenizer,
        image_transform=processor.image_transform,
        prompt_builder_fn=processor.prompt_builder_fn,
    )

    dataset_kwargs = get_libero_dataset_kwargs(cfg.dataset_name, cfg.data_dir)
    dataset = RLDSDataset(
        data_dir=cfg.data_dir,
        dataset_kwargs=[dataset_kwargs],
        batch_transform=batch_transform,
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        train=True,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=4,
        pin_memory=True,
        collate_fn=lambda x: x,
    )

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    # Training loop
    print(f"Starting distillation training for {cfg.num_epochs} epochs")
    global_step = 0
    model.train()

    for epoch in range(cfg.num_epochs):
        for step, batch in enumerate(dataloader):
            # TODO: Implement training loop
            # 1. Get teacher interaction map (no grad)
            # 2. Get student prediction
            # 3. Compute action loss + KL loss
            # 4. Backprop

            global_step += 1

            if global_step % cfg.log_steps == 0 and cfg.local_rank == 0:
                print(f"Epoch {epoch}, Step {global_step}")

    # Save final checkpoint
    if cfg.local_rank == 0:
        final_path = os.path.join(cfg.output_dir, "final")
        print(f"Saving final checkpoint to {final_path}")

    # Cleanup
    if cfg.world_size > 1:
        dist.destroy_process_group()

    print(f"{arm_name} training complete!")


@draccus.wrap()
def main(cfg: DistillConfig):
    """Main entry point."""
    train_distillation(cfg)


if __name__ == "__main__":
    main()
