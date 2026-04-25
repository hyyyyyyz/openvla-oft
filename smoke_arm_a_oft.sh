#!/bin/bash
# Z smoke: Arm A OFT, grad_accumulation_steps=8, use_proprio=False, 5K effective steps.
#
# Why Z (after F1 with grad_accum=4 plateau'd at 0.19):
#   F1 cut loss 47% vs v4 baseline but plateau'd before reaching 0.12 target.
#   Z doubles effective batch (8 vs 4) to test whether batch is the main
#   bottleneck. Decision rule (per codex):
#     - Z 5K mean <= 0.16 with continued downward slope -> proceed to Y (full retrain)
#     - Z 5K still >= 0.18 plateau -> escalate to X (add use_proprio=True)
#     - 0.16-0.18 still trending down -> Y but treat as engineering bet
#
# Output dir: arm_a_spatial_v5_smoke_g8

set -e
set -o pipefail
source /home/hurricane/miniconda3/etc/profile.d/conda.sh
conda activate openvla-oft
cd /home/hurricane/VLA/openvla-oft
export PYTHONPATH=${PYTHONPATH}:LIBERO:.
export WANDB_MODE=disabled
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

CKPT_ROOT=/home/hurricane/nvme0/vla_checkpoints
OUT_DIR=$CKPT_ROOT/arm_a_spatial_v5_smoke_g8
mkdir -p "$CKPT_ROOT"

echo "==============================================="
echo "Z smoke: Arm A OFT, grad_accum=8, 5K effective steps"
echo "  out: $OUT_DIR"
echo "  started: $(date)"
echo "==============================================="

torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir /home/hurricane/VLA/modified_libero_rlds \
  --dataset_name libero_spatial_no_noops \
  --run_root_dir "$OUT_DIR" \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film False \
  --num_images_in_input 1 \
  --use_proprio False \
  --batch_size 1 \
  --grad_accumulation_steps 8 \
  --learning_rate 5e-4 \
  --num_steps_before_decay 100000 \
  --max_steps 5000 \
  --save_freq 5000 \
  --save_latest_checkpoint_only False \
  --image_aug True \
  --lora_rank 32 \
  --merge_lora_during_training False \
  --wandb_entity "disabled" \
  --wandb_project "disabled" \
  --run_id_note "z_smoke_grad_accum_8_no_proprio" \
  2>&1 | tee smoke_z.log

echo ""
echo "==============================================="
echo "Z smoke done at: $(date)"
echo "==============================================="
