#!/bin/bash
# F1 smoke: Arm A OFT, grad_accumulation_steps=4, use_proprio=False, 5K effective steps.
#
# Goal: cheap fail-fast on whether OFT with grad-accum can converge under
# bs=1 + single-view + no proprio on 3090Ti.
#
# Fail-fast cutoffs (mean loss over last few hundred logs):
#   step 1000 effective steps still >= 0.25 horizontal -> dead
#   step 3000 effective steps still >= 0.20 / no clear downward slope -> dead
#   step 5000 effective steps <= 0.12 with downward slope -> proceed to full run
#
# Output dir: arm_a_spatial_v5_smoke (so it does not collide with v4 dirs).

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
OUT_DIR=$CKPT_ROOT/arm_a_spatial_v5_smoke
mkdir -p "$CKPT_ROOT"

echo "==============================================="
echo "F1 smoke: Arm A OFT, grad_accum=4, 5K effective steps"
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
  --grad_accumulation_steps 4 \
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
  --run_id_note "f1_smoke_grad_accum_4_no_proprio" \
  2>&1 | tee smoke_f1.log

echo ""
echo "==============================================="
echo "F1 smoke done at: $(date)"
echo "Decision: read [train] step=N loss=L lines from smoke_f1.log."
echo "==============================================="
