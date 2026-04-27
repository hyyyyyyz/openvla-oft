#!/bin/bash
# Y full retrain via resume from Y' ckpt-25000.
# Y' (max_steps=30K) crashed at step ~27800 due to suspected host RAM/TF
# pipeline pressure (no sudo dmesg, server lost SSH for ~2h).
#
# Codex prescription:
#   1) merge ckpt-25000 lora_adapter into base FIRST -> done before this script
#   2) num_steps_before_decay=75000 so lr decay fires at effective step 100K
#      (resume_step + 75K = 100K, matching original lr decay schedule)
#   3) shuffle_buffer_size=20000 (default 100000 is the most likely RAM killer
#      per the comment in vla-scripts/finetune.py:77)
#   4) save_freq=2500 (cap re-crash loss to ~30 min)
# Output dir: arm_a_spatial_v6 (same as Y'); finetune.py creates a new run_id
# subdir keyed by --run_id_note, so this resume run does not collide with Y'.
#
# After this completes (or 150K steps reached), final LoRA is merged offline
# via vla-scripts/merge_lora_weights_and_save.py.

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
RESUME_DIR="$CKPT_ROOT/arm_a_spatial_v6/openvla-7b+libero_spatial_no_noops+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--v6_y_prime_grad_accum_8_no_proprio_30k--25000_chkpt"
OUT_DIR=$CKPT_ROOT/arm_a_spatial_v6

if [ ! -f "$RESUME_DIR/model-00001-of-00004.safetensors" ]; then
  echo "ERROR: no merged model found in $RESUME_DIR. Run merge_lora_weights_and_save.py first." >&2
  exit 1
fi

echo "==============================================="
echo "Y resume: Arm A spatial from ckpt-25000 -> 150K"
echo "  resume_dir: $RESUME_DIR"
echo "  out: $OUT_DIR"
echo "  started: $(date)"
echo "==============================================="

torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path "$RESUME_DIR" \
  --data_root_dir /home/hurricane/VLA/modified_libero_rlds \
  --dataset_name libero_spatial_no_noops \
  --run_root_dir "$OUT_DIR" \
  --resume True \
  --resume_step 25000 \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film False \
  --num_images_in_input 1 \
  --use_proprio False \
  --batch_size 1 \
  --grad_accumulation_steps 8 \
  --learning_rate 5e-4 \
  --num_steps_before_decay 75000 \
  --max_steps 150000 \
  --shuffle_buffer_size 20000 \
  --save_freq 2500 \
  --save_latest_checkpoint_only False \
  --image_aug True \
  --lora_rank 32 \
  --merge_lora_during_training False \
  --wandb_entity "disabled" \
  --wandb_project "disabled" \
  --run_id_note "v6_resume_from_25k_to_150k" \
  2>&1 | tee train_v6_resume.log

echo ""
echo "==============================================="
echo "Y resume done at: $(date)"
echo "Run merge_lora_weights_and_save.py on the final checkpoint, then re-run"
echo "  run_phase2_gate1_eval.sh."
echo "==============================================="
