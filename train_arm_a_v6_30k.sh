#!/bin/bash
# Y' staged retrain: Arm A spatial only, grad_accum=8, use_proprio=False, 30K effective steps.
#
# Rationale (codex Y' decision after F1 + Z smoke):
#   F1 (grad_accum=4) plateau'd at 0.19; Z (grad_accum=8) reached 0.16 with
#   slope drop only 0.0024 over last 1K steps - not "明显下降". Going straight
#   to a 7-day Y full retrain is over-betting on insufficient evidence. Y'
#   buys mid-trajectory info (30K eff steps, ~33h) before deciding whether to
#   continue to 150K.
#
#   Decision after 30K:
#     - loss still trending down (e.g. 25K-30K bin mean clearly < 20K-25K mean)
#       -> continue training to 150K (resume from checkpoint)
#     - flat plateau >= 0.13 with no slope -> stop, switch to X (proprio) or
#       legacy C1 path
#
# This run does NOT touch lr decay (decay@100k > our max_steps 30k), so we are
# observing the constant-lr regime only. If we continue to 150K, lr_decay will
# kick in later and may give an extra step down.
#
# Output dir: arm_a_spatial_v6 (NVMe). Checkpoint every 5K steps.

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
OUT_DIR=$CKPT_ROOT/arm_a_spatial_v6
mkdir -p "$CKPT_ROOT"

echo "==============================================="
echo "Y' staged retrain: Arm A spatial v6, grad_accum=8, 30K eff steps"
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
  --max_steps 30000 \
  --save_freq 5000 \
  --save_latest_checkpoint_only False \
  --image_aug True \
  --lora_rank 32 \
  --merge_lora_during_training False \
  --wandb_entity "disabled" \
  --wandb_project "disabled" \
  --run_id_note "v6_y_prime_grad_accum_8_no_proprio_30k" \
  2>&1 | tee train_v6_y_prime.log

echo ""
echo "==============================================="
echo "Y' done at: $(date)"
echo "Decision: read [train] step=N loss=L lines from train_v6_y_prime.log"
echo "  - 25-30K bin mean clearly < 20-25K bin mean -> continue to 150K"
echo "  - flat plateau >= 0.13 -> stop, escalate to X (proprio) or C1 (legacy)"
echo "==============================================="
