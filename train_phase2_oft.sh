#!/bin/bash
# Phase 2 retraining with the OFFICIAL OpenVLA-OFT recipe (per LIBERO.md).
#
# Rationale (2026-04-21 pivot):
#   Previous legacy-token runs (train_multiview_teacher_official.py) did not
#   converge for single-view Arm A after two 150K-step attempts (v2 final loss
#   3.65, v3 final loss 3.85; both evaluated at 0% success on every frozen
#   task). Arm D (num_images=2) converged under the same legacy path only
#   because the wrist camera provided a strong visual signal. The official
#   LIBERO.md mandates the OFT recipe (L1 regression action head +
#   image_aug + lr decay at 100K) for both single- and multi-view training.
#
# This script retrains BOTH arms under the official OFT recipe so they are
# architecturally symmetric and directly comparable:
#   Arm D OFT: num_images_in_input=2  (static + wrist)
#   Arm A OFT: num_images_in_input=1  (static only)
# Outputs go to arm_{a,d}_{spatial,goal}_v4 on NVMe.
#
# Checkpoint cadence matches the Arm D v2 run (save every 10K). Per LIBERO.md:
#   - spatial / 150K steps best
#   - goal / 50K steps best
#
# W&B is disabled globally via WANDB_MODE=disabled (CLAUDE.md wandb: false).

set -e
set -o pipefail   # so `torchrun ... | tee` fails loudly instead of silently
source /home/hurricane/miniconda3/etc/profile.d/conda.sh
conda activate openvla-oft
cd /home/hurricane/VLA/openvla-oft
export PYTHONPATH=${PYTHONPATH}:LIBERO:.
export WANDB_MODE=disabled
# Force HF to use local cache only; the server's connection to huggingface.co
# is flaky and causes spurious SSLEOFError at processor_config.json fetch.
# Cache under ~/.cache/huggingface/hub/models--openvla--openvla-7b/ is complete.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

CKPT_ROOT=/home/hurricane/nvme0/vla_checkpoints
mkdir -p "$CKPT_ROOT"

train_oft () {
  local arm=$1           # arm_a | arm_d
  local suite=$2         # libero_spatial | libero_goal
  local max_steps=$3     # 150005 | 50000
  local num_images=$4    # 1 for Arm A, 2 for Arm D
  local out_name="${arm}_${suite#libero_}_v4"
  local out_dir="$CKPT_ROOT/$out_name"
  local log_file="${out_name}.log"

  echo ""
  echo "==============================================="
  echo "[OFT] $out_name : $suite, num_images=$num_images, max_steps=$max_steps"
  echo "  out: $out_dir"
  echo "  started: $(date)"
  echo "==============================================="

  torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
    --vla_path openvla/openvla-7b \
    --data_root_dir /home/hurricane/VLA/modified_libero_rlds \
    --dataset_name "${suite}_no_noops" \
    --run_root_dir "$out_dir" \
    --use_l1_regression True \
    --use_diffusion False \
    --use_film False \
    --num_images_in_input "$num_images" \
    --use_proprio False \
    --batch_size 1 \
    --learning_rate 5e-4 \
    --num_steps_before_decay 100000 \
    --max_steps "$max_steps" \
    --save_freq 10000 \
    --save_latest_checkpoint_only False \
    --image_aug True \
    --lora_rank 32 \
    --merge_lora_during_training False \
    --wandb_entity "disabled" \
    --wandb_project "disabled" \
    --run_id_note "oft_num_images_${num_images}" \
    2>&1 | tee -a "$log_file"

  echo "$out_name done at: $(date)"
  python -c "import torch; torch.cuda.empty_cache()" || true
}

# Teacher first (confirms the whole pipeline works before spending GPU on student)
train_oft arm_d libero_spatial 150005 2
train_oft arm_d libero_goal    50000  2

# Student (the one that previously failed under legacy recipe)
train_oft arm_a libero_spatial 150005 1
train_oft arm_a libero_goal    50000  1

echo ""
echo "==============================================="
echo "Phase 2 OFT training (LoRA-only checkpoints) complete."
echo "  merge_lora_during_training was disabled to avoid OOM during save;"
echo "  now merge the four final LoRA adapters into the base model offline:"
echo ""
for arm in arm_d arm_a; do
  for suite in spatial goal; do
    runid_dir=$(ls -d $CKPT_ROOT/${arm}_${suite}_v4/openvla-7b*oft_num_images_* 2>/dev/null | head -1)
    if [ -n "$runid_dir" ]; then
      final_chkpt=$(ls -d ${runid_dir}--*_chkpt 2>/dev/null | sort -V | tail -1)
      if [ -n "$final_chkpt" ]; then
        echo "Merging $final_chkpt..."
        python vla-scripts/merge_lora_weights_and_save.py \
          --base_checkpoint openvla/openvla-7b \
          --lora_finetuned_checkpoint_dir "$final_chkpt"
      fi
    fi
  done
done
echo ""
echo "All merges complete. Next: rerun Gate 1 evaluation (run_phase2_gate1_eval.sh)"
echo "==============================================="
