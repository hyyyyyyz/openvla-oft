#!/bin/bash
# Retrain Arm A (single-view RGB baseline) after the v2 run produced 0% success
# on every Gate 1 task.
#
# Strategy: reuse train_multiview_teacher_official.py (the script that produced
# the working Arm D checkpoints) with --num_images_in_input 1. That path calls
# vision_backbone.set_num_images_in_input(1) explicitly and turns off wrist
# image loading (use_wrist_image = num_images_in_input > 1).
#
# Steps mirror Arm D v2: 150K for spatial, 50K for goal.
# save_freq=10000 to keep disk use bounded (~70G/arm worst case).

set -e
source /home/hurricane/miniconda3/etc/profile.d/conda.sh
conda activate openvla-oft
cd /home/hurricane/VLA/openvla-oft

CKPT_ROOT=/home/hurricane/nvme0/vla_checkpoints
mkdir -p "$CKPT_ROOT"

train_arm_a () {
  local suite=$1       # libero_spatial | libero_goal
  local max_steps=$2   # 150000 | 50000
  local out_name=$3    # arm_a_spatial_v3 | arm_a_goal_v3
  local resume_step=${4:-0}   # 0 = fresh; >0 = resume from that checkpoint

  local out_dir="$CKPT_ROOT/$out_name"

  echo ""
  echo "==============================================="
  echo "Training $out_name on $suite ($max_steps steps, single-view, resume_step=$resume_step)"
  echo "  out: $out_dir"
  echo "  started: $(date)"
  echo "==============================================="

  local resume_args=""
  if [ "$resume_step" -gt 0 ]; then
    resume_args="--resume_from_checkpoint True --resume_step $resume_step"
  fi

  python experiments/geometry_distill/train_multiview_teacher_official.py \
    --vla_path openvla/openvla-7b \
    --dataset_name "${suite}_no_noops" \
    --data_root_dir /home/hurricane/VLA/modified_libero_rlds \
    --num_images_in_input 1 \
    --batch_size 1 \
    --learning_rate 5e-4 \
    --max_steps "$max_steps" \
    --save_freq 10000 \
    --log_freq 50 \
    --lora_rank 32 \
    --seed 1 \
    $resume_args \
    --run_root_dir "$out_dir" \
    2>&1 | tee -a "${out_name}.log"

  echo "$out_name done at: $(date)"

  # Free GPU memory between runs
  python -c "import torch; torch.cuda.empty_cache()" || true
}

# Power outage history:
#   2026-04-20 20:01  first v3 interrupted at step ~21200 (checkpoint-20000 saved)
#   2026-04-21 09:43  fresh v3 interrupted at step ~102850 (checkpoint-100000 saved)
# Spatial resumes from the last saved checkpoint (100000). Change the 4th arg
# back to 0 for a clean-slate run.
train_arm_a libero_spatial 150000 arm_a_spatial_v3 100000
train_arm_a libero_goal    50000  arm_a_goal_v3    0

echo ""
echo "==============================================="
echo "Arm A v3 retraining complete."
echo "Next: merge LoRA adapters and rerun Gate 1 evaluation."
echo "==============================================="
