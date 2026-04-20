#!/bin/bash
# Phase 2 Gate 1 Evaluation: Arm A vs Arm D on LIBERO frozen 8-task slice
# Uses run_libero_eval_fast.py (hard-coded FROZEN_TASKS: G1-G6 + E1-E2)
# 20 trials per task. Seed 1.

set -e
source /home/hurricane/miniconda3/etc/profile.d/conda.sh
conda activate openvla-oft
cd /home/hurricane/VLA/openvla-oft
export PYTHONPATH=${PYTHONPATH}:LIBERO:.

CKPT_ROOT=/home/hurricane/nvme0/vla_checkpoints
OUT_DIR=./eval_results/phase2_gate1_v2
mkdir -p "$OUT_DIR"

run_eval () {
  local arm=$1          # arm_a | arm_d
  local suite=$2        # libero_spatial | libero_goal
  local model_dir=$3    # arm_a_spatial_v2 | arm_a_goal_v2 | arm_d_spatial_v2 | arm_d_goal_v2
  local num_images=$4   # 1 for A, 2 for D

  echo ""
  echo "==============================================="
  echo "Evaluating $arm on $suite (20 trials)"
  echo "  model: $CKPT_ROOT/$model_dir/merged"
  echo "  started: $(date)"
  echo "==============================================="

  python experiments/robot/libero/run_libero_eval_fast.py \
    --model_path "$CKPT_ROOT/$model_dir/merged" \
    --task_suite "$suite" \
    --model_name "$arm" \
    --num_images_in_input "$num_images" \
    --lora_rank 32 \
    --num_trials 20 \
    --seed 1 \
    --output_dir "$OUT_DIR" \
    2>&1 | tee -a "$OUT_DIR/${arm}_${suite}.log"

  echo "$arm $suite done at: $(date)"
}

# Order: D first (multi-view, more interesting), then A
run_eval arm_d libero_spatial arm_d_spatial_v2 2
run_eval arm_d libero_goal    arm_d_goal_v2    2
run_eval arm_a libero_spatial arm_a_spatial_v2 1
run_eval arm_a libero_goal    arm_a_goal_v2    1

echo ""
echo "==============================================="
echo "All Phase 2 Gate 1 evaluations complete."
echo "Results:"
ls -la "$OUT_DIR"/*.json
echo "==============================================="
