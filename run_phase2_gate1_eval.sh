#!/bin/bash
# Phase 2 Gate 1 Evaluation using the official run_libero_eval.py.
# Evaluates Arm A vs Arm D on the frozen 8-task slice:
#   libero_spatial: G1=0, E2=2, G2=4, G3=7, E1=9
#   libero_goal:    G4=2, G5=3, G6=6
# 20 trials per task. Seed 1.

set -e
source /home/hurricane/miniconda3/etc/profile.d/conda.sh
conda activate openvla-oft
cd /home/hurricane/VLA/openvla-oft
export PYTHONPATH=${PYTHONPATH}:LIBERO:.

CKPT_ROOT=/home/hurricane/nvme0/vla_checkpoints
OUT_DIR=./eval_results/phase2_gate1
mkdir -p "$OUT_DIR"

SPATIAL_IDS="0,2,4,7,9"   # G1, E2, G2, G3, E1
GOAL_IDS="2,3,6"          # G4, G5, G6

run_eval () {
  local arm=$1          # arm_a | arm_d
  local suite=$2        # libero_spatial | libero_goal
  local model_dir=$3    # arm_{a,d}_{spatial,goal}_v2
  local num_images=$4   # 1 for A, 2 for D
  local task_ids=$5

  echo ""
  echo "==============================================="
  echo "Evaluating $arm on $suite (tasks=$task_ids, 20 trials)"
  echo "  model: $CKPT_ROOT/$model_dir/merged"
  echo "  started: $(date)"
  echo "==============================================="

  python experiments/robot/libero/run_libero_eval.py \
    --model_family openvla \
    --pretrained_checkpoint "$CKPT_ROOT/$model_dir/merged" \
    --num_images_in_input "$num_images" \
    --task_suite_name "$suite" \
    --task_ids "$task_ids" \
    --num_trials_per_task 20 \
    --lora_rank 32 \
    --use_proprio False \
    --use_l1_regression False \
    --use_diffusion False \
    --seed 1 \
    --local_log_dir "$OUT_DIR/${arm}_${suite}" \
    2>&1 | tee "$OUT_DIR/${arm}_${suite}.log"

  echo "$arm $suite done at: $(date)"
}

# Arm D (multi-view teacher) first — more interesting
run_eval arm_d libero_spatial arm_d_spatial_v2 2 "$SPATIAL_IDS"
run_eval arm_d libero_goal    arm_d_goal_v2    2 "$GOAL_IDS"
run_eval arm_a libero_spatial arm_a_spatial_v2 1 "$SPATIAL_IDS"
run_eval arm_a libero_goal    arm_a_goal_v2    1 "$GOAL_IDS"

echo ""
echo "==============================================="
echo "All Phase 2 Gate 1 evaluations complete."
echo "  Output root: $OUT_DIR"
echo "Run aggregation next:"
echo "  python experiments/geometry_distill/aggregate_gate1.py --eval_dir $OUT_DIR"
echo "==============================================="
