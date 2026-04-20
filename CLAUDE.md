# CLAUDE.md — VLA Geometry-Critical Failure Diagnosis

## Pipeline Status
- **Current Phase**: Phase 2 → Phase 3 过渡（Gate 1 PASS dirty）
- **Blocker**: Arm A 训练异常（全 0% 成功率），Phase 3 前必须修复
- **Next Milestone**: 修复 Arm A 训练 → 重跑 Gate 1 → 启动 B/C 蒸馏
- **Gate 1 结果 (2026-04-20)**: Arm D G1-G6 mean=79.2% vs Arm A 0%，gap +79.2 pts（门槛 5 pts）

## Remote Server
- **gpu**: remote
- **SSH**: `ssh 192.168.0.214`
- **GPU**: 1x RTX 3090Ti
- **Conda**: `eval "$(/home/hurricane/miniconda3/bin/conda shell.bash hook)" && conda activate openvla-oft`
- **Code dir**: `/home/hurricane/VLA/openvla-oft`
- **code_sync**: git
- **wandb**: false

## Project Constraints
- **Base VLA**: OpenVLA-OFT
- **Benchmark**: LIBERO (libero_spatial + libero_goal)
- **Budget**: ~1,090 GPUh
- **Hardware**: 1x RTX 3090Ti, ~20-25 days
- **Target Venue**: CoRL (strong accept primary, spotlight stretch, poster floor)

## Local Code
- **Path**: `code/openvla-oft/`
- **Remote**: `https://github.com/hyyyyyyz/openvla-oft.git`

## Experiment Overview
- **Method**: Post-training OpenVLA-OFT with privileged multi-view distillation
- **Teacher**: Multi-view RGB (static + wrist camera) privileged (Arm D, training only)
- **Student**: Single-view RGB only (static camera) (Arm A-C, inference)
- **Key Innovation**: Multi-view geometry fusion for geometry-critical failures

## Frozen LIBERO 8-Task Slice

### Geometry-Critical Tasks (G1-G6)
| Slot | Task Name | Suite |
|------|-----------|-------|
| G1 | pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | libero_spatial |
| G2 | pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | libero_spatial |
| G3 | pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | libero_spatial |
| G4 | put_the_wine_bottle_on_top_of_the_cabinet | libero_goal |
| G5 | open_the_top_drawer_and_put_the_bowl_inside | libero_goal |
| G6 | put_the_cream_cheese_in_the_bowl | libero_goal |

### Easy Control Tasks (E1-E2)
| Slot | Task Name | Suite |
|------|-----------|-------|
| E1 | pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | libero_spatial |
| E2 | pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | libero_spatial |

## Phase Gates

- **Phase 0**: Task Freeze + Engineering (T0-T4) ✅ COMPLETED
- **Phase 1**: Pilot (~60 GPUh) - E0 sanity check ✅ COMPLETED
- **Phase 2**: A/D Baselines (~150 GPUh) - Gate 1 🟢 PASS dirty（Arm A 需修复）
- **Phase 3**: B/C Distillation (~400 GPUh) - Gate 2: Recovery >=30%
- **Phase 4**: Anchor + Mechanism (~380 GPUh)
- **Phase 5**: Real Robot (~100 GPUh)

## Arm Definitions

| Arm | Input | Description |
|-----|-------|-------------|
| Arm A | Static RGB only | Baseline (single-view) |
| Arm D | Static RGB + Wrist RGB | Teacher (multi-view privileged) |
| Arm B/C | Static RGB only | Distilled from Arm D |

## Key Files
- `refine-logs/EXPERIMENT_PLAN.md` - Full experiment roadmap
- `refine-logs/FINAL_PROPOSAL.md` - Method description
- `refine-logs/EXPERIMENT_TRACKER.md` - Execution status
- `experiments/geometry_distill/train_multiview_teacher.py` - Arm D training
- `experiments/geometry_distill/train_rgb_baseline.py` - Arm A training

## Notes
- ✅ Phase 0 T3 (Multi-view teacher path) - COMPLETED
- Multi-view fusion uses wrist camera for close-up geometry
- Batch size locked at 1 for 24GB GPU
- Seeds: 1 for screening, 3 for final claims
