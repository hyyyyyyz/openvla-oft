# Experiment Plan: Geometry-Critical Failure Diagnosis via Privileged Multi-View Distillation

**Date**: 2026-04-11 (Updated)
**Status**: Phase 2 IN PROGRESS — Multi-view teacher implemented
**Method**: Post-training OpenVLA-OFT to diagnose and repair geometry-critical failures via multi-view fusion distillation from privileged teacher (static + wrist camera)
**Benchmark**: LIBERO — libero_spatial + libero_goal (official suite-level + frozen 8-task diagnosis slice)
**Target**: CoRL strong accept (primary), spotlight stretch, poster floor
**Compute**: 1x RTX 3090Ti, ~20-25 days
**Budget**: ~1,090 GPUh (locked)

---

## Code Reality: Benchmark & Engineering Constraints

### Official Suite-Level Evaluation (现成支持)
`OpenVLA-OFT` 官方 `run_libero_eval.py` 按 suite 运行：
- `libero_spatial` (10 tasks), `libero_object` (10), `libero_goal` (10), `libero_10` (10)
- 每个 suite 单独跑，无需修改 evaluator
- `run_libero_eval.py` 默认遍历该 suite 所有 task，无需自定义 evaluator

**主评测证据来自 libero_spatial + libero_goal**（libero_object 偏语义，libero_10 偏 long-horizon，均不适合作为 geometry-critical diagnosis 的主要证据）。

### Frozen 8-Task Diagnosis Slice (需 minimal offline wrapper)
跨 suite 的 8-task frozen slice 不是 official evaluator 原生能力：
1. 每个 suite 单独跑官方 evaluator → 拿到 per-task 结果
2. 一个离线 Python 脚本按 task name 筛选并聚合到 diagnosis bucket（G1-G6 geometry-critical + E1-E2 easy）
3. 无需修改 evaluator 本身

### Multi-View Teacher: Engineering Solution (Phase 0 T3 COMPLETED)

**方案调整** (2026-04-11): 从 RGB-D 改为 Multi-view RGB

原方案使用 depth 数据作为 privileged geometry，但 `modified_libero_rlds` 数据集不包含 depth。
经分析 OXE 配置，发现数据集包含 `wrist_image`（腕部相机）。

**新方案**:
- **Arm D (Teacher)**: Static RGB (`image`) + Wrist RGB (`wrist_image`) 多视角融合
- **Arm A (Student)**: Static RGB (`image`) 单视角
- **优势**: 
  - Wrist camera 提供近距离几何信息，对抓取任务至关重要
  - 数据已存在，无需重新渲染
  - 实现简单，修改 configs.py 即可启用

**Implementation**:
- `OXE_DATASET_CONFIGS`: 已配置 `wrist_image` key
- `RLDSBatchTransform`: 启用 `num_images_in_input=2`
- Model forward: 添加 multi-view fusion 分支（Arm D）

---

## Experimental Protocol

### Fixed Constants (Locked)
- **Base VLA**: OpenVLA-OFT checkpoint, same across all arms
- **Teacher**: Multi-view RGB privileged (Arm D: static + wrist), frozen during student training
- **Student**: Arms A-C, single-view RGB only (inference), add lightweight affordance head
- **Multi-view fusion**: Concatenate static + wrist features at vision backbone output
- **Seeds**: 1 for screening, 3 for final claims on critical comparisons
- **Hyperparameters**: λ locked after validation pilot (no mid-experiment tuning)

### Fixed LIBERO 8-Task Slice (Official Task Names, Frozen)

**Source**: libero_spatial + libero_goal official task maps.

**Geometry-Critical Tasks (6)**:

| Slot | Official Task Name | Suite | Failure Type |
|------|-------------------|-------|-------------|
| G1 | `pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate` | libero_spatial | Support ambiguity + occluder proximity |
| G2 | `pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate` | libero_spatial | Partial-occlusion region |
| G3 | `pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate` | libero_spatial | Narrow-surface placement |
| G4 | `put_the_wine_bottle_on_top_of_the_cabinet` | libero_goal | Occluded target, tall placement |
| G5 | `open_the_top_drawer_and_put_the_bowl_inside` | libero_goal | Partial-occlusion container |
| G6 | `put_the_cream_cheese_in_the_bowl` | libero_goal | Depth-order-sensitive placement |

**Easy Control Tasks (2)**:

| Slot | Official Task Name | Suite | Why Easy |
|------|-------------------|-------|----------|
| E1 | `pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate` | libero_spatial | Fully visible target surface |
| E2 | `pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate` | libero_spatial | Object and target fully visible, open scene |

**⚠️ HARD STOP**: 这 8 个 task ID 在 Phase 0 冻结后不得更改。

### Evaluation Framework
- **Primary metric**: Episode success on LIBERO
- **Stress splits**: partial occluder, camera jitter, extra distractor — applied to G1-G6 + E1-E2
- **Secondary metrics**: recovered privileged gap (Arm B vs D), robustness drop under stress, affordance KL, inference latency
- **Diagnosis metrics**: Rollout-level failure taxonomy (5 classes: semantic target · geometric contact-site · approach collision · invalid placement · success); bucket × error-type heatmap

---

## 1. Main Anchor Result

**Experiment**: 4-arm comparison on geometry-critical tasks

| Arm | Description | Input | Purpose |
|:---:|-------------|-------|---------|
| A | RGB-only OpenVLA-OFT baseline | Static RGB | Lower bound |
| B | RGB-only OpenVLA-OFT + multi-view distillation from privileged teacher | Static RGB | **Main contribution: geometry-critical failure repair** |
| C | RGB-only OpenVLA-OFT + same structured target from RGB-only teacher | Static RGB | Control: privileged multi-view vs teacher smoothing |
| D | Multi-view RGB OpenVLA-OFT privileged upper bound | Static RGB + Wrist RGB | Upper bound |

**Success Criteria**:
- B beats A by ≥8 points on geometry-critical buckets
- B beats C by ≥4 points
- B recovers ≥50% of (D - A) gap
- ≤5% latency overhead vs A
- Little change on `easy` tasks

**Purpose**: Prove core thesis (Claims 1, 2, 4)

---

## 2. Novelty Isolation

| Experiment | Isolates | Purpose |
|------------|----------|---------|
| Privileged vs RGB teacher | Privileged geometry vs teacher smoothing | Claim 1 |
| Dense affordance vs final-action/logit KD | Dense supervision vs generic KD | Claim 1, 3 |
| Dense affordance vs depth auxiliary | Action-conditioned structure vs generic geometry | Claim 3 |
| 2D heatmap vs 2.5D bins | Coarse depth necessity | Claim 3 |

**Interpretation**: If any control collapses the gap, narrow the claim.

---

## 3. Simplicity/Deletion Check

**Minimal Design**:
- One auxiliary head on visual tokens
- 4-8 coarse depth bins
- No 3D reconstruction
- No 6D pose labels
- No extra cameras/depth at test time
- No extra decoder or memory

**Deletion Experiments**:
- Heavier heads (1-layer vs 3-layer)
- More bins (4 vs 8 vs 16)
- Richer geometry targets

**Success**: Light head + 4-8 bins reaches ≥95% of best performance.

---

## 4. Frontier Necessity

**Answer**: NO extra frontier primitive needed.

Keep base VLA fixed. Do NOT add:
- New LLM planner
- Separate VLM reasoner
- Diffusion policy
- 3D generative model

Contribution is a **post-training supervision recipe**, not a new model family.

---

## 5. Experiment Table (Compressed for 900-1400 GPUh)

### Core Experiments (Must Run)

| ID | Name | Purpose | Setup | Metrics | Success Criteria | Compute |
|:---:|------|---------|-------|---------|-----------------|:-------:|
| E0 | Pilot sanity | Validate implementation | 1-2 tasks, 5-10% data, 4 bins | Loss convergence, affordance rendering | Teacher head learns, KL stable | ~60h |
| E1 | **Anchor 4-arm** | Core thesis; Claims 1,2,4 | Arms A,B,C,D on libero_spatial + libero_goal (suite-level eval, G1-G6 + E1-E2 aggregation) | Success overall/by bucket; recovered gap; stress robustness; latency | B beats A on geometry-critical, beats C, recovers ≥50% gap | ~400h screen, ~600h final (3 seeds) |
| E2 | **Dense vs Depth Aux** | Claim 3 | Arm B vs same-data depth-prediction auxiliary | Success by bucket; recovered gap | Dense 2.5D beats depth aux on geometry-critical | ~150h screen, ~200h final |
| E3 | **2D vs 2.5D** | Mechanism necessity | 4 bins vs 2D heatmap (no depth) | Success on depth-ambiguous tasks; stress robustness | 2.5D beats 2D on depth-ambiguous | ~100h screen, ~150h final |
| E4 | LIBERO bucket + stress | Claim 2 | Evaluate E1 models on 6 geometry-critical + 2 easy + stress splits | Bucketed success; method × bucket interaction; robustness drop | Gains concentrate in geometry-critical; positive interaction | ~30h eval |
| E5 | Qualitative visualization | Mechanism validation | Visualize affordance maps on LIBERO successes/failures | Qualitative alignment; top-k overlap | Student maps align with teacher on geometry-critical regions | ~20h |
| E6 | **Real robot validation** | Real-world sanity | Primary: occluded narrow-surface placement; Control: visible open-zone placement | Success rate vs sim transfer; qualitative behavior match | Non-zero transfer on primary; qualitative behavior match | ~100h (incl. setup) |

### Secondary / Spotlight Push (If Core Passes Gates)

| ID | Name | Purpose | Priority | Compute |
|:---:|------|---------|:--------:|:-------:|
| E7 | Dense vs Logit KD | Is dense supervision necessary? | Secondary | ~150h |
| E8 | 4 vs 8 bins | Minimality sweep | Only if pilot suggests | ~100h |
| E9 | Dual-RGB teacher | Robustness to sensor modality | Appendix/Optional | ~200h |

### Total Core Budget (Locked)
- **Pilot (E0)**: ~60 GPUh
- **A/D baselines (Gate 1)**: ~150 GPUh
- **B/C distillation (Gate 2, 1-seed screen)**: ~400 GPUh
- **Anchor + mechanism (1-seed screen)**: ~380 GPUh
- **Real robot**: ~100 GPUh
- **Total**: **~1,090 GPUh** (fits 2-3x 4090, ~20-25 days)

---

## 6. Run Order & Decision Gates (Locked)

### Phase 0: Task Freeze + Engineering ✅ COMPLETED
**✅ COMPLETED**: Phase 0 is complete. Proceeding to Phase 1.
- [x] [T0] Freeze 6 geometry-critical task IDs: G1-G6 (official task names)
- [x] [T1] Freeze 2 easy control task IDs: E1-E2 (official task names)
- [x] [T2] Document stress split protocol: partial occluder, camera jitter, extra distractor
- [x] [T3] **ENGINEERING SOLUTION**: Implement Multi-view teacher path
  - Use `wrist_image` from OXE_DATASET_CONFIGS as privileged view
  - Enable multi-view input in OpenVLA-OFT forward pass (Arm D)
  - **Decision**: Use multi-view RGB instead of RGB-D (depth data unavailable)
- [x] [T4] Verified LIBERO dataset structure and wrist camera availability

### Phase 1: Pilot (~60 GPUh)
1. **E0**: Tiny sanity run (1-2 tasks, 5-10% data, 4 bins)
2. Validate: loss convergence, affordance head learns, KL stable
3. Lock λ and bin count (default: 4 bins)

### Phase 2: A/D Baselines (Gate 1) (~150 GPUh)
4. **Arm A**: Train RGB-only baseline (OpenVLA-OFT)
5. **Arm D**: Train RGB-D privileged teacher (OpenVLA-OFT + depth)
6. Evaluate both on libero_spatial (full 10) + libero_goal (full 10), then aggregate G1-G6 + E1-E2

**🔴 Gate 1**: RGB-D teacher clearly better than RGB baseline on geometry-critical subset?
- **YES** (RGB-D > RGB by ≥5 points) → Proceed to Phase 3
- **NO** → **STOP**. Do not proceed. Action: Re-select LIBERO tasks or strengthen geometry-critical criteria.

### Phase 3: B/C Distillation (Gate 2) (~400 GPUh)
7. **Arm B**: RGB + 2.5D distillation from RGB-D teacher
8. **Arm C**: RGB + same target from RGB-only teacher
9. 1-seed screening on G1-G6 + E1-E2 (aggregated from libero_spatial + libero_goal suite runs)

**🔴 Gate 2**: Arm B recovers ≥30% of (D - A) gap?
- **YES** (≥30%) → Proceed to Phase 4
- **NO** (<30%) → **Limited sweep only**: λ in [0.1, 0.5, 1.0], 4→8 bins. **Do NOT open new branches.**

### Phase 4: Anchor Eval + Mechanism Control (~380 GPUh)
10. **E1**: Full 4-arm anchor evaluation (promote to 3 seeds if screening promising)
11. **E2**: Dense 2.5D vs depth auxiliary (priority mechanism control)
12. **E3**: 2D vs 2.5D bins
13. **E4**: LIBERO bucket + stress evaluation
14. **E5**: Qualitative visualization

### Phase 5: Real Robot Validation (~100 GPUh)
15. **E6**: Transfer frozen task pair to real robot: primary (occluded narrow-surface) + control (visible open-zone)
16. Qualitative behavior check; rollout-level failure taxonomy

### Phase 6: Confirmatory Reruns (if needed)
17. Rerun claim-critical conditions with 3 seeds
18. Generate paper figures

---

## 7. Budget Summary (Locked)

| Phase | GPU Hours | Wall Clock (2-3x 4090) |
|-------|:---------:|:----------------------:|
| Task freeze + setup | 0 | 1-2 days |
| Pilot | ~60 | 2-3 days |
| A/D baselines (Gate 1) | ~150 | 4-5 days |
| B/C distillation (Gate 2) | ~400 | 10-12 days |
| Anchor + mechanism | ~380 | 8-10 days |
| Real robot | ~100 | 3-5 days (incl. setup) |
| **Total** | **~1,090** | **~20-25 days** |

### Human Time
- Setup/instrumentation: 2-3 days
- Monitoring and triage: 4-5 days
- Analysis, figures, qualitative: 3-4 days
- **Total**: 9-12 working days

---

## Success Criteria (Tiered)

### Screening (1-seed, quick validation)
- B beats A on ≥4/6 geometry-critical tasks
- B beats C on ≥3/6 geometry-critical tasks
- Recovery ≥25% of (D-A) gap

### Gate 1 Pass (proceed to B/C)
- RGB-D teacher > RGB baseline by ≥5 points on geometry-critical

### Gate 2 Pass (proceed to full eval)
- Arm B recovers ≥30% of (D-A) gap

### Strong Accept Target (3-seed confirmatory)
- B beats A by ≥8 points on geometry-critical
- B beats C by ≥4 points
- B recovers ≥50% of (D-A) gap
- p < 0.05 on primary comparisons

### Spotlight Stretch
- Recovery ≥60% of (D-A) gap
- Cohen's d > 0.5 on geometry-critical
- 4 bins reach ≥95% of 8-bin performance

### Poster Floor
- Positive trend on geometry-critical tasks
- E2 (dense vs depth aux) shows non-negative difference
- Real-robot: non-zero transfer on ≥1 task

---

## 8. Risk Mitigation

| Risk | Severity | Mitigation |
|------|:--------:|------------|
| **RGB-D teacher path not implemented** (depth_obs_keys=None; load_depth=False) | 🔴 BLOCKER | Phase 0 T3 must complete before Phase 1 |
| Privileged teacher weak | 🟡 HIGH | Strengthen benchmark; re-select tasks if no advantage |
| Student cannot learn map | 🟡 HIGH | Normalized KL, top-k, λ warmup |
| Dense ≈ depth auxiliary | 🟡 MED | Reframe: privileged geometry supervision broadly |
| Compute tight | 🟢 LOW | Cut E7 first; keep E1-E6 (claim-bearing) |

### Real Robot Validation Task Pair (Frozen)

**Primary**: `occluded narrow-surface placement` — corresponds to G1/G3 (LIBERO: pick object from between/near occluders and place on narrow target surface). Tests whether privileged geometry repair transfers to real-world partial occlusion.

**Control**: `visible open-zone placement` — corresponds to E1/E2 (LIBERO: pick from open scene, place on visible surface). Tests baseline stability and confirms gains are not from generic improvement.

Both tasks use the same arm, camera, and gripper hardware. Depth camera feeds only the teacher during training.

## Cleanest Paper Story

**Core Figure**: E1 anchor (4-arm comparison)
**Mechanism Controls**: E2, E3, E4
**Task Analysis**: E5
**Minimality**: E7
**Support**: E6, E8

Everything else is supplementary.

---
*Experiment plan by GPT-5.5 (xhigh reasoning)*
