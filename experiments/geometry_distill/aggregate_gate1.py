"""aggregate_gate1.py

Parse logs produced by run_libero_eval.py (Phase 2 Gate 1 run) and aggregate
per-task success rates into G1-G6 + E1-E2 buckets for Arm A and Arm D.

The official evaluator writes a text log per task with lines like:
  Current task success rate: 0.85
Plus a header line identifying the task description via the LIBERO Task object.

We use the --task_ids order we passed into the evaluator, mapped via
FROZEN_TASK_ID_MAP, so we rely on line-order pairing rather than parsing the
task description string.

Usage:
    python experiments/geometry_distill/aggregate_gate1.py \
        --eval_dir ./eval_results/phase2_gate1
"""

import argparse
import json
import os
import re
from pathlib import Path


# task_id -> slot label (must match run_phase2_gate1_eval.sh)
FROZEN_TASK_ID_MAP = {
    "libero_spatial": {0: "G1", 2: "E2", 4: "G2", 7: "G3", 9: "E1"},
    "libero_goal":    {2: "G4", 3: "G5", 6: "G6"},
}

# Order used when we passed --task_ids (sorted numerically since tqdm iterates in given order).
TASK_ID_ORDER = {
    "libero_spatial": [0, 2, 4, 7, 9],
    "libero_goal":    [2, 3, 6],
}

TASK_RATE_RE = re.compile(r"Current task success rate:\s*([0-9.]+)")


def parse_log(log_path: Path, task_id_order):
    """Return list of (task_id, success_rate) pairs parsed from the log in order."""
    rates = []
    with open(log_path, "r") as f:
        for line in f:
            m = TASK_RATE_RE.search(line)
            if m:
                rates.append(float(m.group(1)))
    if len(rates) != len(task_id_order):
        raise ValueError(
            f"Log {log_path} has {len(rates)} task results but expected "
            f"{len(task_id_order)} (task_ids={task_id_order}). "
            "Check whether the run crashed mid-way."
        )
    return list(zip(task_id_order, rates))


def aggregate_arm(eval_dir: Path, arm: str):
    """Read both suites for one arm and return {slot: rate} + bucket means."""
    slot_rates = {}
    for suite in ("libero_spatial", "libero_goal"):
        log_path = eval_dir / f"{arm}_{suite}.log"
        if not log_path.exists():
            raise FileNotFoundError(f"Missing log: {log_path}")
        pairs = parse_log(log_path, TASK_ID_ORDER[suite])
        for task_id, rate in pairs:
            slot = FROZEN_TASK_ID_MAP[suite][task_id]
            slot_rates[slot] = rate

    g_vals = [slot_rates[k] for k in slot_rates if k.startswith("G")]
    e_vals = [slot_rates[k] for k in slot_rates if k.startswith("E")]
    return {
        "per_slot": slot_rates,
        "geometry_critical_mean": sum(g_vals) / len(g_vals) if g_vals else 0.0,
        "easy_control_mean": sum(e_vals) / len(e_vals) if e_vals else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dir", required=True, type=str)
    parser.add_argument("--out", default=None, type=str)
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    out = {
        "arm_a": aggregate_arm(eval_dir, "arm_a"),
        "arm_d": aggregate_arm(eval_dir, "arm_d"),
    }

    a_geo = out["arm_a"]["geometry_critical_mean"]
    d_geo = out["arm_d"]["geometry_critical_mean"]
    a_easy = out["arm_a"]["easy_control_mean"]
    d_easy = out["arm_d"]["easy_control_mean"]

    out["gate1"] = {
        "d_minus_a_geometry_points": (d_geo - a_geo) * 100.0,
        "d_minus_a_easy_points": (d_easy - a_easy) * 100.0,
        "threshold_points": 5.0,
        "pass": (d_geo - a_geo) * 100.0 >= 5.0,
    }

    out_path = args.out or str(eval_dir / "gate1_summary.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print("=" * 60)
    print("Phase 2 Gate 1 Summary")
    print("=" * 60)
    print(f"  Arm A per-slot:  {out['arm_a']['per_slot']}")
    print(f"  Arm D per-slot:  {out['arm_d']['per_slot']}")
    print(f"  Geometry mean:   A={a_geo*100:.1f}%   D={d_geo*100:.1f}%   gap={(d_geo-a_geo)*100:+.1f} pts")
    print(f"  Easy mean:       A={a_easy*100:.1f}%   D={d_easy*100:.1f}%   gap={(d_easy-a_easy)*100:+.1f} pts")
    print(f"  Gate 1 (>=5 pts on G): {'PASS' if out['gate1']['pass'] else 'FAIL'}")
    print(f"  Written to: {out_path}")


if __name__ == "__main__":
    main()
