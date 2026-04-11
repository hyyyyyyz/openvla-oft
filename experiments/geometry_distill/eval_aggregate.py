"""
eval_aggregate.py

Aggregate evaluation results across LIBERO suites for the 8-task frozen slice.
Computes success rates for G1-G6 (geometry-critical) and E1-E2 (easy control).
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import draccus
import numpy as np


# Frozen 8-task slice from FINAL_PROPOSAL.md
FROZEN_TASKS = {
    # Geometry-Critical Tasks (G1-G6)
    "G1": {
        "name": "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
        "suite": "libero_spatial",
        "type": "geometry_critical",
        "failure_type": "Support ambiguity + occluder proximity",
    },
    "G2": {
        "name": "pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate",
        "suite": "libero_spatial",
        "type": "geometry_critical",
        "failure_type": "Partial-occlusion region",
    },
    "G3": {
        "name": "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate",
        "suite": "libero_spatial",
        "type": "geometry_critical",
        "failure_type": "Narrow-surface placement",
    },
    "G4": {
        "name": "put_the_wine_bottle_on_top_of_the_cabinet",
        "suite": "libero_goal",
        "type": "geometry_critical",
        "failure_type": "Occluded target, tall placement",
    },
    "G5": {
        "name": "open_the_top_drawer_and_put_the_bowl_inside",
        "suite": "libero_goal",
        "type": "geometry_critical",
        "failure_type": "Partial-occlusion container",
    },
    "G6": {
        "name": "put_the_cream_cheese_in_the_bowl",
        "suite": "libero_goal",
        "type": "geometry_critical",
        "failure_type": "Depth-order-sensitive placement",
    },
    # Easy Control Tasks (E1-E2)
    "E1": {
        "name": "pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate",
        "suite": "libero_spatial",
        "type": "easy_control",
        "failure_type": "Fully visible target surface",
    },
    "E2": {
        "name": "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate",
        "suite": "libero_spatial",
        "type": "easy_control",
        "failure_type": "Object and target fully visible, open scene",
    },
}


@dataclass
class EvalConfig:
    """Evaluation aggregation configuration."""

    # fmt: off
    # Input files (from run_libero_eval.py outputs)
    libero_spatial_results: str = ""       # Path to libero_spatial eval results JSON
    libero_goal_results: str = ""          # Path to libero_goal eval results JSON
    libero_object_results: Optional[str] = None  # Optional: libero_object results
    libero_10_results: Optional[str] = None      # Optional: libero_10 results

    # Output
    output_dir: str = "./eval_results"     # Output directory for aggregated results
    output_name: str = "aggregate"         # Output file name

    # Analysis options
    compute_recovery: bool = True          # Compute recovery % (Arm B vs A/D)
    arm_a_results: Optional[str] = None    # Arm A results for recovery calc
    arm_d_results: Optional[str] = None    # Arm D results for recovery calc

    # fmt: on


def load_results(filepath: str) -> Dict:
    """Load evaluation results from JSON file."""
    if not filepath or not os.path.exists(filepath):
        return {}
    with open(filepath, 'r') as f:
        return json.load(f)


def aggregate_task_results(results: Dict, task_name: str) -> float:
    """Extract success rate for a specific task from results."""
    # Results format depends on run_libero_eval.py output
    # Expected: {"task_name": {"success_rate": 0.8, "num_episodes": 50, ...}}

    # Try exact match
    if task_name in results:
        task_data = results[task_name]
        if isinstance(task_data, dict):
            return task_data.get("success_rate", 0.0)
        return float(task_data)

    # Try with spaces replaced by underscores
    task_key = task_name.replace(" ", "_")
    if task_key in results:
        task_data = results[task_key]
        if isinstance(task_data, dict):
            return task_data.get("success_rate", 0.0)
        return float(task_data)

    # Try case-insensitive match
    for key, value in results.items():
        if key.lower().replace(" ", "_") == task_name.lower().replace(" ", "_"):
            if isinstance(value, dict):
                return value.get("success_rate", 0.0)
            return float(value)

    return 0.0


def compute_aggregate_metrics(task_results: Dict[str, float]) -> Dict:
    """Compute aggregate metrics for the 8-task slice."""
    # Separate geometry-critical and easy control tasks
    g_tasks = {k: v for k, v in task_results.items() if k.startswith("G")}
    e_tasks = {k: v for k, v in task_results.items() if k.startswith("E")}

    metrics = {
        "geometry_critical": {
            "tasks": g_tasks,
            "mean": np.mean(list(g_tasks.values())) if g_tasks else 0.0,
            "std": np.std(list(g_tasks.values())) if g_tasks else 0.0,
            "min": min(g_tasks.values()) if g_tasks else 0.0,
            "max": max(g_tasks.values()) if g_tasks else 0.0,
        },
        "easy_control": {
            "tasks": e_tasks,
            "mean": np.mean(list(e_tasks.values())) if e_tasks else 0.0,
            "std": np.std(list(e_tasks.values())) if e_tasks else 0.0,
            "min": min(e_tasks.values()) if e_tasks else 0.0,
            "max": max(e_tasks.values()) if e_tasks else 0.0,
        },
        "overall": {
            "mean": np.mean(list(task_results.values())) if task_results else 0.0,
            "std": np.std(list(task_results.values())) if task_results else 0.0,
        },
    }

    return metrics


def compute_recovery_rate(
    arm_b_results: Dict[str, float],
    arm_a_results: Dict[str, float],
    arm_d_results: Dict[str, float],
) -> Dict:
    """Compute recovery rate: how much of (D-A) gap does B recover?"""
    recovery_rates = {}

    for task_id in arm_b_results.keys():
        if task_id not in arm_a_results or task_id not in arm_d_results:
            continue

        a_score = arm_a_results[task_id]
        b_score = arm_b_results[task_id]
        d_score = arm_d_results[task_id]

        # Recovery formula: (B - A) / (D - A)
        gap = d_score - a_score
        recovered = b_score - a_score

        if gap > 0:
            recovery_rate = recovered / gap
        elif gap < 0:
            # D is worse than A (unexpected), report raw difference
            recovery_rate = 0.0
        else:
            # No gap to recover
            recovery_rate = 0.0

        recovery_rates[task_id] = {
            "recovery_rate": recovery_rate,
            "recovered": recovered,
            "gap": gap,
            "arm_a": a_score,
            "arm_b": b_score,
            "arm_d": d_score,
        }

    # Aggregate recovery rates
    g_recovery = [v["recovery_rate"] for k, v in recovery_rates.items() if k.startswith("G")]
    e_recovery = [v["recovery_rate"] for k, v in recovery_rates.items() if k.startswith("E")]

    return {
        "per_task": recovery_rates,
        "geometry_critical": {
            "mean_recovery": np.mean(g_recovery) if g_recovery else 0.0,
            "std_recovery": np.std(g_recovery) if g_recovery else 0.0,
        },
        "easy_control": {
            "mean_recovery": np.mean(e_recovery) if e_recovery else 0.0,
            "std_recovery": np.std(e_recovery) if e_recovery else 0.0,
        },
    }


def print_results_table(task_results: Dict[str, float], metrics: Dict):
    """Print results in a formatted table."""
    print("\n" + "=" * 100)
    print("LIBERO 8-Task Slice Evaluation Results")
    print("=" * 100)

    # Geometry-critical tasks
    print("\n📊 Geometry-Critical Tasks (G1-G6):")
    print("-" * 100)
    print(f"{'Task':<10} {'Name':<70} {'Success Rate':>15}")
    print("-" * 100)
    for task_id in ["G1", "G2", "G3", "G4", "G5", "G6"]:
        if task_id in task_results:
            task_info = FROZEN_TASKS[task_id]
            name = task_info["name"][:65] + "..." if len(task_info["name"]) > 65 else task_info["name"]
            print(f"{task_id:<10} {name:<70} {task_results[task_id]*100:>14.1f}%")
    print("-" * 100)
    print(f"{'Mean':<10} {'':<70} {metrics['geometry_critical']['mean']*100:>14.1f}%")
    print(f"{'Std':<10} {'':<70} {metrics['geometry_critical']['std']*100:>14.1f}%")

    # Easy control tasks
    print("\n📊 Easy Control Tasks (E1-E2):")
    print("-" * 100)
    print(f"{'Task':<10} {'Name':<70} {'Success Rate':>15}")
    print("-" * 100)
    for task_id in ["E1", "E2"]:
        if task_id in task_results:
            task_info = FROZEN_TASKS[task_id]
            name = task_info["name"][:65] + "..." if len(task_info["name"]) > 65 else task_info["name"]
            print(f"{task_id:<10} {name:<70} {task_results[task_id]*100:>14.1f}%")
    print("-" * 100)
    print(f"{'Mean':<10} {'':<70} {metrics['easy_control']['mean']*100:>14.1f}%")

    # Overall
    print("\n📊 Overall:")
    print("-" * 100)
    print(f"8-Task Mean: {metrics['overall']['mean']*100:.1f}%")
    print(f"8-Task Std:  {metrics['overall']['std']*100:.1f}%")
    print("=" * 100)


def print_recovery_table(recovery: Dict):
    """Print recovery rate table."""
    print("\n" + "=" * 100)
    print("Recovery Analysis (Arm B recovery of privileged RGB-D advantage)")
    print("=" * 100)

    per_task = recovery["per_task"]

    print("\n📊 Per-Task Recovery Rates:")
    print("-" * 100)
    print(f"{'Task':<10} {'Arm A':>12} {'Arm B':>12} {'Arm D':>12} {'Gap':>12} {'Recovered':>12} {'Recovery %':>12}")
    print("-" * 100)

    for task_id in ["G1", "G2", "G3", "G4", "G5", "G6", "E1", "E2"]:
        if task_id in per_task:
            r = per_task[task_id]
            recovery_pct = r["recovery_rate"] * 100
            print(f"{task_id:<10} {r['arm_a']*100:>11.1f}% {r['arm_b']*100:>11.1f}% {r['arm_d']*100:>11.1f}% "
                  f"{r['gap']*100:>11.1f}% {r['recovered']*100:>11.1f}% {recovery_pct:>11.1f}%")

    print("-" * 100)
    print(f"\n📈 Geometry-Critical Mean Recovery: {recovery['geometry_critical']['mean_recovery']*100:.1f}%")
    print(f"📈 Easy Control Mean Recovery: {recovery['easy_control']['mean_recovery']*100:.1f}%")
    print("=" * 100)

    # Gate check
    gc_recovery = recovery['geometry_critical']['mean_recovery']
    if gc_recovery >= 0.50:
        status = "✅ STRONG ACCEPT threshold (≥50%)"
    elif gc_recovery >= 0.30:
        status = "✅ GATE 2 PASS threshold (≥30%)"
    elif gc_recovery >= 0.25:
        status = "⚠️ SCREENING threshold (≥25%)"
    else:
        status = "❌ Below screening threshold (<25%)"
    print(f"\n{status}")


def aggregate_evaluation(cfg: EvalConfig):
    """Main aggregation function."""
    # Load results from all suites
    spatial_results = load_results(cfg.libero_spatial_results)
    goal_results = load_results(cfg.libero_goal_results)
    object_results = load_results(cfg.libero_object_results) if cfg.libero_object_results else {}
    libero_10_results = load_results(cfg.libero_10_results) if cfg.libero_10_results else {}

    # Combine all results
    all_results = {}
    all_results.update(spatial_results)
    all_results.update(goal_results)
    all_results.update(object_results)
    all_results.update(libero_10_results)

    # Extract 8-task slice results
    task_results = {}
    for task_id, task_info in FROZEN_TASKS.items():
        task_name = task_info["name"]
        success_rate = aggregate_task_results(all_results, task_name)
        task_results[task_id] = success_rate

    # Compute aggregate metrics
    metrics = compute_aggregate_metrics(task_results)

    # Print results
    print_results_table(task_results, metrics)

    # Compute recovery if requested
    recovery = None
    if cfg.compute_recovery and cfg.arm_a_results and cfg.arm_d_results:
        arm_a_data = load_results(cfg.arm_a_results)
        arm_d_data = load_results(cfg.arm_d_results)

        # Extract 8-task for A and D
        arm_a_tasks = {}
        arm_d_tasks = {}
        for task_id, task_info in FROZEN_TASKS.items():
            arm_a_tasks[task_id] = aggregate_task_results(arm_a_data, task_info["name"])
            arm_d_tasks[task_id] = aggregate_task_results(arm_d_data, task_info["name"])

        recovery = compute_recovery_rate(task_results, arm_a_tasks, arm_d_tasks)
        print_recovery_table(recovery)

    # Save aggregated results
    os.makedirs(cfg.output_dir, exist_ok=True)
    output_path = os.path.join(cfg.output_dir, f"{cfg.output_name}.json")

    output_data = {
        "task_results": task_results,
        "metrics": metrics,
    }
    if recovery:
        output_data["recovery"] = recovery

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n💾 Results saved to: {output_path}")

    return output_data


@draccus.wrap()
def main(cfg: EvalConfig):
    """Main entry point."""
    aggregate_evaluation(cfg)


if __name__ == "__main__":
    main()
