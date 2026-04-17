"""
run_libero_eval_fast.py

Fast evaluation for the 8-task frozen slice.
Only evaluates the specific tasks needed for Phase 2 gate.

Usage:
    python run_libero_eval_fast.py --model_path ./checkpoints/arm_d_official/final/ --task_suite libero_spatial --model_name arm_d
    python run_libero_eval_fast.py --model_path ./checkpoints/arm_a_rgb/final/ --task_suite libero_goal --model_name arm_a
"""

import json
import os
import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import draccus
import numpy as np
import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from libero.libero import benchmark
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.robot_utils import get_model
from experiments.robot.openvla_utils import get_processor
from prismatic.vla.constants import NUM_ACTIONS_CHUNK


# 8-task frozen slice mapping
FROZEN_TASKS = {
    "libero_spatial": {
        0: "G1",   # pick_up_the_black_bowl_between_the_plate_and_the_ramekin...
        2: "E2",   # pick_up_the_black_bowl_from_table_center...
        4: "G2",   # pick_up_the_black_bowl_in_the_top_drawer...
        7: "G3",   # pick_up_the_black_bowl_on_the_stove...
        9: "E1",   # pick_up_the_black_bowl_on_the_wooden_cabinet...
    },
    "libero_goal": {
        2: "G4",   # put_the_wine_bottle_on_top_of_the_cabinet
        3: "G5",   # open_the_top_drawer_and_put_the_bowl_inside
        6: "G6",   # put_the_cream_cheese_in_the_bowl
    },
}

TASK_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_goal": 300,
}


@dataclass
class FastEvalConfig:
    model_path: str = ""                    # Path to model checkpoint
    task_suite: str = "libero_spatial"     # "libero_spatial" or "libero_goal"
    model_name: str = "model"              # Name for output file
    num_images_in_input: int = 1          # 1=single-view, 2=multi-view
    lora_rank: int = 32                   # LoRA rank
    num_trials: int = 10                   # Episodes per task (10 for fast screening)
    seed: int = 7                         # Random seed
    output_dir: str = "./eval_results"    # Output directory


def set_seed(seed: int):
    """Set random seed."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def prepare_observation(obs, resize_size):
    """Prepare observation for policy input."""
    img = get_libero_image(obs)
    wrist_img = get_libero_wrist_image(obs)

    # Resize
    from PIL import Image
    img_resized = img.resize((resize_size, resize_size), Image.BILINEAR)
    wrist_resized = wrist_img.resize((resize_size, resize_size), Image.BILINEAR)

    observation = {
        "full_image": np.array(img_resized),
        "wrist_image": np.array(wrist_resized),
        "state": np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        ),
    }
    return observation, np.array(img)


def resize_image(img: np.ndarray, size: int) -> np.ndarray:
    """Resize image."""
    from PIL import Image
    pil_img = Image.fromarray(img)
    resized = pil_img.resize((size, size), Image.BILINEAR)
    return np.array(resized)


def normalize_gripper_action(action, binarize=True):
    """Normalize gripper action."""
    if binarize:
        threshold = 0.5
        action[-1] = 1.0 if action[-1] > threshold else -1.0
    return action


def invert_gripper_action(action):
    """Invert gripper action sign."""
    action[-1] = -action[-1]
    return action


def get_action_simple(vla, processor, obs, task_description, num_images, unnorm_key, device):
    """Get action from VLA model (simplified)."""
    from PIL import Image

    with torch.inference_mode():
        # Collect images
        all_images = [obs["full_image"]]
        if num_images > 1:
            all_images.append(obs["wrist_image"])

        # Resize all images
        all_images = [resize_image(img, 224) for img in all_images]

        # Build prompt
        prompt = f"In: What action should the robot take to {task_description.lower()}?\nOut:"

        # Process images
        if len(all_images) == 1:
            inputs = processor(prompt, all_images[0]).to(device, dtype=torch.bfloat16)
        else:
            inputs = processor(prompt, all_images[0]).to(device, dtype=torch.bfloat16)
            wrist_inputs = processor(prompt, all_images[1]).to(device, dtype=torch.bfloat16)
            primary_pv = inputs["pixel_values"]
            wrist_pv = wrist_inputs["pixel_values"]
            inputs["pixel_values"] = torch.cat([primary_pv, wrist_pv], dim=1)

        # Get action
        action, _ = vla.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)

    return action


def run_episode(vla, processor, env, task_description, num_images, unnorm_key, device, initial_state, max_steps=300):
    """Run a single episode."""
    env.reset()
    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        # reset() returns the observation in newer robosuite versions
        obs = env.reset()

    action_queue = deque(maxlen=8)
    t = 0
    replay_images = []
    num_steps_wait = 10

    success = False
    try:
        while t < max_steps + num_steps_wait:
            if t < num_steps_wait:
                obs, _, _, _ = env.step(get_libero_dummy_action("openvla"))
                t += 1
                continue

            observation, img = prepare_observation(obs, 224)
            replay_images.append(img)

            if len(action_queue) == 0:
                actions = get_action_simple(vla, processor, observation, task_description, num_images, unnorm_key, device)
                action_queue.extend(actions)

            action = action_queue.popleft()
            action = normalize_gripper_action(action, binarize=True)
            action = invert_gripper_action(action)

            obs, reward, done, info = env.step(action.tolist())
            if done:
                success = True
                break
            t += 1

    except Exception as e:
        print(f"Episode error: {e}")

    return success, replay_images


@draccus.wrap()
def main(cfg: FastEvalConfig):
    """Main evaluation function."""
    set_seed(cfg.seed)

    # Determine device
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Determine task IDs to evaluate
    task_ids = FROZEN_TASKS.get(cfg.task_suite, {}).keys()
    if not task_ids:
        print(f"No frozen tasks for suite: {cfg.task_suite}")
        return

    # Setup config for model loading
    class ModelCfg:
        model_family = "openvla"
        pretrained_checkpoint = cfg.model_path
        num_images_in_input = cfg.num_images_in_input
        lora_rank = cfg.lora_rank
        load_in_8bit = False
        load_in_4bit = False
        use_film = False
        use_l1_regression = False
        use_diffusion = False

    model_cfg = ModelCfg()

    # Load model
    print(f"Loading model from: {cfg.model_path}")
    vla = get_model(model_cfg)
    vla.eval()
    vla = vla.to(device, dtype=torch.bfloat16)

    # Set num_images_in_input
    vla.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)

    # Load processor
    processor = get_processor(model_cfg)

    # Load dataset stats
    stats_path = os.path.join(cfg.model_path, "dataset_statistics.json")
    if os.path.exists(stats_path):
        with open(stats_path, "r") as f:
            norm_stats = json.load(f)
        vla.norm_stats = norm_stats
        print(f"Loaded dataset statistics from: {stats_path}")

    # Determine unnorm_key
    unnorm_key = cfg.task_suite
    if f"{unnorm_key}_no_noops" in vla.norm_stats:
        unnorm_key = f"{unnorm_key}_no_noops"
    print(f"Using unnorm_key: {unnorm_key}")

    # Initialize task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite]()

    # Get max steps
    max_steps = TASK_MAX_STEPS.get(cfg.task_suite, 300)

    # Run evaluation
    results = {}
    total_episodes = 0
    total_successes = 0

    print(f"\nEvaluating {len(task_ids)} tasks ({cfg.num_trials} episodes each)...")

    for task_id in sorted(task_ids):
        task = task_suite.get_task(task_id)
        # Get task description - different LIBERO versions may use different attributes
        task_description = (getattr(task, 'language_instruction', None) or
                           getattr(task, 'task_name', None) or
                           getattr(task, 'goal', None) or
                           str(task))

        env, _ = get_libero_env(task, "openvla", resolution=256)
        initial_states = task_suite.get_task_init_states(task_id)

        task_successes = 0
        for episode_idx in range(cfg.num_trials):
            initial_state = initial_states[episode_idx]

            success, replay_images = run_episode(
                vla, processor, env, task_description,
                cfg.num_images_in_input, unnorm_key, device,
                initial_state, max_steps
            )

            if success:
                task_successes += 1
            total_successes += 1
            total_episodes += 1

            # Save replay video
            save_rollout_video(
                replay_images, total_episodes, success=success,
                task_description=task_description
            )

            print(f"  {FROZEN_TASKS[cfg.task_suite][task_id]} (task {task_id}) ep {episode_idx+1}/{cfg.num_trials}: "
                  f"success={success}, total={total_successes}/{total_episodes} ({total_successes/total_episodes*100:.0f}%)")

        task_success_rate = task_successes / cfg.num_trials
        results[FROZEN_TASKS[cfg.task_suite][task_id]] = {
            "task_id": task_id,
            "task_name": task_description,
            "success_rate": task_success_rate,
            "num_trials": cfg.num_trials,
            "num_successes": task_successes,
        }

        # Save intermediate results
        output_file = os.path.join(cfg.output_dir, f"{cfg.model_name}_{cfg.task_suite}_results.json")
        os.makedirs(cfg.output_dir, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Saved to: {output_file}")

    # Print summary
    print(f"\n=== Summary for {cfg.model_name} on {cfg.task_suite} ===")
    for task_id, info in results.items():
        print(f"  {task_id}: {info['success_rate']*100:.0f}% ({info['num_successes']}/{info['num_trials']})")

    gc_tasks = [v for k, v in results.items() if k.startswith("G")]
    ec_tasks = [v for k, v in results.items() if k.startswith("E")]
    if gc_tasks:
        gc_mean = np.mean([t["success_rate"] for t in gc_tasks])
        print(f"  Geometry-Critical Mean: {gc_mean*100:.1f}%")
    if ec_tasks:
        ec_mean = np.mean([t["success_rate"] for t in ec_tasks])
        print(f"  Easy Control Mean: {ec_mean*100:.1f}%")

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
