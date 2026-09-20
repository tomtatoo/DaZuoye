"""Run a scripted primitive sequence for one or all task layouts."""

import argparse
import csv
import json
import time
from datetime import datetime
from math import pi
from pathlib import Path

import torch

from examples.physics.pick_backend import PickBackend
from examples.tasks.push_grasp_place.executor import PrimitiveExecutor
from examples.tasks.push_grasp_place.layouts import LAYOUTS, LAYOUTS_BY_NAME, get_layout
from examples.tasks.push_grasp_place.primitives import PrimitiveAction, PrimitiveSequence, validate_sequence


def _action(name, params=None, duration=None):
    return PrimitiveAction(name, params or {}, duration_steps=duration)


def build_sequence(layout):
    actions = []
    if layout.name == "occluded_block":
        actions.append(_action("clear", {"obstacle_id": "front_block", "direction": [1, 0], "distance": 0.14}, 35))
        actions.append(_action("push", {"direction": [0, 1], "distance": 0.10}, 35))
    elif layout.name == "against_wall":
        actions.append(_action("push", {"direction": [-1, 0], "distance": 0.18}, 70))
    elif layout.name == "narrow_gap":
        actions.append(_action("push", {"direction": [0, 1], "distance": 0.14}, 70))
    elif layout.name == "rotated_object":
        actions.append(_action("rotate", {"yaw_delta": -0.5 * pi}, 120))
    elif layout.name == "multi_obstacle":
        actions.append(_action("push", {"direction": [0, -1], "distance": 0.14}, 70))
    else:
        raise KeyError(f"No scripted sequence for layout {layout.name!r}")

    actions.extend(
        [
            _action("grasp", {"target_id": "target_object_0", "grasp_height": 0.01}, 35),
            _action("lift", {"height": 0.08}, 20),
            _action("place", {"position": list(layout.place_position), "height": 0.04}, 40),
            _action("release", {}, 6),
            _action("retreat", {"direction": [0, -1], "distance": 0.08}, 12),
            _action("wait", {}, 5),
        ]
    )
    sequence = PrimitiveSequence(tuple(actions))
    validate_sequence(sequence)
    return sequence


def run_layout(layout_name, render=False, shader="default"):
    layout = get_layout(layout_name)
    backend = PickBackend()
    env = backend.create_environment(
        {
            "layout": layout.name,
            "num_envs": 1,
            "obs_mode": "state_dict",
            "render_mode": "human" if render else None,
            "sim_backend": "gpu",
            "control_mode": "pd_joint_delta_pos",
            "shader": shader,
        }
    )
    executor = PrimitiveExecutor(backend, env, kp_pos=0.7, kp_ori=0.6)
    initial_target = env.unwrapped.named_objects["target_object_0"].pose.p.detach().clone()
    sequence = build_sequence(layout)
    results = []
    primitive_logs = []
    layout_start = time.perf_counter()
    for action in sequence.actions:
        primitive_start = time.perf_counter()
        result = executor.execute(action)
        primitive_elapsed = time.perf_counter() - primitive_start
        results.append(result)
        target_state = env.unwrapped.named_objects["target_object_0"].pose.p.detach()
        tcp_state = env.unwrapped.agent.tcp.pose.p.detach()
        primitive_logs.append(
            {
                "action": action.name,
                "success": result.success,
                "target": [round(v, 4) for v in target_state[0, :3].tolist()],
                "tcp": [round(v, 4) for v in tcp_state[0, :3].tolist()],
                "elapsed_sec": round(primitive_elapsed, 6),
            }
        )
    layout_elapsed = time.perf_counter() - layout_start
    final_target = env.unwrapped.named_objects["target_object_0"].pose.p.detach().clone()
    evaluation = env.unwrapped.evaluate()
    success = bool(evaluation["success"][0].item())
    displacement = torch.linalg.norm(final_target[0, :2] - initial_target[0, :2]).item()
    summary = {
        "layout": layout.name,
        "success": success,
        "primitive_steps": sequence.total_duration_steps,
        "primitive_count": len(sequence.actions),
        "target_initial": [round(v, 4) for v in initial_target[0, :3].tolist()],
        "target_final": [round(v, 4) for v in final_target[0, :3].tolist()],
        "target_displacement": round(displacement, 4),
        "primitive_success": all(result.success for result in results),
        "primitive_logs": primitive_logs,
        "replan_count": 0,
        "elapsed_sec": round(layout_elapsed, 6),
    }
    env.close()
    return summary


def write_run_artifacts(summaries, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "baseline_table.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "layout",
                "success",
                "primitive_success",
                "primitive_count",
                "primitive_steps",
                "replan_count",
                "target_displacement",
                "elapsed_sec",
                "target_initial",
                "target_final",
            ],
        )
        writer.writeheader()
        for summary in summaries:
            writer.writerow(
                {
                    "layout": summary["layout"],
                    "success": summary["success"],
                    "primitive_success": summary["primitive_success"],
                    "primitive_count": summary["primitive_count"],
                    "primitive_steps": summary["primitive_steps"],
                    "replan_count": summary["replan_count"],
                    "target_displacement": summary["target_displacement"],
                    "elapsed_sec": summary["elapsed_sec"],
                    "target_initial": json.dumps(summary["target_initial"]),
                    "target_final": json.dumps(summary["target_final"]),
                }
            )

    with (output_dir / "layout_metrics.jsonl").open("w", encoding="utf-8") as handle:
        for summary in summaries:
            handle.write(json.dumps(summary, ensure_ascii=True) + "\n")

    with (output_dir / "primitive_log.jsonl").open("w", encoding="utf-8") as handle:
        for summary in summaries:
            for index, primitive in enumerate(summary["primitive_logs"]):
                record = {"layout": summary["layout"], "index": index}
                record.update(primitive)
                handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    return output_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout", choices=sorted(LAYOUTS_BY_NAME), default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--shader", default="default", choices=["default", "minimal", "rt", "rt-fast"])
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    if args.all:
        layouts = [layout.name for layout in LAYOUTS]
    elif args.layout:
        layouts = [args.layout]
    else:
        parser.error("Use --layout NAME or --all")

    summaries = []
    for layout_name in layouts:
        summary = run_layout(layout_name, render=args.render, shader=args.shader)
        summaries.append(summary)
        print(summary)

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("artifacts") / "push_grasp_place" / timestamp
    output_dir = write_run_artifacts(summaries, output_dir)

    print("total=", len(summaries), "successful=", sum(item["success"] for item in summaries))
    print("artifacts=", output_dir.resolve())


if __name__ == "__main__":
    main()
