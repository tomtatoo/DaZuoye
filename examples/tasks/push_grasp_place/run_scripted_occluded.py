"""Run and inspect a scripted push-grasp-place sequence on occluded_block."""

import argparse

from examples.physics.pick_backend import PickBackend
from examples.tasks.push_grasp_place.executor import PrimitiveExecutor
from examples.tasks.push_grasp_place.layouts import get_layout
from examples.tasks.push_grasp_place.primitives import PrimitiveAction


def _xyz(tensor):
    return [round(value, 4) for value in tensor.detach().cpu().reshape(-1)[:3].tolist()]


def _scene_state(env):
    base = env.unwrapped if hasattr(env, "unwrapped") else env
    target = base.named_objects["target_object_0"].pose.p
    obstacle = base.named_objects["obstacle_front_block"].pose.p
    tcp = base.agent.tcp.pose.p
    return {
        "target": _xyz(target),
        "obstacle": _xyz(obstacle),
        "tcp": _xyz(tcp),
    }


def build_sequence(layout):
    return [
        PrimitiveAction(
            "clear",
            {"obstacle_id": "front_block", "direction": [1.0, 0.0], "distance": 0.12},
            duration_steps=35,
        ),
        PrimitiveAction(
            "push",
            {"direction": [0.0, 1.0], "distance": 0.10},
            duration_steps=35,
        ),
        PrimitiveAction("grasp", {"target_id": "target_object_0", "grasp_height": 0.010}, duration_steps=35),
        PrimitiveAction("lift", {"height": 0.08}, duration_steps=20),
        PrimitiveAction(
            "place",
            {"position": list(layout.place_position), "height": 0.04},
            duration_steps=40,
        ),
        PrimitiveAction("release", {}, duration_steps=6),
        PrimitiveAction("retreat", {"direction": [0.0, -1.0], "distance": 0.08}, duration_steps=12),
        PrimitiveAction("wait", {}, duration_steps=5),
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--shader", default="default", choices=["default", "minimal", "rt", "rt-fast"])
    args = parser.parse_args()

    layout = get_layout("occluded_block")
    backend = PickBackend()
    env = backend.create_environment(
        {
            "layout": layout.name,
            "num_envs": 1,
            "obs_mode": "state_dict",
            "render_mode": "human" if args.render else None,
            "sim_backend": "gpu",
            "control_mode": "pd_joint_delta_pos",
            "shader": args.shader,
        }
    )
    executor = PrimitiveExecutor(backend, env, kp_pos=0.35, kp_ori=0.5)

    print("initial", _scene_state(env))
    for action in build_sequence(layout):
        result = executor.execute(action)
        print(action.name, "steps=", result.steps, _scene_state(env))

    evaluation = env.unwrapped.evaluate()
    print("evaluation", evaluation)
    print("success", bool(evaluation["success"][0].item()))
    env.close()


if __name__ == "__main__":
    main()
