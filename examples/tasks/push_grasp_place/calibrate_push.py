"""Isolate and validate the push primitive."""

import argparse

import torch

from examples.physics.pick_backend import PickBackend
from examples.tasks.push_grasp_place.executor import PrimitiveExecutor
from examples.tasks.push_grasp_place.primitives import PrimitiveAction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=70)
    parser.add_argument("--kp-pos", type=float, default=0.35)
    parser.add_argument("--kp-ori", type=float, default=0.5)
    parser.add_argument("--distance", type=float, default=0.14)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    backend = PickBackend()
    env = backend.create_environment(
        {
            "layout": "rotated_object",
            "num_envs": 1,
            "obs_mode": "state_dict",
            "render_mode": "human" if args.render else None,
            "sim_backend": "gpu",
            "control_mode": "pd_joint_delta_pos",
            "shader": "default",
        }
    )
    target = env.unwrapped.named_objects["target_object_0"]
    initial = target.pose.p.detach().clone()
    executor = PrimitiveExecutor(backend, env, kp_pos=args.kp_pos, kp_ori=args.kp_ori)
    result = executor.execute(
        PrimitiveAction(
            "push",
            {
                "direction": [0.0, 1.0],
                "distance": args.distance,
            },
            duration_steps=args.duration,
        )
    )
    final = target.pose.p.detach().clone()
    displacement = torch.linalg.norm(final[0, :2] - initial[0, :2]).item()

    print("initial=", [round(v, 4) for v in initial[0, :3].tolist()])
    print("final=", [round(v, 4) for v in final[0, :3].tolist()])
    print("displacement=", round(displacement, 4))
    print("success=", result.success)
    env.close()


if __name__ == "__main__":
    main()
