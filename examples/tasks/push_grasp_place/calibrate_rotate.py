"""Isolate and validate the rotate primitive."""

import argparse
from math import atan2, pi

import torch

from examples.physics.pick_backend import PickBackend
from examples.tasks.push_grasp_place.executor import PrimitiveExecutor
from examples.tasks.push_grasp_place.primitives import PrimitiveAction


def _yaw(quaternion):
    w, x, y, z = [float(value) for value in quaternion.reshape(-1)[:4]]
    return atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=60)
    parser.add_argument("--yaw-delta", type=float, default=-0.5 * pi)
    parser.add_argument("--kp-pos", type=float, default=0.35)
    parser.add_argument("--kp-ori", type=float, default=0.5)
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
    initial_pos = target.pose.p.detach().clone()
    initial_yaw = _yaw(target.pose.q)
    executor = PrimitiveExecutor(backend, env, kp_pos=args.kp_pos, kp_ori=args.kp_ori)
    result = executor.execute(
        PrimitiveAction(
            "rotate",
            {"yaw_delta": args.yaw_delta},
            duration_steps=args.duration,
        )
    )
    final_pos = target.pose.p.detach().clone()
    final_tcp = env.unwrapped.agent.tcp.pose.p.detach().clone()
    final_yaw = _yaw(target.pose.q)
    yaw_change = atan2(
        torch.sin(torch.tensor(final_yaw - initial_yaw)).item(),
        torch.cos(torch.tensor(final_yaw - initial_yaw)).item(),
    )
    displacement = torch.linalg.norm(final_pos[0, :2] - initial_pos[0, :2]).item()
    rotation_success = abs(yaw_change) >= 0.2 and displacement <= 0.15

    print("initial_pos=", [round(v, 4) for v in initial_pos[0, :3].tolist()])
    print("final_pos=", [round(v, 4) for v in final_pos[0, :3].tolist()])
    print("final_tcp=", [round(v, 4) for v in final_tcp[0, :3].tolist()])
    print("tcp_xy_distance=", round(torch.linalg.norm(final_tcp[0, :2] - final_pos[0, :2]).item(), 4))
    print("initial_yaw=", round(initial_yaw, 4))
    print("final_yaw=", round(final_yaw, 4))
    print("yaw_change=", round(yaw_change, 4))
    print("displacement=", round(displacement, 4))
    print("success=", result.success)
    print("rotation_success=", rotation_success)
    env.close()


if __name__ == "__main__":
    main()
