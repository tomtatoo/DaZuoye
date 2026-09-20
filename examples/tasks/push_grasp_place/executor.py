"""Execution layer for high-level push-grasp-place primitives."""

from dataclasses import dataclass
from math import atan2, cos, sin, sqrt
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple

import torch

from .primitives import PrimitiveAction, PrimitiveSequence

if TYPE_CHECKING:
    from examples.physics.pick_backend import PickBackend


@dataclass
class PrimitiveResult:
    """Result of executing one primitive action."""

    action_name: str
    steps: int
    success: bool
    observation: Any
    info: Dict[str, Any]


class PrimitiveExecutor:
    """Translate high-level primitives into low-level end-effector commands."""

    def __init__(
        self,
        backend: "PickBackend",
        env: Any,
        open_gripper: float = 0.9,
        close_gripper: float = -0.9,
        kp_pos: Optional[float] = None,
        kp_ori: Optional[float] = None,
        kinematic_grasp: bool = True,
        grasp_tolerance: float = 0.12,
    ) -> None:
        self.backend = backend
        self.env = env
        self.base_env = env.unwrapped if hasattr(env, "unwrapped") else env
        self.open_gripper = float(open_gripper)
        self.close_gripper = float(close_gripper)
        self.kp_pos = backend.Kp_pos if kp_pos is None else float(kp_pos)
        self.kp_ori = backend.Kp_ori if kp_ori is None else float(kp_ori)
        self.kinematic_grasp = bool(kinematic_grasp)
        self.grasp_tolerance = float(grasp_tolerance)
        self.device = self.base_env.device
        self.pos_offset = torch.tensor([0.0, 0.0, -0.33], device=self.device)
        self.grasped = False
        self.attached_object_id: Optional[str] = None
        self.attachment_offset: Optional[torch.Tensor] = None

    def _robot_qpos(self) -> torch.Tensor:
        return self.base_env.agent.robot.get_qpos()

    def _desired_position(self) -> torch.Tensor:
        return self._robot_qpos()[:, :3] - self.pos_offset.unsqueeze(0)

    def _tcp_position(self) -> torch.Tensor:
        return self.base_env.agent.tcp.pose.p

    def _desired_for_tcp(self, tcp_position: torch.Tensor) -> torch.Tensor:
        offset = self._tcp_position() - self._desired_position()
        return tcp_position.reshape(1, 3) - offset

    def _current_orientation(self) -> torch.Tensor:
        return self._robot_qpos()[:, 3:6]

    def _actor_position(self, object_id: str) -> torch.Tensor:
        named = getattr(self.base_env, "named_objects", {})
        actor = named.get(object_id)
        if actor is None and not object_id.startswith("obstacle_"):
            actor = named.get(f"obstacle_{object_id}")
        if actor is None:
            raise KeyError(f"Unknown scene object {object_id!r}")
        return actor.pose.p.squeeze(0)

    def _step_pose(
        self,
        target_position: torch.Tensor,
        target_orientation: torch.Tensor,
        gripper_action: float,
        steps: int,
        update_attachment: bool = True,
    ) -> Tuple[Any, Dict[str, Any]]:
        obs = None
        info: Dict[str, Any] = {}
        target_position = target_position.reshape(1, 3)
        target_orientation = target_orientation.reshape(1, 3)
        for _ in range(steps):
            action = self.backend.control_ee_pose(
                self.base_env,
                target_position,
                target_orientation,
                Kp_pos=self.kp_pos,
                Kp_ori=self.kp_ori,
            )
            action[:, -1] = gripper_action
            obs, _, _, _, info = self.base_env.step(action)
            if update_attachment:
                self._sync_attached_object()
        return obs, info

    def _sync_attached_object(self) -> None:
        if not self.grasped or self.attached_object_id is None or self.attachment_offset is None:
            return
        actor = self.base_env.named_objects.get(self.attached_object_id)
        if actor is None:
            return
        target_position = self._tcp_position() + self.attachment_offset
        self._set_actor_position(actor, target_position)

    def _set_actor_position(self, actor, position: torch.Tensor) -> None:
        from mani_skill.utils.structs import Pose

        target_orientation = actor.pose.q
        actor.set_pose(Pose.create_from_pq(p=position, q=target_orientation))
        scene = getattr(self.base_env, "scene", None)
        if scene is not None and hasattr(scene, "_gpu_apply_all"):
            scene._gpu_apply_all()
            if hasattr(scene, "px") and hasattr(scene.px, "gpu_update_articulation_kinematics"):
                scene.px.gpu_update_articulation_kinematics()
            if hasattr(scene, "_gpu_fetch_all"):
                scene._gpu_fetch_all()

    def _hold_current_pose(self, gripper_action: float, steps: int):
        return self._step_pose(
            self._desired_position(),
            self._current_orientation(),
            gripper_action,
            steps,
        )

    def _move_delta(self, direction: Iterable[float], distance: float, gripper_action: float, steps: int):
        direction_tensor = torch.tensor(tuple(direction), device=self.device, dtype=torch.float32)
        norm = torch.linalg.norm(direction_tensor)
        if norm.item() <= 1e-8:
            raise ValueError("Primitive direction must be non-zero")
        delta = direction_tensor / norm * float(distance)
        target = self._desired_position().clone()
        target[0, :2] += delta
        return self._step_pose(target, self._current_orientation(), gripper_action, steps)

    def _execute_push(self, action: PrimitiveAction):
        target = self._actor_position("target_object_0")
        return self._push_object(target, action)

    def _execute_clear(self, action: PrimitiveAction):
        target = self._actor_position(action.params["obstacle_id"])
        return self._push_object(target, action)

    def _push_object(self, object_position: torch.Tensor, action: PrimitiveAction):
        direction = torch.tensor(tuple(action.params["direction"]), device=self.device, dtype=torch.float32)
        norm = torch.linalg.norm(direction)
        if norm.item() <= 1e-8:
            raise ValueError("Primitive direction must be non-zero")
        direction = direction / norm
        distance = float(action.params["distance"])
        contact_z = object_position[2] + 0.005

        approach = object_position.clone()
        approach[:2] -= direction * 0.08
        approach[2] = contact_z

        final = object_position.clone()
        final[:2] += direction * distance
        final[2] = contact_z

        approach_steps = max(1, action.duration_steps // 2)
        push_steps = action.duration_steps - approach_steps
        obs, info = self._step_pose(
            self._desired_for_tcp(approach),
            self._current_orientation(),
            self.open_gripper,
            approach_steps,
        )
        obs, info = self._step_pose(
            self._desired_for_tcp(final),
            self._current_orientation(),
            self.open_gripper,
            push_steps,
        )
        return obs, info

    def _execute_rotate(self, action: PrimitiveAction):
        center = self._actor_position("target_object_0")
        current = self._tcp_position()[0]
        radial = current[:2] - center[:2]
        current_angle = atan2(float(radial[1]), float(radial[0]))
        contact_radius = 0.045
        contact_z = center[2] + 0.010

        approach = center.clone()
        approach[0] = center[0] + contact_radius * cos(current_angle)
        approach[1] = center[1] + contact_radius * sin(current_angle)
        approach[2] = contact_z
        approach_steps = max(1, action.duration_steps // 4)
        obs, info = self._step_pose(
            self._desired_for_tcp(approach),
            self._current_orientation(),
            self.open_gripper,
            approach_steps,
        )

        remaining_steps = action.duration_steps - approach_steps
        arc_segments = 8
        segment_steps = max(1, remaining_steps // arc_segments)
        yaw_delta = float(action.params["yaw_delta"])
        for segment in range(1, arc_segments + 1):
            angle = current_angle + yaw_delta * (segment / arc_segments)
            target = center.clone()
            target[0] = center[0] + contact_radius * cos(angle)
            target[1] = center[1] + contact_radius * sin(angle)
            target[2] = contact_z
            obs, info = self._step_pose(
                self._desired_for_tcp(target),
                self._current_orientation(),
                self.close_gripper,
                segment_steps,
            )
        return obs, info

    def _execute_grasp(self, action: PrimitiveAction):
        grasp_height = float(action.params.get("grasp_height", 0.05))
        approach_phases = 3
        approach_steps = max(1, action.duration_steps // (approach_phases + 1))
        obs = None
        info: Dict[str, Any] = {}
        for _ in range(approach_phases):
            target = self._actor_position(action.params["target_id"])
            target_position = target.clone()
            target_position[2] += grasp_height
            desired_position = self._desired_for_tcp(target_position)
            obs, info = self._step_pose(
                desired_position,
                self._current_orientation(),
                self.open_gripper,
                approach_steps,
            )
        target = self._actor_position(action.params["target_id"])
        target_position = target.clone()
        target_position[2] += grasp_height
        desired_position = self._desired_for_tcp(target_position)
        close_steps = max(1, action.duration_steps - approach_steps * approach_phases)
        obs, info = self._step_pose(
            desired_position,
            self._current_orientation(),
            self.close_gripper,
            close_steps,
        )
        target_distance = torch.linalg.norm(self._tcp_position()[0] - target).item()
        grasp_success = target_distance <= self.grasp_tolerance
        if self.kinematic_grasp and grasp_success:
            self.grasped = True
            self.attached_object_id = action.params["target_id"]
            self.attachment_offset = target - self._tcp_position()[0]
        info = dict(info)
        info["grasp_success"] = grasp_success
        info["grasp_distance"] = target_distance
        return obs, info

    def _execute_lift(self, action: PrimitiveAction):
        target_tcp = self._tcp_position().clone()
        target_tcp[0, 2] += float(action.params["height"])
        target = self._desired_for_tcp(target_tcp)
        return self._step_pose(target, self._current_orientation(), self.close_gripper, action.duration_steps)

    def _execute_move_to(self, action: PrimitiveAction):
        target_tcp = torch.tensor(action.params["position"], device=self.device, dtype=torch.float32)
        target = self._desired_for_tcp(target_tcp)
        orientation = self._current_orientation()
        if "orientation" in action.params:
            orientation = torch.tensor(action.params["orientation"], device=self.device, dtype=torch.float32).unsqueeze(0)
        return self._step_pose(target, orientation, self.close_gripper if self.grasped else self.open_gripper, action.duration_steps)

    def _execute_place(self, action: PrimitiveAction):
        object_target = torch.tensor(action.params["position"], device=self.device, dtype=torch.float32).clone()
        object_target[2] += float(action.params.get("height", 0.04))
        orientation = self._current_orientation().clone()
        if "yaw" in action.params:
            orientation[0, 2] = float(action.params["yaw"])

        obs = None
        info: Dict[str, Any] = {}
        phase_steps = max(1, action.duration_steps // 3)
        for _ in range(3):
            if self.grasped and self.attached_object_id is not None:
                actual = self._actor_position(self.attached_object_id)
                error = object_target - actual
                target_tcp = self._tcp_position()[0] + error
            else:
                target_tcp = object_target
            target = self._desired_for_tcp(target_tcp)
            obs, info = self._step_pose(target, orientation, self.close_gripper, phase_steps)
        if self.grasped and self.attached_object_id is not None:
            actor = self.base_env.named_objects[self.attached_object_id]
            self.attachment_offset = object_target - self._tcp_position()[0]
            self._set_actor_position(actor, object_target)
        return obs, info

    def _execute_release(self, action: PrimitiveAction):
        obs, info = self._step_pose(
            self._desired_position(),
            self._current_orientation(),
            self.open_gripper,
            action.duration_steps,
            update_attachment=True,
        )
        self.grasped = False
        self.attached_object_id = None
        self.attachment_offset = None
        return obs, info

    def _execute_retreat(self, action: PrimitiveAction):
        return self._move_delta(
            action.params["direction"],
            action.params["distance"],
            self.open_gripper,
            action.duration_steps,
        )

    def _execute_wait(self, action: PrimitiveAction):
        duration = int(action.params.get("duration_steps", action.duration_steps))
        gripper = self.close_gripper if self.grasped else self.open_gripper
        return self._hold_current_pose(gripper, duration)

    def execute(self, action: PrimitiveAction) -> PrimitiveResult:
        """Execute one primitive and return the final observation."""

        handlers = {
            "push": self._execute_push,
            "clear": self._execute_clear,
            "rotate": self._execute_rotate,
            "grasp": self._execute_grasp,
            "lift": self._execute_lift,
            "move_to": self._execute_move_to,
            "place": self._execute_place,
            "release": self._execute_release,
            "retreat": self._execute_retreat,
            "wait": self._execute_wait,
        }
        if action.name not in handlers:
            raise KeyError(f"Unsupported primitive {action.name!r}")
        original_gains = (self.kp_pos, self.kp_ori)
        if action.name == "rotate":
            self.kp_pos, self.kp_ori = 0.8, 0.5
        else:
            self.kp_pos, self.kp_ori = 0.35, 0.5
        try:
            obs, info = handlers[action.name](action)
        finally:
            self.kp_pos, self.kp_ori = original_gains
        success = bool(info.get("grasp_success", True))
        return PrimitiveResult(
            action_name=action.name,
            steps=action.duration_steps,
            success=success,
            observation=obs,
            info=info,
        )

    def execute_sequence(self, sequence: PrimitiveSequence) -> List[PrimitiveResult]:
        """Execute all primitives in order."""

        results: List[PrimitiveResult] = []
        for action in sequence.actions:
            result = self.execute(action)
            results.append(result)
        return results
