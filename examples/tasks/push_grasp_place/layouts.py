"""Initial layout definitions for the push-grasp-place task.

Coordinates use the ManiSkill tabletop world frame:
    x: left/right
    y: depth across the table
    z: height above the table

The object and place poses are specifications for the task environment. They
do not create actors by themselves; the environment integration should consume
these values during reset.
"""

from dataclasses import dataclass
from math import pi
from typing import Dict, Tuple


Vec3 = Tuple[float, float, float]


@dataclass(frozen=True)
class ObstacleSpec:
    """A static or movable box obstacle in the task scene."""

    name: str
    position: Vec3
    half_size: Vec3
    yaw: float = 0.0
    movable: bool = True


@dataclass(frozen=True)
class LayoutSpec:
    """Complete initial state specification for one task layout."""

    name: str
    description: str
    direct_grasp_blocked_reason: str
    target_position: Vec3
    target_yaw: float
    obstacle_specs: Tuple[ObstacleSpec, ...]
    place_position: Vec3
    place_yaw: float
    place_radius: float
    initial_gripper_position: Vec3
    requires_nonprehensile: bool
    required_primitives: Tuple[str, ...]
    max_steps: int
    random_seed: int


LAYOUTS: Tuple[LayoutSpec, ...] = (
    LayoutSpec(
        name="occluded_block",
        description="A movable block sits between the gripper and the target.",
        direct_grasp_blocked_reason="The obstacle blocks the direct approach path.",
        target_position=(0.00, 0.00, 0.02),
        target_yaw=0.0,
        obstacle_specs=(
            ObstacleSpec(
                name="front_block",
                position=(0.00, -0.09, 0.04),
                half_size=(0.04, 0.025, 0.04),
                yaw=0.0,
                movable=True,
            ),
        ),
        place_position=(0.25, -0.14, 0.01),
        place_yaw=0.0,
        place_radius=0.06,
        initial_gripper_position=(0.00, -0.50, 0.30),
        requires_nonprehensile=True,
        required_primitives=("clear", "push", "grasp", "place"),
        max_steps=100,
        random_seed=1001,
    ),
    LayoutSpec(
        name="against_wall",
        description="The target is pressed against a fixed wall.",
        direct_grasp_blocked_reason="The wall prevents a valid approach from the rear side.",
        target_position=(0.03, 0.26, 0.02),
        target_yaw=0.0,
        obstacle_specs=(
            ObstacleSpec(
                name="back_wall",
                position=(0.00, 0.42, 0.18),
                half_size=(0.40, 0.08, 0.18),
                yaw=0.0,
                movable=False,
            ),
        ),
        place_position=(-0.24, -0.12, 0.01),
        place_yaw=0.0,
        place_radius=0.06,
        initial_gripper_position=(0.00, -0.50, 0.30),
        requires_nonprehensile=True,
        required_primitives=("push", "grasp", "place"),
        max_steps=100,
        random_seed=1002,
    ),
    LayoutSpec(
        name="narrow_gap",
        description="The target rests between two movable blocks.",
        direct_grasp_blocked_reason="The side obstacles block a top-down grasp pose.",
        target_position=(-0.02, 0.02, 0.02),
        target_yaw=0.0,
        obstacle_specs=(
            ObstacleSpec(
                name="left_gap_block",
                position=(-0.09, 0.02, 0.04),
                half_size=(0.02, 0.08, 0.04),
                yaw=0.0,
                movable=True,
            ),
            ObstacleSpec(
                name="right_gap_block",
                position=(0.07, 0.02, 0.04),
                half_size=(0.02, 0.08, 0.04),
                yaw=0.0,
                movable=True,
            ),
        ),
        place_position=(0.24, 0.16, 0.01),
        place_yaw=0.0,
        place_radius=0.06,
        initial_gripper_position=(0.00, -0.50, 0.30),
        requires_nonprehensile=True,
        required_primitives=("push", "grasp", "place"),
        max_steps=120,
        random_seed=1003,
    ),
    LayoutSpec(
        name="rotated_object",
        description="The target has an orientation that is not directly graspable.",
        direct_grasp_blocked_reason="The long axis blocks a stable parallel-jaw grasp.",
        target_position=(0.12, 0.05, 0.02),
        target_yaw=0.5 * pi,
        obstacle_specs=(),
        place_position=(-0.22, 0.16, 0.01),
        place_yaw=0.0,
        place_radius=0.06,
        initial_gripper_position=(0.00, -0.50, 0.30),
        requires_nonprehensile=True,
        required_primitives=("rotate", "grasp", "place"),
        max_steps=100,
        random_seed=1004,
    ),
    LayoutSpec(
        name="multi_obstacle",
        description="Several movable blocks must be cleared before grasping.",
        direct_grasp_blocked_reason="The target has no collision-free direct grasp corridor.",
        target_position=(0.02, 0.04, 0.02),
        target_yaw=0.0,
        obstacle_specs=(
            ObstacleSpec(
                name="front_left_block",
                position=(-0.08, -0.02, 0.04),
                half_size=(0.03, 0.03, 0.04),
                yaw=0.2,
                movable=True,
            ),
            ObstacleSpec(
                name="front_right_block",
                position=(0.10, 0.02, 0.04),
                half_size=(0.03, 0.03, 0.04),
                yaw=-0.2,
                movable=True,
            ),
            ObstacleSpec(
                name="rear_block",
                position=(0.02, 0.13, 0.04),
                half_size=(0.04, 0.025, 0.04),
                yaw=0.0,
                movable=True,
            ),
        ),
        place_position=(0.26, -0.16, 0.01),
        place_yaw=0.0,
        place_radius=0.06,
        initial_gripper_position=(0.00, -0.50, 0.30),
        requires_nonprehensile=True,
        required_primitives=("clear", "push", "grasp", "place"),
        max_steps=140,
        random_seed=1005,
    ),
)


LAYOUTS_BY_NAME: Dict[str, LayoutSpec] = {layout.name: layout for layout in LAYOUTS}


def get_layout(name: str) -> LayoutSpec:
    """Return a layout by name and raise a clear error when it is unknown."""

    try:
        return LAYOUTS_BY_NAME[name]
    except KeyError as exc:
        available = ", ".join(sorted(LAYOUTS_BY_NAME))
        raise KeyError(f"Unknown layout {name!r}. Available layouts: {available}") from exc
