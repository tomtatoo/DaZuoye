"""High-level primitive action definitions for push-grasp-place planning."""

from dataclasses import dataclass, field
from math import isfinite
from typing import Any, Dict, Iterable, Optional, Tuple


@dataclass(frozen=True)
class PrimitiveSpec:
    """Static contract for one high-level manipulation primitive."""

    name: str
    description: str
    required_params: Tuple[str, ...]
    optional_params: Tuple[str, ...] = ()
    default_duration_steps: int = 10


PRIMITIVE_SPECS: Dict[str, PrimitiveSpec] = {
    "push": PrimitiveSpec(
        name="push",
        description="Move the target or another object along a direction.",
        required_params=("direction", "distance"),
        optional_params=("speed",),
        default_duration_steps=8,
    ),
    "clear": PrimitiveSpec(
        name="clear",
        description="Move one obstacle away from the target approach corridor.",
        required_params=("obstacle_id", "direction", "distance"),
        optional_params=("speed",),
        default_duration_steps=10,
    ),
    "rotate": PrimitiveSpec(
        name="rotate",
        description="Rotate an object in place.",
        required_params=("yaw_delta",),
        optional_params=("center", "speed"),
        default_duration_steps=8,
    ),
    "grasp": PrimitiveSpec(
        name="grasp",
        description="Close the gripper around the target object.",
        required_params=("target_id",),
        optional_params=("gripper_width", "grasp_height"),
        default_duration_steps=12,
    ),
    "lift": PrimitiveSpec(
        name="lift",
        description="Lift the grasped object.",
        required_params=("height",),
        optional_params=("speed",),
        default_duration_steps=10,
    ),
    "move_to": PrimitiveSpec(
        name="move_to",
        description="Move the end effector to a target pose.",
        required_params=("position",),
        optional_params=("orientation", "speed"),
        default_duration_steps=20,
    ),
    "place": PrimitiveSpec(
        name="place",
        description="Move the grasped object into the placement region.",
        required_params=("position",),
        optional_params=("yaw", "height"),
        default_duration_steps=18,
    ),
    "release": PrimitiveSpec(
        name="release",
        description="Open the gripper and release the object.",
        required_params=(),
        optional_params=("gripper_width",),
        default_duration_steps=8,
    ),
    "retreat": PrimitiveSpec(
        name="retreat",
        description="Move the end effector away after placement.",
        required_params=("direction", "distance"),
        optional_params=("speed",),
        default_duration_steps=10,
    ),
    "wait": PrimitiveSpec(
        name="wait",
        description="Hold position and allow physics to settle.",
        required_params=(),
        optional_params=("duration_steps",),
        default_duration_steps=5,
    ),
}


def _as_finite_float(value: Any, param_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"Primitive parameter {param_name!r} must be numeric")
    value = float(value)
    if not isfinite(value):
        raise ValueError(f"Primitive parameter {param_name!r} must be finite")
    return value


def _validate_vector(value: Any, size: int, param_name: str) -> None:
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ValueError(f"Primitive parameter {param_name!r} must have length {size}")
    for item in value:
        _as_finite_float(item, param_name)


def _validate_primitive(name: str, params: Dict[str, Any]) -> None:
    if name not in PRIMITIVE_SPECS:
        available = ", ".join(sorted(PRIMITIVE_SPECS))
        raise KeyError(f"Unknown primitive {name!r}. Available primitives: {available}")

    spec = PRIMITIVE_SPECS[name]
    missing = [key for key in spec.required_params if key not in params]
    if missing:
        raise ValueError(f"Primitive {name!r} is missing required parameters: {missing}")

    allowed = set(spec.required_params) | set(spec.optional_params)
    unknown = set(params) - allowed
    if unknown:
        raise ValueError(f"Primitive {name!r} has unknown parameters: {sorted(unknown)}")

    if "direction" in params:
        _validate_vector(params["direction"], 2, "direction")
        dx, dy = params["direction"]
        if abs(float(dx)) + abs(float(dy)) <= 1e-8:
            raise ValueError("Primitive direction must be non-zero")

    if "position" in params:
        _validate_vector(params["position"], 3, "position")

    if "center" in params:
        _validate_vector(params["center"], 2, "center")

    if "orientation" in params:
        _validate_vector(params["orientation"], 3, "orientation")

    for key in ("distance", "height", "gripper_width", "grasp_height", "speed", "duration_steps"):
        if key in params:
            value = _as_finite_float(params[key], key)
            if value < 0:
                raise ValueError(f"Primitive parameter {key!r} must be non-negative")

    if "distance" in params and float(params["distance"]) <= 0:
        raise ValueError("Primitive parameter 'distance' must be positive")

    if "height" in params and float(params["height"]) <= 0:
        raise ValueError("Primitive parameter 'height' must be positive")

    if "yaw_delta" in params:
        _as_finite_float(params["yaw_delta"], "yaw_delta")

    if "yaw" in params:
        _as_finite_float(params["yaw"], "yaw")

    for key in ("target_id", "obstacle_id"):
        if key in params and (not isinstance(params[key], str) or not params[key]):
            raise ValueError(f"Primitive parameter {key!r} must be a non-empty string")


@dataclass
class PrimitiveAction:
    """One executable high-level action with parameters and a duration."""

    name: str
    params: Dict[str, Any] = field(default_factory=dict)
    duration_steps: Optional[int] = None

    def __post_init__(self) -> None:
        self.name = str(self.name).strip().lower()
        self.params = dict(self.params)
        _validate_primitive(self.name, self.params)
        if self.duration_steps is None:
            self.duration_steps = PRIMITIVE_SPECS[self.name].default_duration_steps
        if isinstance(self.duration_steps, bool) or int(self.duration_steps) != self.duration_steps:
            raise TypeError("duration_steps must be an integer")
        self.duration_steps = int(self.duration_steps)
        if self.duration_steps <= 0:
            raise ValueError("duration_steps must be positive")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "params": dict(self.params),
            "duration_steps": self.duration_steps,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PrimitiveAction":
        return cls(
            name=data["name"],
            params=data.get("params", {}),
            duration_steps=data.get("duration_steps"),
        )


@dataclass
class PrimitiveSequence:
    """An ordered sequence of high-level primitive actions."""

    actions: Tuple[PrimitiveAction, ...]

    def __post_init__(self) -> None:
        if not self.actions:
            raise ValueError("PrimitiveSequence must contain at least one action")
        self.actions = tuple(self.actions)
        if not all(isinstance(action, PrimitiveAction) for action in self.actions):
            raise TypeError("PrimitiveSequence actions must be PrimitiveAction objects")

    @property
    def total_duration_steps(self) -> int:
        return sum(action.duration_steps for action in self.actions)

    def to_dict(self) -> Dict[str, Any]:
        return {"actions": [action.to_dict() for action in self.actions]}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PrimitiveSequence":
        return cls(tuple(PrimitiveAction.from_dict(item) for item in data["actions"]))

    @classmethod
    def from_iterable(cls, actions: Iterable[PrimitiveAction]) -> "PrimitiveSequence":
        return cls(tuple(actions))


def make_primitive(name: str, **params: Any) -> PrimitiveAction:
    """Convenience constructor used by planners and tests."""

    duration_steps = params.pop("duration_steps", None)
    return PrimitiveAction(name=name, params=params, duration_steps=duration_steps)


def validate_sequence(sequence: PrimitiveSequence) -> None:
    """Validate an already constructed sequence, including required stages."""

    names = [action.name for action in sequence.actions]
    if "grasp" not in names:
        raise ValueError("A complete push-grasp-place sequence must contain 'grasp'")
    if "place" not in names:
        raise ValueError("A complete push-grasp-place sequence must contain 'place'")
    if names.index("grasp") > names.index("place"):
        raise ValueError("'grasp' must occur before 'place'")
