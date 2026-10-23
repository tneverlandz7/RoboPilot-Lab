"""A small deterministic visual-navigation environment.

Coordinates are in metres. World x points right and y points down. Heading 0
points right; positive angular velocity turns clockwise in world coordinates.
The policy-facing image is rotated so the robot always faces up.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, degrees, pi, sin
from random import Random

import numpy as np
from PIL import Image, ImageDraw


WORLD_SIZE = 8.0
PIXELS_PER_METRE = 80
ROBOT_RADIUS = 0.18
CAMERA_SIZE = 128
CAMERA_SPAN_METRES = 4.0
STATION_COLORS = {"red": (225, 45, 55), "blue": (45, 95, 225)}


@dataclass(frozen=True)
class Action:
    linear_mps: float
    angular_radps: float


@dataclass(frozen=True)
class Observation:
    image: np.ndarray  # uint8, shape (128, 128, 3); policy-facing data
    instruction: str


@dataclass(frozen=True)
class StepResult:
    observation: Observation
    reward: float
    terminated: bool
    info: dict[str, object]


@dataclass(frozen=True)
class Rectangle:
    left: float
    top: float
    right: float
    bottom: float

    def contains_disc(self, x: float, y: float, radius: float) -> bool:
        closest_x = min(max(x, self.left), self.right)
        closest_y = min(max(y, self.top), self.bottom)
        return (x - closest_x) ** 2 + (y - closest_y) ** 2 < radius**2


class RobotEnv:
    """Simulator state is intentionally separate from Observation.

    The caller building a policy must pass only Observation and previous tool
    results to it. Pose, obstacles and station coordinates are evaluator data.
    """

    def __init__(self, seed: int = 0, max_steps: int = 300):
        self.seed = seed
        self.max_steps = max_steps
        self.reset(seed)

    def reset(self, seed: int | None = None) -> Observation:
        if seed is not None:
            self.seed = seed
        rng = Random(self.seed)
        self.pose = [1.2, 1.4, 0.0]
        self.stations = {
            "red": (2.45 + rng.uniform(-0.12, 0.12), 1.45),
            "blue": (6.45, 6.35 + rng.uniform(-0.25, 0.25)),
        }
        self.obstacles = (
            Rectangle(3.7, 0.9, 4.4, 3.3),
            Rectangle(3.7, 4.6, 4.4, 7.1),
            Rectangle(5.6, 3.4, 6.7, 4.0),
        )
        self.goals = ("red", "blue")
        self.goal_index = 0
        self.steps = 0
        self.collisions = 0
        self.terminated = False
        return self.observe()

    def observe(self) -> Observation:
        full = self._draw_world(show_robot=False)
        pad = round(CAMERA_SPAN_METRES * PIXELS_PER_METRE)
        padded = Image.new("RGB", (full.width + 2 * pad, full.height + 2 * pad), (242, 242, 238))
        padded.paste(full, (pad, pad))
        cx = round(self.pose[0] * PIXELS_PER_METRE) + pad
        cy = round(self.pose[1] * PIXELS_PER_METRE) + pad
        rotated = padded.rotate(90 + degrees(self.pose[2]), center=(cx, cy), fillcolor=(242, 242, 238))
        half = round(CAMERA_SPAN_METRES * PIXELS_PER_METRE / 2)
        crop = rotated.crop((cx - half, cy - half, cx + half, cy + half))
        crop = crop.resize((CAMERA_SIZE, CAMERA_SIZE), Image.Resampling.BILINEAR)
        draw = ImageDraw.Draw(crop)
        mid = CAMERA_SIZE // 2
        draw.polygon([(mid, mid - 7), (mid - 5, mid + 5), (mid + 5, mid + 5)], fill=(25, 115, 65))
        return Observation(np.asarray(crop, dtype=np.uint8).copy(), "Visit red, then blue.")

    def step(self, action: Action) -> StepResult:
        if self.terminated:
            raise RuntimeError("Episode has ended; call reset()")
        if not isinstance(action, Action):
            raise TypeError("action must be an Action")
        if not (-1.0 <= action.linear_mps <= 1.0 and -pi <= action.angular_radps <= pi):
            raise ValueError("Action exceeds velocity limits")

        dt = 0.1
        x, y, heading = self.pose
        new_heading = (heading + action.angular_radps * dt + pi) % (2 * pi) - pi
        new_x = x + action.linear_mps * cos(new_heading) * dt
        new_y = y + action.linear_mps * sin(new_heading) * dt
        collision = self._collides(new_x, new_y)
        if collision:
            self.collisions += 1
            new_x, new_y = x, y
        self.pose = [new_x, new_y, new_heading]
        self.steps += 1

        reached: str | None = None
        target = self.goals[self.goal_index]
        tx, ty = self.stations[target]
        if (new_x - tx) ** 2 + (new_y - ty) ** 2 <= 0.38**2:
            reached = target
            self.goal_index += 1

        self.terminated = self.goal_index == len(self.goals) or self.steps >= self.max_steps
        reward = (1.0 if reached else 0.0) - (0.1 if collision else 0.0) - 0.001
        return StepResult(
            self.observe(),
            reward,
            self.terminated,
            {"collision": collision, "reached": reached, "steps": self.steps},
        )

    def _collides(self, x: float, y: float) -> bool:
        if not (ROBOT_RADIUS <= x <= WORLD_SIZE - ROBOT_RADIUS):
            return True
        if not (ROBOT_RADIUS <= y <= WORLD_SIZE - ROBOT_RADIUS):
            return True
        return any(obstacle.contains_disc(x, y, ROBOT_RADIUS) for obstacle in self.obstacles)

    def debug_map(self) -> Image.Image:
        """Full map for debugging and evaluation only; never pass to a policy."""
        image = self._draw_world(show_robot=True)
        return image

    def _draw_world(self, show_robot: bool) -> Image.Image:
        edge = round(WORLD_SIZE * PIXELS_PER_METRE)
        image = Image.new("RGB", (edge, edge), (242, 242, 238))
        draw = ImageDraw.Draw(image)
        for obstacle in self.obstacles:
            draw.rectangle(
                (
                    round(obstacle.left * PIXELS_PER_METRE),
                    round(obstacle.top * PIXELS_PER_METRE),
                    round(obstacle.right * PIXELS_PER_METRE),
                    round(obstacle.bottom * PIXELS_PER_METRE),
                ),
                fill=(55, 60, 68),
            )
        for name, (x, y) in self.stations.items():
            px, py = round(x * PIXELS_PER_METRE), round(y * PIXELS_PER_METRE)
            draw.ellipse((px - 14, py - 14, px + 14, py + 14), fill=STATION_COLORS[name])
        if show_robot:
            x, y, heading = self.pose
            px, py = x * PIXELS_PER_METRE, y * PIXELS_PER_METRE
            draw.ellipse((px - 12, py - 12, px + 12, py + 12), fill=(25, 115, 65))
            draw.line((px, py, px + 24 * cos(heading), py + 24 * sin(heading)), fill=(250, 250, 250), width=4)
        return image
