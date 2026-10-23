import numpy as np
from robopilot.env import Action, Observation


def choose_action(observation: Observation, target: str) -> Action:
    if target != "red":
        raise ValueError("Currently only red is supported")

    center = find_red_center(observation.image)
    if center is None:
        return Action(linear_mps=0.0, angular_radps=0.8)  # 看不到目标：原地搜索

    x, y = center
    half_width = observation.image.shape[1] / 2
    half_height = observation.image.shape[0] / 2

    error_x = (x - half_width) / half_width
    error_y = (half_height - y) / half_height

    linear = max(0.0, min(0.6, 0.8 * error_y))
    angular = 1.5 * error_x
    return Action(linear_mps=linear, angular_radps=angular)


def find_red_center(image: np.ndarray) -> tuple[float, float] | None:
    mask = (
        (image[:, :, 0] > 170)
        & (image[:, :, 1] < 100)
        & (image[:, :, 2] < 100)
    )
    ys, xs = np.where(mask)

    if xs.size == 0:
        return None

    return float(xs.mean()), float(ys.mean())