import unittest

import numpy as np

from robopilot.env import Observation
from robopilot.baselines.color_tracker import choose_action


def observation_with_red_at(x: int | None) -> Observation:
    image = np.full((128, 128, 3), 242, dtype=np.uint8)
    if x is not None:
        image[20:30, x - 5:x + 5] = (225, 45, 55)
    return Observation(image=image, instruction="Visit red.")


class ColorTrackerTests(unittest.TestCase):
    def test_turns_toward_red_target(self):
        left = choose_action(observation_with_red_at(24), "red")
        right = choose_action(observation_with_red_at(104), "red")

        self.assertLess(left.angular_radps, 0)
        self.assertGreater(right.angular_radps, 0)

    def test_searches_without_moving_when_target_missing(self):
        action = choose_action(observation_with_red_at(None), "red")

        self.assertEqual(action.linear_mps, 0.0)
        self.assertGreater(action.angular_radps, 0)


if __name__ == "__main__":
    unittest.main()