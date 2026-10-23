import unittest

import numpy as np

from robopilot.env import Action, RobotEnv


class RobotEnvTests(unittest.TestCase):
    def test_reset_is_deterministic(self):
        first = RobotEnv(seed=17).observe().image
        second = RobotEnv(seed=17).observe().image
        self.assertTrue(np.array_equal(first, second))

    def test_observation_has_only_policy_data(self):
        observation = RobotEnv(seed=0).observe()
        self.assertEqual(observation.image.shape, (128, 128, 3))
        self.assertEqual(observation.image.dtype, np.uint8)
        self.assertEqual(set(vars(observation)), {"image", "instruction"})

    def test_forward_action_changes_pose(self):
        env = RobotEnv(seed=0)
        old_x = env.pose[0]
        result = env.step(Action(0.5, 0.0))
        self.assertGreater(env.pose[0], old_x)
        self.assertFalse(result.info["collision"])

    def test_wall_collision_blocks_motion(self):
        env = RobotEnv(seed=0)
        env.pose = [7.8, 2.0, 0.0]
        old_x = env.pose[0]
        result = env.step(Action(1.0, 0.0))
        self.assertEqual(env.pose[0], old_x)
        self.assertTrue(result.info["collision"])

    def test_excess_velocity_is_rejected(self):
        env = RobotEnv(seed=0)
        with self.assertRaises(ValueError):
            env.step(Action(2.0, 0.0))


if __name__ == "__main__":
    unittest.main()
