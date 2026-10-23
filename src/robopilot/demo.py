"""Write a tiny simulator preview without a trained policy."""

from pathlib import Path

from PIL import Image

from .env import Action, RobotEnv


def main() -> None:
    output = Path("results")
    output.mkdir(exist_ok=True)
    env = RobotEnv(seed=7)
    Image.fromarray(env.observe().image).resize((512, 512)).save(output / "observation_start.png")
    env.debug_map().save(output / "debug_map_start.png")
    for _ in range(8):
        env.step(Action(linear_mps=0.5, angular_radps=0.0))
    Image.fromarray(env.observe().image).resize((512, 512)).save(output / "observation_after_8_steps.png")
    print(f"Saved previews to {output.resolve()}")
    print("The debug map is evaluator-only data; a policy must not read it.")


if __name__ == "__main__":
    main()
