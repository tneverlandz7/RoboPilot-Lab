"""Regenerate the README preview from the deterministic seed-7 episode."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from robopilot.env import RobotEnv


def font(size: int) -> ImageFont.ImageFont:
    path = Path("C:/Windows/Fonts/arial.ttf")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def main() -> None:
    env = RobotEnv(seed=7)
    debug = env.debug_map().resize((540, 540))
    camera = Image.fromarray(env.observe().image).resize((540, 540), Image.Resampling.NEAREST)
    canvas = Image.new("RGB", (1160, 725), (250, 250, 248))
    draw = ImageDraw.Draw(canvas)
    title_font, label_font = font(31), font(20)
    draw.text((40, 23), "RoboPilot Lab", fill=(28, 36, 46), font=title_font)
    draw.text((40, 72), "One environment, two views", fill=(90, 98, 107), font=label_font)
    draw.text((40, 111), "Evaluator-only debug map", fill=(28, 36, 46), font=label_font)
    draw.text((610, 111), "Policy-facing local RGB view", fill=(28, 36, 46), font=label_font)
    canvas.paste(debug, (40, 145))
    canvas.paste(camera, (610, 145))
    draw.rectangle((39, 144, 581, 686), outline=(190, 194, 198), width=2)
    draw.rectangle((609, 144, 1151, 686), outline=(190, 194, 198), width=2)
    output = Path("docs/assets/preview.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    print(output.resolve())


if __name__ == "__main__":
    main()
