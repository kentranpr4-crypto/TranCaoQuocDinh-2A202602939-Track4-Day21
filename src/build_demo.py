"""Build a short GIF and MP4 from the real topic A projection overlays.

Run from repository root: python -m src.build_demo
"""
from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from starter.datasets import load_frame
from src.topic_a import YAW_LEVELS, draw_overlay

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "figures"
SCENES = (
    ("000019", "NEAR FIELD", "Objects at close range"),
    ("000011", "PEDESTRIAN SCENE", "Mixed street traffic"),
    ("000001", "LONG RANGE", "Cyclist at 45.8 m"),
)
W, H = 960, 450
TEAL = (52, 213, 189)
CORAL = (255, 117, 88)
INK = (18, 27, 34)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "Lato-Bold.ttf" if bold else "Lato-Regular.ttf"
    for path in (Path("/usr/share/fonts/truetype/lato") / name,
                 Path("/usr/share/fonts/truetype/dejavu") /
                 ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")):
        if path.exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def panel(frame: dict, frame_id: str, category: str, description: str,
          yaw: float, retention: float, shift: float) -> Image.Image:
    overlay = draw_overlay(frame, yaw)
    photo = Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    photo = photo.resize((W, 290), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (W, H), INK)
    canvas.paste(photo, (0, 72))
    draw = ImageDraw.Draw(canvas)
    draw.text((28, 16), "LIDAR / CAMERA", font=font(24, True), fill=(247, 250, 250))
    draw.text((W - 29, 21), f"KITTI  /  {frame_id}", font=font(16, True),
              fill=(180, 198, 201), anchor="ra")
    draw.rectangle((0, 72, W, 75), fill=TEAL)
    draw.text((28, 381), category, font=font(14, True), fill=TEAL)
    draw.text((28, 410), description, font=font(17), fill=(213, 224, 225))
    draw.text((486, 381), "YAW DRIFT", font=font(14, True), fill=(166, 185, 187))
    draw.text((486, 408), f"{yaw:g}°", font=font(28, True), fill=(247, 250, 250))
    draw.text((625, 381), "BOX RETENTION", font=font(14, True), fill=(166, 185, 187))
    draw.text((625, 408), f"{retention:.1f}%", font=font(28, True),
              fill=TEAL if yaw < 2 else CORAL)
    draw.text((824, 381), "PIXEL SHIFT", font=font(14, True), fill=(166, 185, 187))
    draw.text((824, 408), f"{shift:.1f}px", font=font(25, True), fill=(247, 250, 250))
    for i, level in enumerate(YAW_LEVELS):
        x = 488 + i * 26
        draw.ellipse((x, 441, x + 8, 449), fill=TEAL if level <= yaw else (75, 92, 98))
    return canvas


def main() -> None:
    with (ROOT / "results" / "yaw_perturb_sweep.csv").open(newline="", encoding="utf-8") as file:
        metrics = {float(row["yaw_deg"]): row for row in csv.DictReader(file)}
    frames = []
    for frame_id, category, description in SCENES:
        frame = load_frame(ROOT / "data" / "kitti_mini", frame_id)
        for yaw in YAW_LEVELS:
            row = metrics[yaw]
            frames.append(panel(frame, frame_id, category, description, yaw,
                                float(row["retention_pct"]),
                                float(row["median_shift_px"])))
    OUT.mkdir(parents=True, exist_ok=True)
    gif = OUT / "calibration_drift_demo.gif"
    frames[0].save(gif, save_all=True, append_images=frames[1:], duration=700,
                   loop=0, optimize=True)

    video = OUT / "calibration_drift_demo.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 12, (W, H))
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not open MP4 writer")
    for frame in frames:
        bgr = cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR)
        for _ in range(8):
            writer.write(bgr)
    writer.release()
    print(f"GIF: {gif} ({gif.stat().st_size / 1e6:.1f} MB)")
    print(f"MP4: {video} ({video.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
