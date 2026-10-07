"""Reproducible KITTI LiDAR-camera calibration drift experiment.

Run from the repository root: python -m src.topic_a --help
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from starter.datasets import list_frames, load_frame
from starter.projection import (cam_to_image, draw_box2d, overlay_points,
                                perturb_extrinsic, velo_to_cam)

CLASSES = {"Car", "Van", "Truck", "Pedestrian", "Cyclist"}
YAW_LEVELS = (0.0, 0.5, 1.0, 2.0, 3.0)
DEMO_FRAMES = ("000019", "000011", "000009")


def in_box(points_cam: np.ndarray, obj) -> np.ndarray:
    """KITTI box location is bottom center in rectified camera coordinates."""
    local = points_cam - obj.location
    c, s = np.cos(obj.rotation_y), np.sin(obj.rotation_y)
    x = c * local[:, 0] - s * local[:, 2]
    z = s * local[:, 0] + c * local[:, 2]
    h, w, length = obj.dimensions
    return ((np.abs(x) <= length / 2) & (np.abs(z) <= w / 2)
            & (local[:, 1] >= -h) & (local[:, 1] <= 0))


def projected_pixels(points_cam: np.ndarray, P2: np.ndarray, shape) -> tuple[np.ndarray, np.ndarray]:
    """Restore one pixel row per input point using cam_to_image's mask."""
    uv, _, mask = cam_to_image(points_cam, P2, shape)
    all_uv = np.full((len(points_cam), 2), np.nan)
    all_uv[mask] = uv
    return all_uv, mask


def inside_bbox(uv: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    return ((uv[:, 0] >= bbox[0]) & (uv[:, 0] <= bbox[2])
            & (uv[:, 1] >= bbox[1]) & (uv[:, 1] <= bbox[3]))


def object_samples(frame: dict, min_points: int) -> list[dict]:
    points = frame["points"][:, :3]
    finite = np.isfinite(points).all(axis=1)
    clean_points = points[finite]
    camera_points = velo_to_cam(clean_points, frame["calib"])
    samples = []
    for index, obj in enumerate(frame["labels"]):
        if obj.type not in CLASSES or not 5 <= obj.location[2] <= 50:
            continue
        selected = in_box(camera_points, obj)
        if selected.sum() < min_points:
            continue
        lidar_points = clean_points[selected]
        baseline_uv, _ = projected_pixels(camera_points[selected], frame["calib"].P2,
                                          frame["image"].shape)
        support = inside_bbox(baseline_uv, obj.bbox)
        if support.sum() < min_points:
            continue
        samples.append({"index": index, "object": obj,
                        "lidar_points": lidar_points[support],
                        "baseline_uv": baseline_uv[support]})
    return samples


def evaluate(frame: dict, samples: list[dict], yaw: float) -> tuple[dict, list[dict]]:
    calib = perturb_extrinsic(frame["calib"], yaw_deg=yaw)
    per_object = []
    shifts = []
    for sample in samples:
        obj = sample["object"]
        camera_points = velo_to_cam(sample["lidar_points"], calib)
        uv, visible = projected_pixels(camera_points, calib.P2, frame["image"].shape)
        retained = visible & inside_bbox(uv, obj.bbox)
        both = visible & np.isfinite(sample["baseline_uv"]).all(axis=1)
        shifts.extend(np.linalg.norm(uv[both] - sample["baseline_uv"][both], axis=1))
        per_object.append({"frame_id": frame["frame_id"], "object_index": sample["index"],
                           "class": obj.type, "range_m": round(float(obj.location[2]), 2),
                           "yaw_deg": yaw, "baseline_support": len(sample["lidar_points"]),
                           "visible_points": int(visible.sum()),
                           "retained_points": int(retained.sum()),
                           "retention_pct": round(100 * retained.mean(), 3)})
    support = sum(row["baseline_support"] for row in per_object)
    visible = sum(row["visible_points"] for row in per_object)
    retained = sum(row["retained_points"] for row in per_object)
    return {"yaw_deg": yaw, "frames": 1, "objects": len(samples),
            "baseline_support": support, "visible_points": visible,
            "retained_points": retained,
            "retention_pct": 100 * retained / support if support else np.nan,
            "median_shift_px": float(np.median(shifts)) if shifts else np.nan,
            "shifts": shifts}, per_object


def save_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: row[key] for key in columns} for row in rows)


def draw_overlay(frame: dict, yaw: float, highlight_index: int | None = None) -> np.ndarray:
    calib = perturb_extrinsic(frame["calib"], yaw_deg=yaw)
    points = frame["points"]
    points = points[np.isfinite(points[:, :3]).all(axis=1)]
    uv, depth, _ = cam_to_image(velo_to_cam(points[:, :3], calib), calib.P2,
                                frame["image"].shape)
    out = overlay_points(frame["image"], uv, depth, radius=2)
    for index, obj in enumerate(frame["labels"]):
        if obj.type in CLASSES and 5 <= obj.location[2] <= 50:
            color = (0, 0, 255) if index == highlight_index else (0, 255, 0)
            out = draw_box2d(out, obj.bbox, color=color, label=obj.type)
    return out


def write_failure(frame: dict, row: dict, figure_dir: Path,
                  min_points: int) -> None:
    sample = next(s for s in object_samples(frame, min_points)
                  if s["index"] == row["object_index"])
    obj = frame["labels"][row["object_index"]]
    base = draw_box2d(frame["image"], obj.bbox, color=(0, 0, 255))
    drift = base.copy()
    drift_calib = perturb_extrinsic(frame["calib"], yaw_deg=row["yaw_deg"])
    drift_uv, visible = projected_pixels(
        velo_to_cam(sample["lidar_points"], drift_calib), drift_calib.P2,
        frame["image"].shape)
    for point in sample["baseline_uv"]:
        cv2.circle(base, tuple(np.rint(point).astype(int)), 4, (255, 255, 0), -1)
    for point in drift_uv[visible]:
        cv2.circle(drift, tuple(np.rint(point).astype(int)), 4, (255, 255, 0), -1)
    x1, y1, x2, y2 = obj.bbox.astype(int)
    h, w = base.shape[:2]
    margin = 80
    left, top = max(0, x1 - margin), max(0, y1 - margin)
    right, bottom = min(w, x2 + margin), min(h, y2 + margin)
    base_crop = base[top:bottom, left:right]
    drift_crop = drift[top:bottom, left:right]
    if base_crop.size == 0:
        raise ValueError("Failure object has no visible image crop")
    scale = 560 / max(base_crop.shape[1], 1)
    size = (560, max(180, int(base_crop.shape[0] * scale)))
    base_crop = cv2.resize(base_crop, size)
    drift_crop = cv2.resize(drift_crop, size)
    header = np.full((70, 1120, 3), 255, dtype=np.uint8)
    cv2.putText(header, f"Frame {frame['frame_id']} | {row['class']} | {row['range_m']:.1f} m",
                (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (30, 30, 30), 2)
    cv2.putText(header, f"Baseline: 100% support", (15, 59),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 30, 30), 2)
    cv2.putText(header, f"Yaw {row['yaw_deg']:g} deg: {row['retention_pct']:.1f}% retained",
                (575, 59), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 30, 30), 2)
    cv2.imwrite(str(figure_dir / "fail_01_yaw_drift.png"),
                np.vstack((header, np.hstack((base_crop, drift_crop)))))


def plot_sweep(rows: list[dict], figure_dir: Path) -> None:
    yaw = [row["yaw_deg"] for row in rows]
    retention = [row["retention_pct"] for row in rows]
    shift = [row["median_shift_px"] for row in rows]
    fig, left = plt.subplots(figsize=(8, 4.8))
    left.plot(yaw, retention, "o-", color="#007f78", linewidth=2, label="Box retention")
    left.set(xlabel="Yaw drift (degrees)", ylabel="LiDAR point retention in 2D GT box (%)",
             ylim=(0, 105))
    left.grid(alpha=0.25)
    right = left.twinx()
    right.plot(yaw, shift, "s--", color="#c85036", linewidth=2, label="Median shift")
    right.set_ylabel("Median pixel shift (px)")
    fig.suptitle(f"Calibration drift on KITTI mini ({rows[0]['frames']} frames)")
    fig.tight_layout()
    fig.savefig(figure_dir / "yaw_sweep.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/kitti_mini")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--frames", nargs="*", help="Frame IDs; default: all frames")
    parser.add_argument("--min-points", type=int, default=5,
                        help="Minimum baseline LiDAR support per labeled object")
    args = parser.parse_args()
    if args.min_points < 1:
        parser.error("--min-points must be positive")
    frame_ids = args.frames if args.frames else list_frames(args.data_root)
    if not frame_ids:
        parser.error("No frames found")
    out_dir = Path(args.out_dir)
    figure_dir = out_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    totals = {yaw: {"yaw_deg": yaw, "frames": 0, "objects": 0,
                    "baseline_support": 0, "visible_points": 0,
                    "retained_points": 0, "shifts": []}
              for yaw in YAW_LEVELS}
    object_rows = []
    failure_candidates = []
    for frame_id in frame_ids:
        frame = load_frame(args.data_root, frame_id)
        samples = object_samples(frame, args.min_points)
        for yaw in YAW_LEVELS:
            metrics, per_object = evaluate(frame, samples, yaw)
            total = totals[yaw]
            for key in ("frames", "objects", "baseline_support", "visible_points",
                        "retained_points"):
                total[key] += metrics[key]
            total["shifts"].extend(metrics["shifts"])
            object_rows.extend(per_object)
            if yaw == 2.0:
                failure_candidates.extend(per_object)
        if frame_id in DEMO_FRAMES:
            cv2.imwrite(str(figure_dir / f"demo_{frame_id}_baseline.png"),
                        draw_overlay(frame, 0.0))
    rows = []
    for yaw in YAW_LEVELS:
        total = totals[yaw]
        if not total["baseline_support"]:
            raise SystemExit("No eligible 3D boxes; check dataset or --min-points")
        total["retention_pct"] = round(100 * total["retained_points"] /
                                       total["baseline_support"], 3)
        total["visible_pct"] = round(100 * total["visible_points"] /
                                     total["baseline_support"], 3)
        total["median_shift_px"] = round(float(np.median(total["shifts"])), 3)
        rows.append(total)
    columns = ["yaw_deg", "frames", "objects", "baseline_support", "visible_points",
               "visible_pct", "retained_points", "retention_pct", "median_shift_px"]
    save_csv(out_dir / "yaw_perturb_sweep.csv", rows, columns)
    save_csv(out_dir / "per_object.csv", object_rows,
             ["frame_id", "object_index", "class", "range_m", "yaw_deg",
              "baseline_support", "visible_points", "retained_points", "retention_pct"])
    plot_sweep(rows, figure_dir)
    failure = min(failure_candidates, key=lambda row: row["retention_pct"])
    write_failure(load_frame(args.data_root, failure["frame_id"]), failure,
                  figure_dir, args.min_points)
    for row in rows:
        print(f"yaw={row['yaw_deg']:>3g} deg  retained={row['retention_pct']:>6.2f}% "
              f"median_shift={row['median_shift_px']:>6.2f}px  "
              f"objects={row['objects']}")
    print(f"Failure: {failure['frame_id']} {failure['class']} "
          f"({failure['range_m']} m), {failure['retention_pct']}% retained at 2 deg")
    print(f"Results: {out_dir}")


if __name__ == "__main__":
    main()
