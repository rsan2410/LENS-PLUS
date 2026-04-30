#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any, Callable
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import numpy as np
    from PIL import Image, ImageEnhance, ImageFilter
    from ultralytics import YOLO
except ImportError as exc:
    raise SystemExit(
        "Missing dependency for robustness eval. Install with: "
        "pip install ultralytics pillow numpy\n"
        f"Import error: {exc}"
    ) from exc

from app.evaluation.detection_metrics import (
    GroundTruthBox,
    PredictedBox,
    evaluate_detection_suite,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run robustness evaluation on perturbed images.")
    parser.add_argument("--model", default="yolo11s.pt", help="Model path/name.")
    parser.add_argument(
        "--input",
        required=True,
        help="JSON path with image paths and ground-truth boxes.",
    )
    parser.add_argument("--imgsz", type=int, default=960, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--output", default="", help="Optional output report path.")
    return parser.parse_args()


def _to_bbox(raw_bbox: list[Any]) -> tuple[float, float, float, float]:
    if len(raw_bbox) != 4:
        raise ValueError("bbox must have four values.")
    return tuple(float(value) for value in raw_bbox)  # type: ignore[return-value]


def _load_samples(path: Path) -> tuple[list[dict[str, Any]], list[GroundTruthBox]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_images = payload.get("images", [])
    if not isinstance(raw_images, list):
        raise ValueError("'images' must be a list.")

    image_entries: list[dict[str, Any]] = []
    all_gts: list[GroundTruthBox] = []
    for item in raw_images:
        if not isinstance(item, dict):
            raise ValueError("Each image entry must be an object.")
        image_id = str(item["image_id"])
        image_path = str(item["path"])
        gts = item.get("ground_truths", [])
        if not isinstance(gts, list):
            raise ValueError("ground_truths must be a list.")
        image_entries.append({"image_id": image_id, "path": image_path})
        for gt in gts:
            all_gts.append(
                GroundTruthBox(
                    image_id=image_id,
                    label=str(gt["label"]),
                    bbox_xyxy=_to_bbox(gt["bbox"]),
                )
            )
    return image_entries, all_gts


def _predict_image(
    model: YOLO, image_path: Path, *, imgsz: int, conf: float, image_id: str
) -> list[PredictedBox]:
    result = model.predict(str(image_path), imgsz=imgsz, conf=conf, verbose=False)[0]
    predictions: list[PredictedBox] = []
    names = model.names
    if result.boxes is None:
        return predictions

    for box in result.boxes:
        x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
        class_id = int(box.cls.item())
        label = names[class_id] if isinstance(names, dict) else str(class_id)
        score = float(box.conf.item())
        predictions.append(
            PredictedBox(
                image_id=image_id,
                label=str(label),
                bbox_xyxy=(x1, y1, x2, y2),
                score=score,
            )
        )
    return predictions


def _save_temp_image(image: Image.Image, suffix: str) -> Path:
    fd, temp_path = tempfile.mkstemp(prefix="lens_robust_", suffix=suffix)
    Path(temp_path).unlink(missing_ok=True)
    output = Path(temp_path)
    image.save(output)
    return output


def _perturbations() -> dict[str, Callable[[Image.Image], Image.Image]]:
    return {
        "clean": lambda image: image.copy(),
        "brightness_down": lambda image: ImageEnhance.Brightness(image).enhance(0.5),
        "contrast_down": lambda image: ImageEnhance.Contrast(image).enhance(0.6),
        "gaussian_blur": lambda image: image.filter(ImageFilter.GaussianBlur(radius=2)),
        "jpeg_artifact": lambda image: _jpeg_compress_roundtrip(image, quality=30),
        "noise": lambda image: _add_gaussian_noise(image, sigma=18.0),
    }


def _jpeg_compress_roundtrip(image: Image.Image, quality: int) -> Image.Image:
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as tmp:
        image.save(tmp.name, quality=quality)
        compressed = Image.open(tmp.name)
        return compressed.convert("RGB")


def _add_gaussian_noise(image: Image.Image, sigma: float) -> Image.Image:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    noise = np.random.normal(0.0, sigma, size=arr.shape)
    noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)


def main() -> int:
    args = parse_args()
    np.random.seed(42)
    model = YOLO(args.model)
    image_entries, ground_truths = _load_samples(Path(args.input).resolve())
    if not image_entries:
        raise SystemExit("No images found in input.")

    perturbations = _perturbations()
    report: dict[str, Any] = {}

    for perturb_name, transform in perturbations.items():
        predictions: list[PredictedBox] = []
        temp_paths: list[Path] = []
        try:
            for entry in image_entries:
                image_path = Path(entry["path"]).resolve()
                image_id = entry["image_id"]
                if not image_path.exists():
                    raise FileNotFoundError(f"Image file missing: {image_path}")

                image = Image.open(image_path).convert("RGB")
                transformed = transform(image)
                temp_image_path = _save_temp_image(transformed, suffix=".jpg")
                temp_paths.append(temp_image_path)
                predictions.extend(
                    _predict_image(
                        model,
                        temp_image_path,
                        imgsz=args.imgsz,
                        conf=args.conf,
                        image_id=image_id,
                    )
                )
        finally:
            for temp_path in temp_paths:
                temp_path.unlink(missing_ok=True)

        report[perturb_name] = evaluate_detection_suite(predictions, ground_truths)

    print(json.dumps(report, indent=2))
    if args.output:
        output_path = Path(args.output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nSaved report to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
