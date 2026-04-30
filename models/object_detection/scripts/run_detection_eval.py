#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.evaluation.detection_metrics import (
    GroundTruthBox,
    PredictedBox,
    evaluate_detection_suite,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate object detection metrics (IoU/F1/mAP/per-class/confusion)."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to JSON file with predictions and ground_truths.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional output path for JSON metrics report.",
    )
    return parser.parse_args()


def _to_bbox(raw_bbox: list[Any]) -> tuple[float, float, float, float]:
    if len(raw_bbox) != 4:
        raise ValueError("bbox must have 4 numeric values.")
    return (
        float(raw_bbox[0]),
        float(raw_bbox[1]),
        float(raw_bbox[2]),
        float(raw_bbox[3]),
    )


def _load_payload(input_path: Path) -> tuple[list[PredictedBox], list[GroundTruthBox]]:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Input JSON must be an object.")

    raw_predictions = payload.get("predictions", [])
    raw_ground_truths = payload.get("ground_truths", [])
    if not isinstance(raw_predictions, list) or not isinstance(raw_ground_truths, list):
        raise ValueError("'predictions' and 'ground_truths' must be lists.")

    predictions: list[PredictedBox] = []
    ground_truths: list[GroundTruthBox] = []

    for raw in raw_predictions:
        if not isinstance(raw, dict):
            raise ValueError("Each prediction must be an object.")
        predictions.append(
            PredictedBox(
                image_id=str(raw["image_id"]),
                label=str(raw["label"]),
                bbox_xyxy=_to_bbox(raw["bbox"]),
                score=float(raw["score"]),
            )
        )

    for raw in raw_ground_truths:
        if not isinstance(raw, dict):
            raise ValueError("Each ground truth must be an object.")
        ground_truths.append(
            GroundTruthBox(
                image_id=str(raw["image_id"]),
                label=str(raw["label"]),
                bbox_xyxy=_to_bbox(raw["bbox"]),
            )
        )

    return predictions, ground_truths


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).resolve()
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    predictions, ground_truths = _load_payload(input_path)
    report = evaluate_detection_suite(predictions, ground_truths)

    print(json.dumps(report, indent=2))

    if args.output:
        output_path = Path(args.output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nSaved report to: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
