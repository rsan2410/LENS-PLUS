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

from app.evaluation.segmentation_metrics import (
    SegmentationSample,
    evaluate_segmentation_suite,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate segmentation metrics (Dice, Pixel Accuracy, mIoU)."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to JSON file with segmentation samples.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional output path for JSON metrics report.",
    )
    return parser.parse_args()


def _load_payload(input_path: Path) -> list[SegmentationSample]:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Input JSON must be an object.")

    raw_samples = payload.get("samples", [])
    if not isinstance(raw_samples, list):
        raise ValueError("'samples' must be a list.")

    samples: list[SegmentationSample] = []
    for raw in raw_samples:
        if not isinstance(raw, dict):
            raise ValueError("Each sample must be an object.")
        pred_mask = raw.get("pred_mask")
        gt_mask = raw.get("gt_mask")
        if not isinstance(pred_mask, list) or not isinstance(gt_mask, list):
            raise ValueError("pred_mask and gt_mask must be 2D lists.")

        samples.append(
            SegmentationSample(
                image_id=str(raw.get("image_id", "")),
                pred_mask=[[int(value) for value in row] for row in pred_mask],
                gt_mask=[[int(value) for value in row] for row in gt_mask],
            )
        )
    return samples


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).resolve()
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    samples = _load_payload(input_path)
    report = evaluate_segmentation_suite(samples)
    print(json.dumps(report, indent=2))

    if args.output:
        output_path = Path(args.output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nSaved report to: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
