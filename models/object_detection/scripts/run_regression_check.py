#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regression gate on metrics reports.")
    parser.add_argument("--current", required=True, help="Current metrics JSON report.")
    parser.add_argument(
        "--baseline", default="", help="Optional baseline metrics JSON report."
    )
    parser.add_argument("--min-map50", type=float, default=0.0, help="Absolute min map@0.5")
    parser.add_argument(
        "--min-map5095",
        type=float,
        default=0.0,
        help="Absolute min map@0.5:0.95",
    )
    parser.add_argument(
        "--max-map50-drop",
        type=float,
        default=0.02,
        help="Allowed map@0.5 drop vs baseline.",
    )
    parser.add_argument(
        "--max-map5095-drop",
        type=float,
        default=0.02,
        help="Allowed map@0.5:0.95 drop vs baseline.",
    )
    return parser.parse_args()


def _load(path: str) -> dict:
    return json.loads(Path(path).resolve().read_text(encoding="utf-8"))


def _extract_metrics(payload: dict) -> tuple[float, float]:
    map50 = float(payload.get("map_0_5", 0.0))
    map5095 = float(payload.get("map_0_5_to_0_95", 0.0))
    return map50, map5095


def main() -> int:
    args = parse_args()
    current = _load(args.current)
    current_map50, current_map5095 = _extract_metrics(current)

    failures: list[str] = []
    if current_map50 < args.min_map50:
        failures.append(
            f"map@0.5 {current_map50:.4f} below minimum {args.min_map50:.4f}"
        )
    if current_map5095 < args.min_map5095:
        failures.append(
            f"map@0.5:0.95 {current_map5095:.4f} below minimum {args.min_map5095:.4f}"
        )

    if args.baseline:
        baseline = _load(args.baseline)
        baseline_map50, baseline_map5095 = _extract_metrics(baseline)

        if current_map50 < baseline_map50 - args.max_map50_drop:
            failures.append(
                "map@0.5 regression: "
                f"{current_map50:.4f} < {baseline_map50:.4f} - {args.max_map50_drop:.4f}"
            )
        if current_map5095 < baseline_map5095 - args.max_map5095_drop:
            failures.append(
                "map@0.5:0.95 regression: "
                f"{current_map5095:.4f} < {baseline_map5095:.4f} - {args.max_map5095_drop:.4f}"
            )

    if failures:
        print("REGRESSION CHECK: FAIL")
        for line in failures:
            print(f"- {line}")
        return 1

    print("REGRESSION CHECK: PASS")
    print(f"- map@0.5: {current_map50:.4f}")
    print(f"- map@0.5:0.95: {current_map5095:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
