#!/usr/bin/env python3
from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path
from urllib.parse import urlparse

try:
    import cv2
    import numpy as np
    import requests
    from ultralytics import YOLO
except ImportError as exc:
    raise SystemExit(
        "Missing dependency for latency benchmark. Install with: "
        "pip install ultralytics opencv-python numpy requests\n"
        f"Import error: {exc}"
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark detection latency/FPS.")
    parser.add_argument("--model", default="yolo11s.pt", help="Model path/name.")
    parser.add_argument(
        "--source",
        required=True,
        help="Video path, camera index, stream URL, or latest.jpg URL.",
    )
    parser.add_argument("--frames", type=int, default=120, help="Frames to measure.")
    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
        help="Warmup frames before measuring.",
    )
    parser.add_argument("--imgsz", type=int, default=960, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    return parser.parse_args()


def _is_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https", "rtsp", "rtmp"} and bool(parsed.netloc)


def _read_snapshot(session: requests.Session, url: str) -> np.ndarray | None:
    response = session.get(url, params={"t": str(int(time.time() * 1000))}, timeout=3.0)
    if response.status_code != 200:
        return None
    return cv2.imdecode(np.frombuffer(response.content, dtype=np.uint8), cv2.IMREAD_COLOR)


def _iter_frames(source: str):
    if source.lower().endswith(".jpg") and _is_url(source):
        with requests.Session() as session:
            while True:
                frame = _read_snapshot(session, source)
                if frame is None:
                    continue
                yield frame
        return

    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
    else:
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {source}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame is not None and frame.size > 0:
                yield frame
    finally:
        cap.release()


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(round((pct / 100.0) * (len(ordered) - 1)))
    return ordered[idx]


def main() -> int:
    args = parse_args()
    model = YOLO(args.model)
    frame_iter = _iter_frames(args.source)

    for _ in range(max(0, args.warmup)):
        try:
            frame = next(frame_iter)
        except StopIteration:
            raise SystemExit("Source ended before warmup completed.")
        model.predict(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)

    latencies_ms: list[float] = []
    started_at = time.perf_counter()
    for _ in range(max(1, args.frames)):
        try:
            frame = next(frame_iter)
        except StopIteration:
            break
        t0 = time.perf_counter()
        model.predict(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    elapsed = time.perf_counter() - started_at
    measured_frames = len(latencies_ms)
    if measured_frames == 0:
        raise SystemExit("No frames measured.")

    fps = measured_frames / elapsed if elapsed > 0 else 0.0
    print("Latency/FPS benchmark")
    print(f"- model: {args.model}")
    print(f"- frames_measured: {measured_frames}")
    print(f"- avg_ms: {statistics.mean(latencies_ms):.2f}")
    print(f"- p50_ms: {_percentile(latencies_ms, 50):.2f}")
    print(f"- p95_ms: {_percentile(latencies_ms, 95):.2f}")
    print(f"- p99_ms: {_percentile(latencies_ms, 99):.2f}")
    print(f"- fps: {fps:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
