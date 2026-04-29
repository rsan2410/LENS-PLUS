#!/usr/bin/env python3
"""
Minimal real-time object detection overlay for the existing LENS-PLUS live feed.

How to run:
1) Start the project services:
   docker compose up --build
2) In the web UI, start a source and click Connect so the backend receives frames.
3) Run detection:
   python run_live_detection.py

Optional:
  python run_live_detection.py --session-id <session_id>
  python run_live_detection.py --api-base http://localhost:8000
  python run_live_detection.py --source path/to/video.mp4
  python run_live_detection.py --model yolo11m.pt --imgsz 1024 --conf 0.2

If dependencies are missing:
  pip install ultralytics opencv-python numpy requests
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

try:
    import cv2
    import numpy as np
    import requests
    import urllib3
    from ultralytics import YOLO
except ImportError as exc:
    print(f"[ERROR] Missing dependency: {exc}")
    print("Install with: pip install ultralytics opencv-python numpy requests")
    sys.exit(1)


urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live object detection for LENS-PLUS feed")
    parser.add_argument(
        "--source",
        default=None,
        help="Optional override source (video file, camera index, RTSP/HTTP URL, or latest.jpg URL).",
    )
    parser.add_argument(
        "--api-base",
        default=None,
        help="Optional API base for auto feed discovery, e.g. http://localhost:8000 or https://localhost:5173/api",
    )
    parser.add_argument(
        "--session-id",
        default=None,
        help="Use a specific session id from /debug/sessions.",
    )
    parser.add_argument("--model", default="yolo11s.pt", help="Ultralytics model path/name.")
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=960,
        help="Inference image size (default: 960).",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=0.03,
        help="Poll interval for snapshot feed in seconds (default: 0.03).",
    )
    parser.add_argument(
        "--no-wait-for-feed",
        action="store_true",
        help="Do not keep retrying when no active live feed is available at startup.",
    )
    parser.add_argument(
        "--feed-retry-interval",
        type=float,
        default=1.0,
        help="Retry interval in seconds while waiting for live feed discovery (default: 1.0).",
    )
    return parser.parse_args()


def is_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https", "rtsp", "rtmp"} and bool(parsed.netloc)


def is_windows_camera_index(value: str) -> bool:
    return value.isdigit()


def parse_env_file(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    if not path.exists():
        return data
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key.strip()] = value.strip()
    return data


def parse_iso_timestamp(value: str | None) -> float:
    if not value:
        return 0.0
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def should_verify_tls(url: str) -> bool:
    # Dev setup here commonly uses self-signed mkcert certs.
    return not url.lower().startswith("https://")


def build_api_base_candidates(project_root: Path, override: str | None) -> list[str]:
    candidates: list[str] = []

    def add(value: str | None) -> None:
        if not value:
            return
        normalized = value.strip().rstrip("/")
        if normalized and normalized not in candidates:
            candidates.append(normalized)

    add(override)
    add(os.getenv("LENS_API_BASE"))

    env_values = parse_env_file(project_root / ".env")
    signaling_base = env_values.get("VITE_SIGNALING_BASE_URL")
    if signaling_base:
        if signaling_base.startswith("http://") or signaling_base.startswith("https://"):
            add(signaling_base)
        elif signaling_base.startswith("/"):
            add(f"https://localhost:5173{signaling_base}")
            add(f"http://localhost:5173{signaling_base}")

    add("http://localhost:8000")
    add("http://127.0.0.1:8000")
    add("https://localhost:5173/api")
    add("http://localhost:5173/api")

    return candidates


@dataclass
class SessionChoice:
    api_base: str
    session_id: str
    snapshot_url: str


def choose_session_id(sessions: list[dict[str, Any]], requested: str | None) -> str:
    if requested:
        for session in sessions:
            if session.get("session_id") == requested:
                return requested
        available = ", ".join(str(s.get("session_id")) for s in sessions[:10]) or "<none>"
        raise RuntimeError(
            f"Requested session_id '{requested}' not found. Available: {available}"
        )

    if not sessions:
        raise RuntimeError(
            "No active sessions found.\n"
            "Start the web app feed first (Start Source + Connect), then run this script."
        )

    ranked = sorted(
        sessions,
        key=lambda s: (
            bool(s.get("has_snapshot")),
            parse_iso_timestamp(s.get("latest_jpeg_at")),
            int(s.get("total_frames") or 0),
        ),
        reverse=True,
    )
    best = ranked[0].get("session_id")
    if not best:
        raise RuntimeError("Could not choose a valid session_id from /debug/sessions.")
    return str(best)


def discover_existing_live_feed(
    project_root: Path, api_base_override: str | None, session_id: str | None
) -> SessionChoice:
    api_main = project_root / "api" / "app" / "main.py"
    if not api_main.exists():
        raise RuntimeError(
            "Could not find api/app/main.py, so the expected live feed implementation is missing."
        )

    candidates = build_api_base_candidates(project_root, api_base_override)
    errors: list[str] = []

    for base in candidates:
        url = f"{base}/debug/sessions"
        try:
            response = requests.get(url, timeout=2.5, verify=should_verify_tls(url))
        except requests.RequestException as exc:
            errors.append(f"{url}: {exc}")
            continue

        if response.status_code != 200:
            errors.append(f"{url}: HTTP {response.status_code}")
            continue

        try:
            payload = response.json()
        except ValueError:
            errors.append(f"{url}: invalid JSON")
            continue

        sessions = payload.get("sessions")
        if not isinstance(sessions, list):
            errors.append(f"{url}: missing 'sessions' list")
            continue

        selected = choose_session_id(sessions, session_id)
        snapshot_url = f"{base}/debug/sessions/{selected}/latest.jpg"
        return SessionChoice(api_base=base, session_id=selected, snapshot_url=snapshot_url)

    detail = "\n".join(errors[-5:]) if errors else "No API candidate URLs were generated."
    raise RuntimeError(
        "Could not discover an active LENS-PLUS live feed.\n"
        f"Tried API bases: {candidates}\n"
        f"Last errors:\n{detail}"
    )


class FrameSource:
    source_label: str = ""
    last_error: str | None = None

    def read(self) -> tuple[bool, np.ndarray | None, bool]:
        raise NotImplementedError

    def close(self) -> None:
        return None


class OpenCvCaptureSource(FrameSource):
    def __init__(self, source: str | int, is_file: bool) -> None:
        self.source = source
        self.is_file = is_file
        self.source_label = f"opencv:{source}"
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open source with OpenCV: {source}")

    def read(self) -> tuple[bool, np.ndarray | None, bool]:
        ok, frame = self.cap.read()
        if not ok:
            if self.is_file:
                return False, None, True
            self.last_error = "No frame from stream."
            return False, None, False
        if frame is None or frame.size == 0:
            self.last_error = "Received empty frame."
            return False, None, False
        return True, frame, False

    def close(self) -> None:
        self.cap.release()


class SnapshotHttpSource(FrameSource):
    def __init__(self, snapshot_url: str, poll_interval: float = 0.03) -> None:
        self.snapshot_url = snapshot_url
        self.poll_interval = max(0.0, poll_interval)
        self.source_label = f"snapshot:{snapshot_url}"
        self.session = requests.Session()

    def read(self) -> tuple[bool, np.ndarray | None, bool]:
        params = {"t": str(int(time.time() * 1000))}
        verify = should_verify_tls(self.snapshot_url)
        try:
            response = self.session.get(
                self.snapshot_url, params=params, timeout=2.5, verify=verify
            )
        except requests.RequestException as exc:
            self.last_error = f"Request failed: {exc}"
            time.sleep(self.poll_interval)
            return False, None, False

        if response.status_code == 404:
            self.last_error = "Snapshot not ready yet (HTTP 404)."
            time.sleep(self.poll_interval)
            return False, None, False

        if response.status_code != 200:
            self.last_error = f"Snapshot endpoint HTTP {response.status_code}."
            time.sleep(self.poll_interval)
            return False, None, False

        image = cv2.imdecode(np.frombuffer(response.content, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            self.last_error = "Could not decode JPEG frame."
            time.sleep(self.poll_interval)
            return False, None, False

        return True, image, False

    def close(self) -> None:
        self.session.close()


def make_source(args: argparse.Namespace, project_root: Path) -> FrameSource:
    if args.source:
        source = args.source
        if is_windows_camera_index(source):
            return OpenCvCaptureSource(int(source), is_file=False)
        if Path(source).exists():
            return OpenCvCaptureSource(str(Path(source)), is_file=True)
        if is_url(source):
            if source.lower().endswith(".jpg"):
                return SnapshotHttpSource(source, poll_interval=args.poll_interval)
            return OpenCvCaptureSource(source, is_file=False)
        raise RuntimeError(
            f"Source '{source}' was not found as a file and is not a valid URL/camera index."
        )

    choice = discover_existing_live_feed(
        project_root=project_root,
        api_base_override=args.api_base,
        session_id=args.session_id,
    )
    print("[INFO] Detected existing live feed from this repo:")
    print(f"       API base: {choice.api_base}")
    print(f"       session_id: {choice.session_id}")
    print(f"       snapshot: {choice.snapshot_url}")
    return SnapshotHttpSource(choice.snapshot_url, poll_interval=args.poll_interval)


def open_source_with_retry(args: argparse.Namespace, project_root: Path) -> FrameSource:
    if args.source or args.no_wait_for_feed:
        return make_source(args, project_root)

    retry_interval = max(0.2, float(args.feed_retry_interval))
    print("[INFO] Waiting for live phone feed/session. Press Ctrl+C to cancel.")
    last_log_at = 0.0
    last_error: str | None = None

    while True:
        try:
            return make_source(args, project_root)
        except Exception as exc:
            message = str(exc).strip()
            now = time.monotonic()
            if message != last_error or (now - last_log_at) >= 2.0:
                print(f"[WAIT] {message}")
                last_log_at = now
                last_error = message
            time.sleep(retry_interval)


def draw_box_with_label(
    frame: np.ndarray, box: tuple[int, int, int, int], label: str, color: tuple[int, int, int]
) -> None:
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 1
    (text_w, text_h), baseline = cv2.getTextSize(label, font, scale, thickness)

    text_x = max(0, x1)
    text_y = y1 - 8
    if text_y < text_h + 4:
        text_y = min(frame.shape[0] - 4, y1 + text_h + 8)

    bg_tl = (text_x, max(0, text_y - text_h - 4))
    bg_br = (min(frame.shape[1] - 1, text_x + text_w + 6), min(frame.shape[0] - 1, text_y + baseline))
    cv2.rectangle(frame, bg_tl, bg_br, color, thickness=-1)
    cv2.putText(frame, label, (text_x + 3, text_y - 2), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)


def resolve_class_name(names: Any, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, list) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parent

    try:
        source = open_source_with_retry(args, project_root)
    except KeyboardInterrupt:
        print("\n[INFO] Waiting cancelled by user.")
        return 0
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 1

    print(f"[INFO] Opening source: {source.source_label}")
    print(f"[INFO] Loading model: {args.model}")
    try:
        model = YOLO(args.model)
    except Exception as exc:
        print(f"[ERROR] Could not load model '{args.model}': {exc}")
        source.close()
        return 1

    window_name = "LENS-PLUS Live Detection"
    last_frame_time = time.perf_counter()
    smoothed_fps = 0.0
    last_warning = 0.0

    print("[INFO] Press 'q' in the OpenCV window to quit.")

    try:
        while True:
            ok, frame, end_of_stream = source.read()

            if end_of_stream:
                print("[INFO] End of stream reached. Exiting.")
                break

            if not ok or frame is None:
                now = time.monotonic()
                if now - last_warning > 2.0 and source.last_error:
                    print(f"[WARN] {source.last_error}")
                    last_warning = now
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            try:
                result = model.predict(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
            except Exception as exc:
                now = time.monotonic()
                if now - last_warning > 2.0:
                    print(f"[WARN] Inference failed on frame: {exc}")
                    last_warning = now
                continue

            names = model.names
            boxes = result.boxes
            if boxes is not None:
                for det in boxes:
                    coords = det.xyxy[0].tolist()
                    x1, y1, x2, y2 = (int(v) for v in coords)
                    class_id = int(det.cls.item())
                    conf = float(det.conf.item())
                    label_name = resolve_class_name(names, class_id)
                    label = f"{label_name} {conf:.2f}"
                    draw_box_with_label(frame, (x1, y1, x2, y2), label, color=(0, 220, 90))

            current = time.perf_counter()
            dt = max(1e-6, current - last_frame_time)
            last_frame_time = current
            instant_fps = 1.0 / dt
            smoothed_fps = instant_fps if smoothed_fps == 0 else (0.9 * smoothed_fps + 0.1 * instant_fps)

            fps_text = f"FPS: {smoothed_fps:.1f}"
            cv2.rectangle(frame, (10, 10), (140, 38), (0, 0, 0), thickness=-1)
            cv2.putText(frame, fps_text, (16, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        source.close()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
