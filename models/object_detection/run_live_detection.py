import os
import sys
import re
import json
import time
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from ultralytics import YOLO

BASE_DIR     = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[1]
APP_DIR      = PROJECT_ROOT / "api" / "app"

OUTPUT_DIR   = BASE_DIR / "output"
OUTPUT_WIDTH  = 640
OUTPUT_HEIGHT = 360

DEMO_BATCH_SIZE = 3

def natural_key(path: Path) -> list:
    return [
        int(t) if t.isdigit() else t.lower()
        for t in re.split(r"(\d+)", path.name)
    ]


def read_and_resize(path: Path) -> np.ndarray | None:
    frame = cv2.imread(str(path))
    if frame is not None:
        return cv2.resize(frame, (OUTPUT_WIDTH, OUTPUT_HEIGHT))
    return None

class ObjectDetector:
    def __init__(
        self,
        frames_root: str,
        model_path: str = "yolov8n.pt",
        conf: float = 0.5,
        target_fps: int = 5,
    ):
        self.frames_root = Path(frames_root)
        self.target_fps  = target_fps
        self.conf        = conf

        print(f"Loading YOLO model: {model_path}")
        self.model = YOLO(model_path)
        print("Model loaded.")

    def find_latest_artifact(self) -> Path:
        artifacts = [p for p in self.frames_root.iterdir() if p.is_dir()]
        if not artifacts:
            raise FileNotFoundError("No artifacts found")
        artifacts.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return artifacts[0]

    def get_group_folders(self, artifact_dir: Path) -> list[Path]:
        groups = [
            p for p in artifact_dir.iterdir()
            if p.is_dir() and p.name.startswith("group-")
        ]
        groups.sort(key=natural_key)
        return groups

    def load_frame_paths(self, folder: Path) -> list[Path]:
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        frames = [p for p in folder.iterdir() if p.suffix.lower() in exts]
        frames.sort(key=natural_key)
        return frames

    def infer_real_fps(self, frame_paths: list[Path]) -> int:
        if len(frame_paths) < 2:
            return self.target_fps

        def parse_ts(path: Path) -> datetime:
            stamp = path.stem.split("-")[-1]
            return datetime.strptime(stamp, "%Y%m%dT%H%M%S%fZ")

        try:
            start   = parse_ts(frame_paths[0])
            end     = parse_ts(frame_paths[-1])
            seconds = (end - start).total_seconds()
            if seconds <= 0:
                return self.target_fps
            return max(1, round(len(frame_paths) / seconds))
        except Exception:
            return self.target_fps

    def process_group(self, frame_paths: list[Path], output_path: Path) -> dict:
        if not frame_paths:
            return {}

        print(f"  Pre-fetching {len(frame_paths)} frames...")
        t_io = time.perf_counter()
        with ThreadPoolExecutor(max_workers=8) as ex:
            frames = list(ex.map(read_and_resize, frame_paths))
        print(f"  Loaded in {(time.perf_counter() - t_io)*1000:.0f}ms")

        fps    = self.infer_real_fps(frame_paths)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out    = cv2.VideoWriter(str(output_path), fourcc, fps, (OUTPUT_WIDTH, OUTPUT_HEIGHT))

        if not out.isOpened():
            raise RuntimeError(f"Could not open writer: {output_path}")

        frame_summaries = []
        total_detections = 0

        for idx, (frame_path, frame) in enumerate(zip(frame_paths, frames)):
            if frame is None:
                continue

            frame = cv2.resize(frame, (OUTPUT_WIDTH, OUTPUT_HEIGHT))

            t0 = time.perf_counter()
            results = self.model(frame, conf=self.conf, verbose=False)
            latency_ms = (time.perf_counter() - t0) * 1000

            detections = []
            annotated  = results[0].plot()
            annotated  = cv2.resize(annotated, (OUTPUT_WIDTH, OUTPUT_HEIGHT))

            if results[0].boxes is not None:
                for box in results[0].boxes:
                    xyxy    = box.xyxy[0].tolist()
                    conf    = float(box.conf[0])
                    cls_id  = int(box.cls[0])
                    label   = self.model.names[cls_id]
                    detections.append({
                        "label":      label,
                        "confidence": round(conf, 3),
                        "xyxy":       [round(v, 1) for v in xyxy],
                    })

            sidecar = frame_path.with_suffix(".detections.json")
            sidecar.write_text(json.dumps({"detections": detections}, indent=2))

            out.write(annotated)
            total_detections += len(detections)

            labels_str = ", ".join(d["label"] for d in detections) if detections else "none"
            print(
                f"  [{frame_path.name}] {len(detections)} detection(s): "
                f"{labels_str}  ({latency_ms:.0f}ms)"
            )

            frame_summaries.append({
                "frame":              frame_path.name,
                "detections":         detections,
                "inference_latency_ms": round(latency_ms, 2),
            })

        out.release()

        return {
            "total_frames":      len(frame_summaries),
            "total_detections":  total_detections,
            "inferred_fps":      fps,
            "frame_summaries":   frame_summaries,
        }

    def merge_n_groups(self, artifact_name: str, group_keys: list[str], batch_num: int):
        mp4s = [OUTPUT_DIR / f"{k}.mp4" for k in group_keys
                if (OUTPUT_DIR / f"{k}.mp4").exists()]
        if not mp4s:
            return

        demo_path = OUTPUT_DIR / f"{artifact_name}_demo_{batch_num}.mp4"
        fourcc    = cv2.VideoWriter_fourcc(*"mp4v")
        out       = cv2.VideoWriter(str(demo_path), fourcc, self.target_fps, (OUTPUT_WIDTH, OUTPUT_HEIGHT))

        for mp4 in mp4s:
            cap = cv2.VideoCapture(str(mp4))
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (OUTPUT_WIDTH, OUTPUT_HEIGHT))
                out.write(frame)
            cap.release()

        out.release()
        print(f"  Detection demo: {demo_path.name}  ({len(mp4s)} groups)")

    def run(self):
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        processed_groups  = set()
        processed_order   = []
        last_new_group_time = time.time()
        demo_batch_num    = 1
        last_merged_count = 0

        while True:
            try:
                artifact = self.find_latest_artifact()
            except FileNotFoundError:
                print("Waiting for a session to start...")
                time.sleep(2)
                continue

            groups = self.get_group_folders(artifact)

            is_closed     = False
            manifest_path = artifact / "session.json"
            if manifest_path.exists():
                try:
                    manifest_data = json.loads(manifest_path.read_text())
                    if manifest_data.get("closed_at") is not None:
                        is_closed = True
                except Exception:
                    pass

            safe_groups = groups if is_closed else (groups[:-1] if len(groups) > 1 else [])

            for group in safe_groups:
                key = f"{artifact.name}_{group.name}"
                if key in processed_groups:
                    continue

                frame_paths = self.load_frame_paths(group)
                if not frame_paths:
                    continue

                print(f"[Detection] Processing: {group.name}")

                video_path = OUTPUT_DIR / f"{key}.mp4"
                json_path  = OUTPUT_DIR / f"{key}.json"

                try:
                    results = self.process_group(frame_paths, video_path)
                    results["group"]    = group.name
                    results["artifact"] = artifact.name
                    json_path.write_text(json.dumps(results, indent=2) + "\n")
                    print(f"  JSON: {json_path.name}")

                    processed_groups.add(key)
                    processed_order.append(key)

                except Exception as error:
                    print(f"  Group failed: {group.name} — {error}")

            unmerged = len(processed_order) - last_merged_count
            if unmerged >= DEMO_BATCH_SIZE:
                batch_keys = processed_order[last_merged_count : last_merged_count + DEMO_BATCH_SIZE]
                self.merge_n_groups(artifact.name, batch_keys, demo_batch_num)
                last_merged_count += DEMO_BATCH_SIZE
                demo_batch_num    += 1

            elif is_closed and unmerged > 0:
                print(f"  Flushing final {unmerged} group(s) into demo...")
                batch_keys = processed_order[last_merged_count:]
                self.merge_n_groups(artifact.name, batch_keys, demo_batch_num)
                last_merged_count += unmerged
                demo_batch_num    += 1

            time.sleep(2)


if __name__ == "__main__":
    detector = ObjectDetector(
        frames_root=str(APP_DIR / "session_artifacts"),
        model_path="yolov8n.pt",
        conf=0.3,
        target_fps=5,
    )
    detector.run()