import os
import sys
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DEEPLAB_PATH = os.path.join(BASE_DIR, "DeepLabV3Plus-Pytorch")
sys.path.insert(0, DEEPLAB_PATH)

import cv2
import numpy as np
import torch
from torchvision import transforms
from ultralytics import YOLO

import network


# ==========================================================
# CONFIG
# ==========================================================

OUTPUT_DIR = r"D:\Zayaan\D_git\LENS-PLUS\models\segmentation\output"

OUTPUT_WIDTH = 640
OUTPUT_HEIGHT = 360

MERGE_IDLE_SECONDS = 40


# ==========================================================
# CITYSCAPES ACCESSIBILITY MAPPER
# ==========================================================

class CityscapesAccessibilityMapper:
    def __init__(self):
        self.class_names = [
            "road",
            "sidewalk",
            "building",
            "wall",
            "fence",
            "pole",
            "traffic light",
            "traffic sign",
            "vegetation",
            "terrain",
            "sky",
            "person",
            "rider",
            "car",
            "truck",
            "bus",
            "train",
            "motorcycle",
            "bicycle",
        ]

        self.walkable_classes = [1, 9]  # sidewalk, terrain
        self.hazard_classes = [0]       # road
        self.dynamic_obstacle_classes = [
            11, 12, 13, 14, 15, 16, 17, 18
        ]

    def get_walkable_mask(self, preds):
        mask = np.zeros_like(preds, dtype=np.uint8)
        for c in self.walkable_classes:
            mask[preds == c] = 1
        return mask

    def get_hazard_mask(self, preds):
        mask = np.zeros_like(preds, dtype=np.uint8)
        for c in self.hazard_classes:
            mask[preds == c] = 1
        return mask

    def get_dynamic_obstacle_mask(self, preds):
        mask = np.zeros_like(preds, dtype=np.uint8)
        for c in self.dynamic_obstacle_classes:
            mask[preds == c] = 1
        return mask

    def get_traffic_sign_mask(self, preds):
        return (preds == 7).astype(np.uint8)


# ==========================================================
# SEGMENTATION + NAVIGATION
# ==========================================================

class ImprovedSegmentation:
    def __init__(
        self,
        frames_root: str,
        yolo_model_path: str,
        deeplab_model_path: str,
        target_fps: int = 10,
        use_yolo: bool = True,
        deeplab_every_n_frames: int = 2,
    ):
        self.frames_root = Path(frames_root)
        self.target_fps = target_fps
        self.use_yolo = use_yolo
        self.deeplab_every_n_frames = deeplab_every_n_frames

        self.mapper = CityscapesAccessibilityMapper()

        self.prev_walkable_mask = None
        self.prev_hazard_mask = None

        self.yolo_model = YOLO(yolo_model_path) if use_yolo else None
        self.deeplab_model = self.load_deeplab(deeplab_model_path)

        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    # ------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------

    def natural_key(self, path):
        return [
            int(t) if t.isdigit() else t.lower()
            for t in re.split(r"(\d+)", path.name)
        ]

    def find_latest_artifact(self):
        artifacts = [
            p for p in self.frames_root.iterdir()
            if p.is_dir()
        ]

        if not artifacts:
            raise FileNotFoundError("No artifacts found")

        artifacts.sort(
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )

        return artifacts[0]

    def get_group_folders(self, artifact_dir):
        groups = [
            p for p in artifact_dir.iterdir()
            if p.is_dir() and p.name.startswith("group-")
        ]

        groups.sort(key=self.natural_key)
        return groups

    def load_frame_paths(self, folder):
        exts = {".jpg", ".jpeg", ".png", ".bmp"}

        frames = [
            p for p in folder.iterdir()
            if p.suffix.lower() in exts
        ]

        frames.sort(key=self.natural_key)
        return frames

    # ------------------------------------------------------
    # MODEL LOADING
    # ------------------------------------------------------

    def load_deeplab(self, path):
        model = network.modeling.__dict__["deeplabv3plus_mobilenet"](
            num_classes=19,
            output_stride=16,
        )

        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            model.load_state_dict(checkpoint["model_state"])
        else:
            model.load_state_dict(checkpoint)

        model.eval()
        return model

    # ------------------------------------------------------
    # FPS FROM FRAME TIMESTAMPS
    # ------------------------------------------------------

    def infer_real_fps(self, frame_paths):
        if len(frame_paths) < 2:
            return self.target_fps

        def parse_timestamp(path):
            stamp = path.stem.split("-")[-1]
            return datetime.strptime(
                stamp,
                "%Y%m%dT%H%M%S%fZ"
            )

        start = parse_timestamp(frame_paths[0])
        end = parse_timestamp(frame_paths[-1])

        seconds = (end - start).total_seconds()

        if seconds <= 0:
            return self.target_fps

        fps = len(frame_paths) / seconds

        return max(1, round(fps))

    # ------------------------------------------------------
    # SEGMENTATION
    # ------------------------------------------------------

    def get_semantic_predictions(self, image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = self.transform(rgb).unsqueeze(0)

        with torch.no_grad():
            outputs = self.deeplab_model(tensor)

        preds = outputs.max(1)[1].cpu().numpy()[0]

        preds = cv2.resize(
            preds.astype(np.uint8),
            (OUTPUT_WIDTH, OUTPUT_HEIGHT),
            interpolation=cv2.INTER_NEAREST,
        )

        return preds

    # ------------------------------------------------------
    # TEMPORAL SMOOTHING
    # ------------------------------------------------------

    def apply_temporal_smoothing(
        self,
        current_mask,
        previous_mask
    ):
        if previous_mask is None:
            return current_mask.astype(np.uint8)

        if current_mask.shape != previous_mask.shape:
            previous_mask = cv2.resize(
                previous_mask.astype(np.uint8),
                (
                    current_mask.shape[1],
                    current_mask.shape[0],
                ),
                interpolation=cv2.INTER_NEAREST,
            )

        smoothed = (
            0.7 * current_mask.astype(np.float32)
            + 0.3 * previous_mask.astype(np.float32)
        )

        return (smoothed > 0.5).astype(np.uint8)

    # ------------------------------------------------------
    # NAVIGATION LOGIC
    # ------------------------------------------------------

    def analyze_navigation(
        self,
        walkable,
        hazard,
        dynamic
    ):
        h, w = walkable.shape

        # use lower half (closer to user)
        walkable = walkable[h // 2:, :]
        hazard = hazard[h // 2:, :]
        dynamic = dynamic[h // 2:, :]

        third = w // 3

        zones = {
            "LEFT": slice(0, third),
            "CENTER": slice(third, third * 2),
            "RIGHT": slice(third * 2, w),
        }

        scores = {}

        for name, zone in zones.items():
            walk = np.sum(walkable[:, zone])
            haz = np.sum(hazard[:, zone])
            obs = np.sum(dynamic[:, zone])

            score = walk - (haz * 2) - (obs * 1.5)
            scores[name] = score

        best_zone = max(scores, key=scores.get)

        center_good = (
            scores["CENTER"]
            >= max(scores["LEFT"], scores["RIGHT"]) * 0.9
        )

        if center_good:
            direction = "FORWARD"
        elif best_zone == "LEFT":
            direction = "MOVE LEFT"
        else:
            direction = "MOVE RIGHT"

        total_pixels = walkable.size

        hazard_ratio = np.sum(hazard) / total_pixels

        if hazard_ratio > 0.35:
            status = "UNWALKABLE"
        else:
            status = "WALKABLE"

        return {
            "status": status,
            "direction": direction,
            "scores": scores,
        }

    def draw_navigation_arrow(
        self,
        frame,
        direction
    ):
        h, w = frame.shape[:2]

        start = (w // 2, h - 40)

        if direction == "FORWARD":
            end = (w // 2, h - 140)

        elif direction == "MOVE LEFT":
            end = (w // 2 - 140, h - 110)

        else:
            end = (w // 2 + 140, h - 110)

        cv2.arrowedLine(
            frame,
            start,
            end,
            (0, 255, 0),
            6,
            tipLength=0.25,
        )

    # ------------------------------------------------------
    # VISUALIZATION
    # ------------------------------------------------------

    def create_visualization(
        self,
        image,
        yolo_results,
        walkable,
        hazard,
        dynamic,
        signs,
    ):
        if yolo_results is not None:
            overlay = yolo_results[0].plot()
            overlay = cv2.resize(
                overlay,
                (OUTPUT_WIDTH, OUTPUT_HEIGHT),
            )
        else:
            overlay = image.copy()

        overlay = overlay.astype(np.float32)

        green = np.zeros_like(overlay)
        green[:] = [0, 255, 0]

        red = np.zeros_like(overlay)
        red[:] = [0, 0, 255]

        yellow = np.zeros_like(overlay)
        yellow[:] = [0, 255, 255]

        cyan = np.zeros_like(overlay)
        cyan[:] = [255, 255, 0]

        def blend(base, mask, color, alpha):
            mask3 = np.stack([mask, mask, mask], axis=2)

            return np.where(
                mask3 > 0,
                base * (1 - alpha) + color * alpha,
                base,
            )

        overlay = blend(overlay, walkable, green, 0.45)
        overlay = blend(overlay, hazard, red, 0.55)
        overlay = blend(overlay, dynamic, yellow, 0.55)
        overlay = blend(overlay, signs, cyan, 0.50)

        overlay = overlay.astype(np.uint8)

        nav = self.analyze_navigation(
            walkable,
            hazard,
            dynamic,
        )

        self.draw_navigation_arrow(
            overlay,
            nav["direction"],
        )

        lines = [
            f"STATUS: {nav['status']}",
            f"DIRECTION: {nav['direction']}",
        ]

        y = 30

        for line in lines:
            cv2.putText(
                overlay,
                line,
                (15, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
            )
            y += 34

        return overlay

    # ------------------------------------------------------
    # GROUP PROCESSING
    # ------------------------------------------------------

    def process_group(
        self,
        frame_paths,
        output_path
    ):
        if not frame_paths:
            return

        self.prev_walkable_mask = None
        self.prev_hazard_mask = None

        fps = self.infer_real_fps(frame_paths)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        out = cv2.VideoWriter(
            str(output_path),
            fourcc,
            fps,
            (OUTPUT_WIDTH, OUTPUT_HEIGHT),
        )

        if not out.isOpened():
            raise RuntimeError(
                f"Could not open writer {output_path}"
            )

        processed_count = 0
        segmentation_cache = None

        for frame_path in frame_paths:

            frame = cv2.imread(str(frame_path))

            if frame is None:
                continue

            frame = cv2.resize(
                frame,
                (OUTPUT_WIDTH, OUTPUT_HEIGHT)
            )

            if self.use_yolo:
                yolo_results = self.yolo_model(
                    frame,
                    conf=0.5,
                    verbose=False,
                )
            else:
                yolo_results = None

            if processed_count % self.deeplab_every_n_frames == 0:
                semantic = self.get_semantic_predictions(frame)
                segmentation_cache = semantic
            else:
                semantic = segmentation_cache

            walkable = self.mapper.get_walkable_mask(semantic)
            hazard = self.mapper.get_hazard_mask(semantic)
            dynamic = self.mapper.get_dynamic_obstacle_mask(semantic)
            signs = self.mapper.get_traffic_sign_mask(semantic)

            walkable = self.apply_temporal_smoothing(
                walkable,
                self.prev_walkable_mask,
            )

            hazard = self.apply_temporal_smoothing(
                hazard,
                self.prev_hazard_mask,
            )

            viz = self.create_visualization(
                frame,
                yolo_results,
                walkable,
                hazard,
                dynamic,
                signs,
            )

            out.write(viz)

            self.prev_walkable_mask = walkable
            self.prev_hazard_mask = hazard

            processed_count += 1

        out.release()

    # ------------------------------------------------------
    # MERGE VIDEOS
    # ------------------------------------------------------

    def merge_group_videos(
        self,
        artifact_name
    ):
        output_dir = Path(OUTPUT_DIR)

        mp4s = list(
            output_dir.glob(
                f"{artifact_name}_group-*.mp4"
            )
        )

        if not mp4s:
            return

        mp4s.sort(key=self.natural_key)

        final_path = output_dir / (
            f"{artifact_name}_FINAL.mp4"
        )

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        out = cv2.VideoWriter(
            str(final_path),
            fourcc,
            self.target_fps,
            (OUTPUT_WIDTH, OUTPUT_HEIGHT),
        )

        if not out.isOpened():
            raise RuntimeError(
                f"Cannot create {final_path}"
            )

        for video in mp4s:

            cap = cv2.VideoCapture(str(video))

            while True:
                ret, frame = cap.read()

                if not ret:
                    break

                if frame is None:
                    continue

                frame = cv2.resize(
                    frame,
                    (
                        OUTPUT_WIDTH,
                        OUTPUT_HEIGHT,
                    )
                )

                out.write(frame)

            cap.release()

        out.release()

        print("Merged:", final_path)

    # ------------------------------------------------------
    # MAIN LOOP
    # ------------------------------------------------------

    def run(self):
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        processed_groups = set()

        last_new_group_time = time.time()
        last_merged_count = 0

        while True:

            artifact = self.find_latest_artifact()
            groups = self.get_group_folders(artifact)

            new_group_found = False

            for group in groups:

                key = f"{artifact.name}_{group.name}"

                if key in processed_groups:
                    continue

                frame_paths = self.load_frame_paths(group)

                if not frame_paths:
                    continue

                print("Processing:", group)

                output_path = Path(OUTPUT_DIR) / (
                    f"{key}.mp4"
                )

                try:
                    self.process_group(
                        frame_paths,
                        output_path,
                    )

                    processed_groups.add(key)
                    new_group_found = True

                except Exception as error:
                    print(
                        "Group failed:",
                        group,
                        error,
                    )

            if new_group_found:
                last_new_group_time = time.time()

            idle = time.time() - last_new_group_time

            current_count = len(processed_groups)

            if (
                idle >= MERGE_IDLE_SECONDS
                and current_count > 0
                and current_count != last_merged_count
            ):
                self.merge_group_videos(
                    artifact.name
                )

                last_merged_count = current_count

                print("Session complete.")
                break

            time.sleep(2)


# ==========================================================
# ENTRY
# ==========================================================

if __name__ == "__main__":
    model = ImprovedSegmentation(
        frames_root=os.path.abspath(
            os.path.join(
                BASE_DIR,
                "..",
                "..",
                "..",
                "api",
                "app",
                "session_artifacts",
            )
        ),
        yolo_model_path="yolov8n-seg.pt",
        deeplab_model_path=os.path.join(
            BASE_DIR,
            "deeplabv3plus-mobilenet.pth",
        ),
        target_fps=5,
        use_yolo=True,
        deeplab_every_n_frames=2,
    )

    model.run()