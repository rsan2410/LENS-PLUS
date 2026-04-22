import os


import sys
sys.path.insert(0,"")

import cv2
import ultralytics
from ultralytics import YOLO
from IPython.display import Image 


ultralytics.checks()


class Segmentation:
    def __init__(self, video_path, model_path, frames_output_dir, segmented_imgs_output_dir):
        self.video_path = video_path
        self.model_path = model_path
        self.frames_output_dir = frames_output_dir
        self.segmented_imgs_output_dir = segmented_imgs_output_dir

    def split_video(self):
        video = cv2.VideoCapture(self.video_path)
        
        if not video.isOpened():
            print("ERROR: Video could not be opened")
            exit()
        
        count = 0

        while True:
            isSuccess, image = video.read()
            if not isSuccess:
                break

            cv2.imwrite(os.path.join(self.frames_output_dir, f"frame{count}.jpg"), image)
            print(f"Frame {count} read - saved as - frame{count}.jpg")
            count += 1

        video.release()

    def segment_frames(self):
        model = YOLO(self.model_path)
        print("Pre-trained classes", model.names.values())
        
        count = 1
        if len(os.listdir(self.frames_output_dir)) != 0:
            for image_name in os.listdir(self.frames_output_dir):
                full_path = os.path.join(self.frames_output_dir, image_name)
                results = model(full_path)
                results[0].save(f"{self.segmented_imgs_output_dir}/output{count}.jpg")
                count += 1
        else:
            print("ERROR: directory is empty")


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    vid_path = os.path.join(BASE_DIR, "..", "media", "cat.mp4")
    frames_output_dir = os.path.join(BASE_DIR, "frames")
    segmented_imgs_output_dir = os.path.join(BASE_DIR, "segmentation_output")

    model_path = "yolov8n-seg.pt"


    model = Segmentation(vid_path, model_path, frames_output_dir, segmented_imgs_output_dir)
    model.split_video()
    model.segment_frames()