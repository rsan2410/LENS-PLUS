import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

OBJ_DETECTION_SCRIPT = BASE_DIR / "object_detection" / "run_live_detection.py"
SEGMENTATION_SCRIPT = BASE_DIR / "segmentation" / "src" / "segmentation-live-feed.py"
DEPTH_SCRIPT = BASE_DIR / "depth_estimation" / "depth_estimator.py"

def main():
    print("Pipeline Starting ...")
    
    print(f"Starting Object Detection: {OBJ_DETECTION_SCRIPT.name}")
    obj_det_process = subprocess.Popen([sys.executable, str(OBJ_DETECTION_SCRIPT)])
    
    time.sleep(2)

    print(f"Starting Segmentation: {SEGMENTATION_SCRIPT.name}")
    seg_process = subprocess.Popen([sys.executable, str(SEGMENTATION_SCRIPT)])
    
    time.sleep(2) 
    
    print(f"Starting Depth Estimation: {DEPTH_SCRIPT.name}")
    depth_process = subprocess.Popen([sys.executable, str(DEPTH_SCRIPT)])

    print("\nPipeline is live. Press Ctrl+C to stop.\n")

    try:
        while True:
            time.sleep(1)
            if obj_det_process.poll() is not None:
                print("WARN: Object Detection script stopped unexpectedly!")
            if seg_process.poll() is not None:
                print("WARN: Segmentation script stopped unexpectedly!")
            if depth_process.poll() is not None:
                print("WARN: Depth script stopped unexpectedly!")
                
    except KeyboardInterrupt:
        print("Shutting down pipeline...")
        obj_det_process.terminate()
        seg_process.terminate()
        depth_process.terminate()
        
        obj_det_process.wait()
        seg_process.wait()
        depth_process.wait()
        print("Shutdown complete.")

if __name__ == "__main__":
    main()