import os
import cv2
import json
from glob import glob
from ultralytics import YOLO
import argparse


def process_video_frames(frame_dir, save_path, model):
    """
    对某个视频的所有帧做人体检测，输出为 JSON。
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    frame_paths = sorted(glob(os.path.join(frame_dir, "*.jpg")))
    results_json = []

    for idx, frame_path in enumerate(frame_paths):
        img = cv2.imread(frame_path)
        if img is None:
            print(f"[WARN] Cannot read image: {frame_path}")
            continue

        res = model(img, verbose=False)[0]

        boxes = []
        for b in res.boxes:
            if int(b.cls) == 0:  # 0 = person
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                boxes.append([float(x1), float(y1), float(x2), float(y2)])

        results_json.append({"frame": idx, "boxes": boxes})

    with open(save_path, "w") as f:
        json.dump(results_json, f)

    print(f"[OK] YOLO processed: {frame_dir} → {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames_path", type=str, default="../data/datasets/frames",
                        help="Path to extracted frames directory")
    parser.add_argument("--save_path", type=str, default="../data/datasets/detections",
                        help="Path to save YOLO detection JSON files")
    parser.add_argument("--model", type=str, default="../models/yolov8s.pt",
                        help="YOLO model to use")
    args = parser.parse_args()

    model = YOLO(args.model)

    splits = ["train", "val"]
    classes = ["fights", "noFights"]

    for split in splits:
        for clazz in classes:
            video_dirs = glob(os.path.join(args.frames_path, split, clazz, "*"))

            for v in video_dirs:
                vid_name = os.path.basename(v)
                save_file = os.path.join(args.save_path, split, clazz, vid_name + ".json")

                process_video_frames(v, save_file, model)


if __name__ == "__main__":
    main()
