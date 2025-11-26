import os
import json
import argparse
import numpy as np
from glob import glob
from tqdm import tqdm
import mediapipe as mp
import cv2

mp_pose = mp.solutions.pose.Pose(static_image_mode=True)


def extract_pose(frame, box):
    h, w, _ = frame.shape
    x1, y1, x2, y2 = map(int, box)
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w - 1, x2)
    y2 = min(h - 1, y2)

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((33, 2))

    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    result = mp_pose.process(crop_rgb)

    # 如果没检测到姿态，返回0
    if not result.pose_landmarks:
        return np.zeros((33, 2))

    keypoints = []
    for lm in result.pose_landmarks.landmark:
        px = lm.x * (x2 - x1)
        py = lm.y * (y2 - y1)
        keypoints.append([px, py])

    return np.array(keypoints)


def load_detection_json(json_path):
    """
    自动识别 YOLO detection JSON 的结构：
    - dict: { "frame_00001.jpg": {"boxes": [...]} }
    - list: [ {"frame": "...", "boxes": [...]} ]
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    # Case 1: dict 格式
    if isinstance(data, dict):
        frames = sorted(data.keys())
        return frames, data, "dict"

    # Case 2: list 格式
    if isinstance(data, list):
        frames = sorted([item["frame"] for item in data])
        # 方便通过 frame 查找对应 bbox
        data_dict = {item["frame"]: item for item in data}
        return frames, data_dict, "list"

    raise ValueError("Unknown detection JSON format!")


def process_video(frame_dir, detection_json, save_path):
    frames, data, fmt = load_detection_json(detection_json)

    keypoints_seq = []

    for fname in frames:

        # 判断 fname 的类型并构造正确的文件名
        if isinstance(fname, int):
            # 如果是数字，说明是 list 格式的 JSON
            # f"frame_{fname:05d}.jpg" 会把 1 变成 frame_00001.jpg
            # :05d 表示补齐5位数字，如果是6位请改为 :06d
            file_name = f"frame_{fname+1:05d}.jpg"
        else:
            # 如果已经是字符串 (dict 格式)，直接使用
            file_name = fname

        frame_path = os.path.join(frame_dir, file_name)

        if not os.path.exists(frame_path):
            # 调试打印：如果找不到文件，打印出来看看路径对不对
            # print(f"Missing: {frame_path}")
            keypoints_seq.append(np.zeros((33, 2)))
            continue

        frame = cv2.imread(frame_path)
        if frame is None:
            keypoints_seq.append(np.zeros((33, 2)))
            continue

        boxes = data[fname]["boxes"]

        # 选最大 bbox
        if len(boxes) > 0:
            areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
            max_idx = np.argmax(areas)
            box = boxes[max_idx]
            kps = extract_pose(frame, box)
        else:
            box = [0, 0, 0, 0]
            kps = np.zeros((33, 2))

        # flatten + concat bbox → (70,)
        feat = np.concatenate([kps.reshape(-1), np.array(box, dtype=np.float32)])

        keypoints_seq.append(feat)

    keypoints_seq = np.array(keypoints_seq)
    np.save(save_path, keypoints_seq)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames_path", type=str, default="../data/datasets/frames")
    parser.add_argument("--detection_path", type=str, default="../data/datasets/detections")
    parser.add_argument("--save_path", type=str, default="../data/datasets/keypoints")
    args = parser.parse_args()

    splits = ["train", "val"]
    classes = ["Fight", "NonFight"]

    for split in splits:
        for clazz in classes:
            print(f"\n=== Extracting keypoints: {split}/{clazz} ===")

            det_files = sorted(glob(os.path.join(args.detection_path, split, clazz, "*.json")))
            print(f"[INFO] Found {len(det_files)} detection JSON files.")

            for det_file in tqdm(det_files):
                vid_name = os.path.basename(det_file).replace(".json", "")
                frame_dir = os.path.join(args.frames_path, split, clazz, vid_name)

                if not os.path.exists(frame_dir):
                    print(f"[WARN] Missing frames: {frame_dir}")
                    continue

                save_path = os.path.join(args.save_path, split, clazz, vid_name + ".npy")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)

                process_video(frame_dir, det_file, save_path)


if __name__ == "__main__":
    main()
