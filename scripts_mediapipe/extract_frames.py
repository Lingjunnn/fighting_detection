import cv2
import os
import argparse
from glob import glob
from tqdm import tqdm

def extract_frames(video_path, out_dir, target_fps=5, min_frames=10):
    """
    Extract frames at a target FPS.
    Will NOT create an empty folder. Only creates folder after at least 1 frame is saved.
    """

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return False

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        print(f"[ERROR] Invalid FPS for video: {video_path}")
        return False

    # Determine frame sampling interval
    frame_interval = max(1, int(round(video_fps / target_fps)))

    frame_count = 0
    saved = 0
    folder_created = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Save frame
        if frame_count % frame_interval == 0:
            if not folder_created:
                os.makedirs(out_dir, exist_ok=True)
                folder_created = True

            save_path = os.path.join(out_dir, f"frame_{saved:05d}.jpg")
            cv2.imwrite(save_path, frame)
            saved += 1

        frame_count += 1

    cap.release()

    # If folder was never created → no frames extracted
    if not folder_created:
        return False

    # Delete folder if frames too few
    if saved < min_frames:
        for f in os.listdir(out_dir):
            os.remove(os.path.join(out_dir, f))
        os.rmdir(out_dir)
        print(f"[WARN] Too few frames ({saved}), removed: {out_dir}")
        return False

    return True



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rwf_path", type=str, default="../data/datasets/RWF2000",
                        help="Path to original RWF2000 videos")
    parser.add_argument("--save_path", type=str, default="../data/datasets/frames",
                        help="Where to save extracted frames")
    parser.add_argument("--fps", type=int, default=5, help="Target FPS to extract")
    parser.add_argument("--min_frames", type=int, default=10,
                        help="Minimum valid frames; otherwise delete folder")

    # NEW: independent limits for train and val
    parser.add_argument("--max_train", type=int, default=16,
                        help="Max videos per class for TRAIN (fights & noFights)")
    parser.add_argument("--max_val", type=int, default=40,
                        help="Max videos per class for VAL (fights & noFights)")

    args = parser.parse_args()

    splits = ["train", "val"]
    classes = ["fights", "noFights"]

    for split in splits:
        print(f"\n=== Processing {split} split ===")

        # Determine limit based on split
        if split == "train":
            limit = args.max_train
        else:
            limit = args.max_val

        for clazz in classes:
            print(f"\n--- Extracting {split}/{clazz} ---")

            in_dir = os.path.join(args.rwf_path, split, clazz)
            out_dir_split = os.path.join(args.save_path, split, clazz)
            os.makedirs(out_dir_split, exist_ok=True)

            videos = sorted(glob(os.path.join(in_dir, "*")))

            # Apply per-split limit
            if limit is not None:
                videos = videos[:limit]
                print(f"[INFO] Limiting {split}/{clazz} to {limit} videos")

            for video_path in tqdm(videos):
                if not os.path.isfile(video_path):
                    continue

                vid_name = os.path.splitext(os.path.basename(video_path))[0]
                out_dir = os.path.join(out_dir_split, vid_name)

                extract_frames(
                    video_path,
                    out_dir,
                    target_fps=args.fps,
                    min_frames=args.min_frames
                )


if __name__ == "__main__":
    main()
