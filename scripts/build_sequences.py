import os
import argparse
import numpy as np
from glob import glob
from tqdm import tqdm


def load_keypoints(path):
    """Load npy and reshape to (T,66). Return None if invalid."""
    try:
        kps = np.load(path)

        # 空文件 / 损坏文件直接跳过
        if kps.size == 0 or kps.shape[0] == 0:
            print(f"[SKIP] Empty npy: {path}")
            return None

        # 形状必须是 (T,33,2)
        if len(kps.shape) != 3 or kps.shape[1:] != (33, 2):
            print(f"[SKIP] Bad shape {kps.shape} in {path}")
            return None

        # 检查 NaN
        if np.isnan(kps).any():
            print(f"[SKIP] Contains NaN: {path}")
            return None

        T = kps.shape[0]
        return kps.reshape(T, -1)

    except Exception as e:
        print(f"[SKIP] Load error {path}: {e}")
        return None


def make_sequences(split_path, seq_len, stride):
    """Return X, y for a data split."""
    X, y = [], []

    fight_dir = os.path.join(split_path, "Fight")
    nonfight_dir = os.path.join(split_path, "NonFight")

    # --- Fight ---
    for npy_file in tqdm(sorted(glob(os.path.join(fight_dir, "*.npy"))),
                         desc="Fight", leave=False):
        seq = load_keypoints(npy_file)
        if seq is None:
            continue

        T = seq.shape[0]
        if T < seq_len:
            continue

        for start in range(0, T - seq_len + 1, stride):
            X.append(seq[start:start + seq_len])
            y.append(1)

    # --- NonFight ---
    for npy_file in tqdm(sorted(glob(os.path.join(nonfight_dir, "*.npy"))),
                         desc="NonFight", leave=False):
        seq = load_keypoints(npy_file)
        if seq is None:
            continue

        T = seq.shape[0]
        if T < seq_len:
            continue

        for start in range(0, T - seq_len + 1, stride):
            X.append(seq[start:start + seq_len])
            y.append(0)

    X = np.array(X)
    y = np.array(y)
    return X, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keypoints_path", default="../data/datasets/keypoints")
    parser.add_argument("--save_path", default="../data/datasets/sequences")
    parser.add_argument("--seq_len", type=int, default=15)
    parser.add_argument("--stride", type=int, default=3)
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)

    print("=== Processing TRAIN ===")
    X_train, y_train = make_sequences(
        os.path.join(args.keypoints_path, "train"),
        args.seq_len,
        args.stride
    )
    print(f"[OK] Train: X = {X_train.shape}, y = {y_train.shape}")

    print("\n=== Processing VAL ===")
    X_val, y_val = make_sequences(
        os.path.join(args.keypoints_path, "val"),
        args.seq_len,
        args.stride
    )
    print(f"[OK] Val: X = {X_val.shape}, y = {y_val.shape}")

    np.save(os.path.join(args.save_path, "X_train.npy"), X_train)
    np.save(os.path.join(args.save_path, "y_train.npy"), y_train)
    np.save(os.path.join(args.save_path, "X_val.npy"), X_val)
    np.save(os.path.join(args.save_path, "y_val.npy"), y_val)

    print("\n=== DONE ===")
    print(f"Saved sequences to: {args.save_path}")


if __name__ == "__main__":
    main()
