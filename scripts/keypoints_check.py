import numpy as np
import matplotlib.pyplot as plt

npy_path = r"../data/datasets/keypoints/train/Fight/1dsLuL5Lvbc_1.npy"

arr = np.load(npy_path)  # (T, 70)
frame_id = 0  # 想看的帧

feat = arr[frame_id]
kps = feat[:66].reshape(33, 2)

plt.scatter(kps[:,0], kps[:,1])
plt.gca().invert_yaxis()
plt.title(f"Frame {frame_id} keypoints")
plt.show()
