import cv2
import numpy as np
import os


def process_and_save_frames(source_folder, output_folder, target_length=20):
    # 如果输出目录不存在，自动创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取文件夹内所有视频文件 (这里假设是 .avi 或 .mp4)
    video_files = [f for f in os.listdir(source_folder) if f.endswith(('.avi', '.mp4', '.mpg'))]
    print(f"发现 {len(video_files)} 个视频，开始处理...")

    for video_file in video_files:
        video_path = os.path.join(source_folder, video_file)
        # 获取文件名（不带后缀），用于给图片命名
        video_name = os.path.splitext(video_file)[0]

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames > 0:
            # 核心逻辑：生成均匀分布的20个帧索引
            # 例如 25帧取20帧 -> [0, 1, 2, ..., 24] 中均匀取值
            indices = np.linspace(0, total_frames - 1, target_length, dtype=int)

            for i, frame_idx in enumerate(indices):
                # 定位到指定帧
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()

                if ret:
                    # 命名格式：视频名_帧序号.jpg (例如 fight01_00.jpg, fight01_01.jpg ...)
                    # {:02d} 保证序号是两位数 (00-19)
                    save_name = f"{video_name}_{i:02d}.jpg"
                    save_path = os.path.join(output_folder, save_name)
                    cv2.imwrite(save_path, frame)

        cap.release()

    print("所有帧提取并保存完成！")


# --- 使用示例 ---
# 请修改为你的实际路径
source_dir = "../data/datasets/movieFights/noFights"  # 存放视频的文件夹
output_dir = "../data/datasets/movieFights/frames/noFights"  # 存放结果图片的文件夹

process_and_save_frames(source_dir, output_dir, target_length=20)