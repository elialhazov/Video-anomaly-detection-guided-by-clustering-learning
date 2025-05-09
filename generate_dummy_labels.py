import os
import numpy as np

video_root = './dataset/ucsdpeds/vidf'
label_output = './dataset/ucsdpeds/labels'

os.makedirs(label_output, exist_ok=True)

for folder_name in sorted(os.listdir(video_root)):
    folder_path = os.path.join(video_root, folder_name)
    if not os.path.isdir(folder_path):
        continue

    frame_count = len([
        f for f in os.listdir(folder_path)
        if f.endswith('.png') or f.endswith('.jpg')
    ])

    label_array = np.zeros(frame_count, dtype=np.uint8)
    save_path = os.path.join(label_output, folder_name + '.npy')
    np.save(save_path, label_array)
    print(f'✓ Created label for {folder_name} with {frame_count} frames')
