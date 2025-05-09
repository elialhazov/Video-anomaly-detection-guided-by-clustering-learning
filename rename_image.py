import os

root_dir = 'dataset/ucsdpeds/vidf'

# עובר על כל התיקיות
for folder in sorted(os.listdir(root_dir)):
    folder_path = os.path.join(root_dir, folder)
    if os.path.isdir(folder_path):
        images = sorted([f for f in os.listdir(folder_path) if f.endswith('.png')])
        for i, img_name in enumerate(images):
            new_name = f"{i:03}.png"
            old_path = os.path.join(folder_path, img_name)
            new_path = os.path.join(folder_path, new_name)
            os.rename(old_path, new_path)
        print(f"✔ תיקיה '{folder}' עודכנה ({len(images)} תמונות)")
