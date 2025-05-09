import glob
import os
import random
import cv2
import matplotlib
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import torch
import torchvision
import logging
import math
import scipy.io as scio
from einops import rearrange
from PIL import Image
from skimage import io, transform, color
from torch import nn

def save_tensor_video(x, output_dir="video_show", save_name=None):
    _, _, c, _, _ = x.size()
    x = rearrange(x, 'B C D H W -> B D C H W')
    os.makedirs(output_dir, exist_ok=True)
    for i, video in enumerate(x):
        video_dir = os.path.join(output_dir, str(i))
        os.makedirs(video_dir, exist_ok=True)
        for j, image in enumerate(video):
            if save_name is None:
                torchvision.utils.save_image(image, os.path.join(video_dir, f"img{j}.jpg"))
            else:
                torchvision.utils.save_image(image, os.path.join(video_dir, save_name))
        print(f"הסרטון {i} נשמר בהצלחה")

def load_pretrain_model(ckp_path, model):
    if not os.path.isfile(ckp_path):
        return
    print(f"Found checkpoint at {ckp_path}")
    checkpoint = torch.load(ckp_path, map_location='cuda:0')
    model_dict = model.state_dict()
    for key, value in checkpoint.items():
        key = key[7:]
        if key in model_dict and value is not None:
            try:
                model_dict[key] = value
                print(f"=> loaded '{key}' from checkpoint '{ckp_path}'")
            except Exception as e:
                print(f"=> failed to load '{key}': {e}")
        else:
            print(f"=> key '{key}' not found in checkpoint: '{ckp_path}'")
    model.load_state_dict(model_dict)
    print("המודל נטען בהצלחה")

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter("[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s")
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

def load_model(ckp_path, model):
    if not os.path.isfile(ckp_path):
        return
    print(f"Found checkpoint at {ckp_path}")
    checkpoint = torch.load(ckp_path)
    checkpoint_dict = checkpoint['state_dict']
    model_dict = model.state_dict()
    for key, value in checkpoint_dict.items():
        if key in model_dict and value is not None:
            try:
                model_dict[key] = value
                print(f"=> loaded '{key}' from checkpoint '{ckp_path}'")
            except Exception as e:
                print(f"=> failed to load '{key}': {e}")
        else:
            print(f"=> key '{key}' not found in checkpoint: '{ckp_path}'")
    model.load_state_dict(model_dict)
    print("המודל נטען בהצלחה")

def psnr(mse):
    return [10 * math.log10(1.0 / mse_item) for mse_item in mse]

def anomly_score(psnr):
    max_psnr = max(psnr)
    min_psnr = min(psnr)
    return [1.0 - (x - min_psnr) / (max_psnr - min_psnr) for x in psnr]

def fix_random_seeds(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def image_tensor2cv2(input_tensor: torch.Tensor):
    assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
    input_tensor = input_tensor.clone().detach().cpu().squeeze()
    input_tensor = input_tensor.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
    return cv2.cvtColor(input_tensor, cv2.COLOR_RGB2BGR)

def Error_thermogram_visualize(recon_image, origin_image, recon_file_name, origin_file_name, file_name):
    recon_image = image_tensor2cv2(recon_image)
    origin_image = image_tensor2cv2(origin_image)
    cv2.imwrite(recon_file_name, recon_image)
    cv2.imwrite(origin_file_name, origin_image)
    recon_image = color.rgb2gray(recon_image)
    origin_image = color.rgb2gray(origin_image)
    d = (abs(origin_image - recon_image) ** 2) * 10
    fig = plt.figure(dpi=200)
    plt.imshow(d, norm=matplotlib.colors.Normalize(vmin=0, vmax=1), cmap="jet")
    plt.axis("off")
    plt.savefig(file_name, bbox_inches='tight', dpi=400, pad_inches=0)
    plt.close()

def freeze_bn(m):
    if isinstance(m, nn.BatchNorm3d):
        m.eval()

def mat2npy(save_dir):
    for mat_path in glob.glob(os.path.join(save_dir, '*.mat')):
        index = mat_path.split('.')[0][-2:]
        data = scio.loadmat(mat_path)['frame_label']
        np.save(os.path.join(save_dir, f'{index}.npy'), data)
    print('תגיות נשמרו')

def Avenue_Ped2_test_dataset_format(folder_path):
    for path in glob.glob(os.path.join(folder_path, '*')):
        index = int(os.path.split(path)[-1])
        newname = os.path.join(os.path.split(path)[0], f'01_{index:04d}')
        os.rename(path, newname)
        print(f"{path} ======> {newname}")

def Avenue_Ped2_test_label_format(folder_path):
    for path in glob.glob(os.path.join(folder_path, '*.npy')):
        index = int(os.path.split(path)[-1].split(".")[0])
        newname = os.path.join(os.path.split(path)[0], f'01_{index:04d}.npy')
        os.rename(path, newname)
        print(f"{path} ======> {newname}")

def plot_embedding(data, label, title):
    data = (data - np.min(data, 0)) / (np.max(data, 0) - np.min(data, 0))
    colors = plt.get_cmap('viridis', 5)
    fig = plt.figure()
    for i in range(data.shape[0]):
        plt.scatter(data[i, 0], data[i, 1], color=colors(label[i]), s=2)
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig
