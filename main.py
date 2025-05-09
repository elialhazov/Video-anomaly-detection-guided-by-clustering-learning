from torch.autograd import Variable
from torchvision.transforms import InterpolationMode
import argparse
import os
import sys
import time
import math
from pathlib import Path

from einops import rearrange
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision import models as torchvision_models
import dataset.utils_dataset as dt
from misc import utils
from model import swin_transformer as encoder
from model import swin_decoder as decoder
from model import backbone
from loss_tool.Recon_Loss import Recon_Loss
from tqdm import tqdm
import matplotlib.pyplot as plt

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torchvision_archs = sorted(name for name in torchvision_models.__dict__
                           if name.islower() and not name.startswith("__")
                           and callable(torchvision_models.__dict__[name]))

def get_args_parser():
    parser = argparse.ArgumentParser('Swin Transformer', add_help=False)
    parser.add_argument('--frame_num', default=4, type=int)
    parser.add_argument('--patch_size', default=(2, 4, 4), type=tuple)
    parser.add_argument('--weight_decay', type=float, default=0.024)
    parser.add_argument('--weight_decay_end', type=float, default=0.02)
    parser.add_argument('--clip_grad', type=float, default=3.0)
    parser.add_argument('--batch_size_per_gpu', default=3, type=int)
    parser.add_argument('--epochs', default=1, type=int)  # Run only 1 epoch for quick test
    parser.add_argument('--freeze_last_layer', default=2, type=int)
    parser.add_argument("--lr", default=0.000008, type=float)
    parser.add_argument("--warmup_epochs", default=5, type=int)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--optimizer', default='adamw', type=str, choices=['adamw', 'sgd'])
    parser.add_argument('--drop_path_rate', type=float, default=0.1)
    parser.add_argument('--image_format', default='jpg', type=str, help='Image file extension (e.g. jpg or png)')
    parser.add_argument('--data_path', default='dataset/ucsdpeds/vidf', type=str)
    parser.add_argument('--model_pretrain', default='', type=str)
    parser.add_argument('--output_dir', default="log_dir", type=str)
    parser.add_argument('--saveckp_freq', default=20, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    return parser

def train_dino(args):
    utils.fix_random_seeds(args.seed)
    cudnn.benchmark = True

    transform = dt.DataTransforms()
    dataset = dt.DataLoader(args.data_path, transform=transform, frames_num=args.frame_num, image_format=args.image_format)
    dataset.samples = dataset.samples[:50]

    sampler = torch.utils.data.RandomSampler(dataset)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Loaded {len(dataset)} video clips.")

    model = backbone.Mymodel(args).to(device)
    print("Model initialized.")

    recon_loss = nn.MSELoss(reduction='none')
    loss_log = []
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.epochs, eta_min=args.min_lr, last_epoch=-1)

    plt.ion()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    logging_file_path = os.path.join(args.output_dir, 'exp.log')
    logger = utils.get_logger(logging_file_path)

    print("Training started...")
    data_iter = 0
    for epoch in range(args.epochs):
        model.train()
        if epoch % 1 == 0:
            torch.save(model.state_dict(), os.path.join(args.output_dir, f'checkpoint{epoch}.pth'))
        train_stats, data_iter = train_one_epoch(model, recon_loss, data_loader, optimizer, scheduler, epoch, data_iter, dataset, logger, args, loss_log)

def train_one_epoch(model, recon_loss, data_loader, optimizer, scheduler, epoch, data_iter, dataset, logger, args, loss_log):
    loop = tqdm(data_loader, leave=False)
    for it, (video, idx) in enumerate(loop):
        video = video.to(device)

        recon_video, cluster_loss, space_loss, *_ = model(video)

        if data_iter % 10 == 0:
            utils.save_tensor_video(video, output_dir='video_show_origin')
            utils.save_tensor_video(recon_video)

        optimizer.zero_grad()
        loss_pixel = torch.mean(recon_loss(recon_video, video))
        if cluster_loss is not None:
            cluster_loss = torch.mean(cluster_loss)
            space_loss = torch.mean(space_loss)
            loss = loss_pixel + cluster_loss + space_loss
        else:
            loss = loss_pixel
        loss.backward(retain_graph=True)

        if not math.isfinite(loss.item()):
            print(f"Loss is {loss.item()}, stopping training", force=True)
            sys.exit(1)
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        logger.info(f'Epoch:[{epoch}/{args.epochs}] batch:[{it}/{len(loop)}] loss={loss:.5f} lr={lr:.3f}')

        if data_iter > 2500:
            loss_log.append(float(loss.detach().cpu().numpy()))
            plt.clf()
            plt.plot(loss_log, 'r', label='Training loss')
            plt.title('Training loss')
            plt.legend()
            plt.pause(0.001)
            plt.ioff()

        data_iter += 1
        loop.set_description(f'Epoch [{epoch}/{args.epochs}]')
        loop.set_postfix(loss=loss.item())
        optimizer.step()
    scheduler.step()
    return 0, data_iter

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_dino(args)
