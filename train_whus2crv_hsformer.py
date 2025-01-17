# -- coding: utf-8 --
# @Time : 2024/11/9 16:09
# @Author : Stephanie
# @Email : sunc696@gmail.com
# @File : train_whus2crv.py
import shutil
import yaml
from attrdict import AttrMap
import time
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import itertools

from tqdm import tqdm

from data_whus2crv_manager import *
from models.HSFormer_cr import Transformer
import utils
from utils import gpu_manage, checkpoint
from eval_whus2_crv import test, test_1
from log_report import LogReport
from log_report import TestReport
from torch.cuda.amp import autocast, GradScaler
from pytorch_msssim import SSIM
import random
import torch.nn.functional as F
from loss_utils.sobel_loss import *
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

class ReplayBuffer:
    def __init__(self, max_size=50):
        assert (max_size > 0), "Empty buffer or trying to create a black hole. Be careful."
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return torch.cat(to_return)

class DecayLR:
    def __init__(self, epochs, offset, decay_epochs):
        epoch_flag = epochs - decay_epochs
        assert (epoch_flag > 0), "Decay must start before the training session ends!"
        self.epochs = epochs
        self.offset = offset
        self.decay_epochs = decay_epochs

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_epochs) / (
                self.epochs - self.decay_epochs)

import os
from glob import glob

def get_image_paths(base_path):
    data_paths = {
        "clearDNclips": {
            "10m": sorted(glob(os.path.join(base_path, "clearDNclips/10m/**/*.tif"))),
            "20m": sorted(glob(os.path.join(base_path, "clearDNclips/20m/**/*.tif"))),
            "60m": sorted(glob(os.path.join(base_path, "clearDNclips/60m/**/*.tif"))),
        },
        "cloudDNclips": {
            "10m": sorted(glob(os.path.join(base_path, "cloudDNclips/10m/**/*.tif"))),
            "20m": sorted(glob(os.path.join(base_path, "cloudDNclips/20m/**/*.tif"))),
            "60m": sorted(glob(os.path.join(base_path, "cloudDNclips/60m/**/*.tif"))),
        }
    }

    for res in ["10m", "20m", "60m"]:
        assert len(data_paths["clearDNclips"][res]) == len(data_paths["cloudDNclips"][res]), \
            f"Mismatch in number of images for resolution {res}: clearDNclips={len(data_paths['clearDNclips'][res])}, cloudDNclips={len(data_paths['cloudDNclips'][res])}"

    return data_paths


# 基础路径
base_path = "./data/WHUS2-CRv/train"
base_path_val = "./data/WHUS2-CRv/val"
base_path_test = "./data/WHUS2-CRv/test"

# 获取图像路径
image_paths = get_image_paths(base_path)
image_val_paths = get_image_paths(base_path_val)
image_test_paths = get_image_paths(base_path_test)

x_train_datalists_10m = image_paths["cloudDNclips"]["10m"]
x_train_datalists_20m = image_paths["cloudDNclips"]["20m"]
x_train_datalists_60m = image_paths["cloudDNclips"]["60m"]
y_train_datalists_10m = image_paths["clearDNclips"]["10m"]
y_train_datalists_20m = image_paths["clearDNclips"]["20m"]
y_train_datalists_60m = image_paths["clearDNclips"]["60m"]

x_val_datalists_10m = image_val_paths["cloudDNclips"]["10m"]
x_val_datalists_20m = image_val_paths["cloudDNclips"]["20m"]
x_val_datalists_60m = image_val_paths["cloudDNclips"]["60m"]
y_val_datalists_10m = image_val_paths["clearDNclips"]["10m"]
y_val_datalists_20m = image_val_paths["clearDNclips"]["20m"]
y_val_datalists_60m = image_val_paths["clearDNclips"]["60m"]

x_test_datalists_10m = image_test_paths["cloudDNclips"]["10m"]
x_test_datalists_20m = image_test_paths["cloudDNclips"]["20m"]
x_test_datalists_60m = image_test_paths["cloudDNclips"]["60m"]
y_test_datalists_10m = image_test_paths["clearDNclips"]["10m"]
y_test_datalists_20m = image_test_paths["clearDNclips"]["20m"]
y_test_datalists_60m = image_test_paths["clearDNclips"]["60m"]

scale = 10000
leny = len(y_train_datalists_60m)
k_list = np.random.randint(low=-3, high=3, size=leny)


def visualize_true_color(img):
    img = img.squeeze(0)
    true_color_img = torch.stack([img[2], img[1], img[0]], dim=-1)  # 按 RGB 顺序排列
    true_color_img = true_color_img / true_color_img.max()  # 确保像素值在 0-1 范围内
    gamma = 2.2
    true_color_img = torch.pow(true_color_img, 1 / gamma)
    true_color_img = true_color_img.cpu().numpy()
    plt.figure(figsize=(6, 6))
    plt.imshow(true_color_img)
    plt.title("True Color Composite (Bands 2, 3, 4)")
    plt.axis('off')
    plt.show()


def train(config):
    gpu_manage(config)

    ### DATASET LOAD ###
    print('===> Loading datasets')

    file_list_10m = list(zip(x_train_datalists_10m, y_train_datalists_10m))
    file_list_20m = list(zip(x_train_datalists_20m, y_train_datalists_20m))
    file_list_60m = list(zip(x_train_datalists_60m, y_train_datalists_60m))

    file_val_list_10m = list(zip(x_val_datalists_10m, y_val_datalists_10m))
    file_val_list_20m = list(zip(x_val_datalists_20m, y_val_datalists_20m))
    file_val_list_60m = list(zip(x_val_datalists_60m, y_val_datalists_60m))

    train_dataloader = iterate_paired_img(file_list_10m, file_list_20m, file_list_60m, config.batchsize, num_workers=2, pin_memory=True)

    val_dataloader = iterate_paired_img(file_val_list_10m, file_val_list_20m, file_val_list_60m, config.batchsize)

    total_batches = len(train_dataloader)


    ### MODELS LOAD ###
    print('===> Loading models')
    # gen = TCME().cuda()
    # gen = VAE(in_channels=13).cuda()
    gen = Transformer(img_size=(384,384)).cuda()

    # setup optimizer
    opt_gen = optim.Adam(gen.parameters(), lr=config.lr, betas=(config.beta1, 0.999), weight_decay=0.00001)
    criterionL1 = nn.L1Loss()
    criterionL1 = criterionL1.cuda()
    logreport = LogReport(log_dir=config.out_dir)
    validationreport = TestReport(log_dir=config.out_dir)

    criterion_psnr = nn.MSELoss()
    criterion_psnr = criterion_psnr.cuda()
    criterion_ssim = SSIM(data_range=1, size_average=True, channel=13).cuda()
    sobel_loss = sobel_l1loss_range_1()

    print('===> begin')
    start_time = time.time()

    # main
    for epoch in range(1, config.epoch + 1):

        progress_bar = tqdm(enumerate(train_dataloader), total=total_batches)
        epoch_start_time = time.time()
        for iteration, batch_img in progress_bar:

            batch_inputx_img = batch_img[0].permute(0, 3, 1, 2)  # (1, 4, 384, 384)
            batch_inputx_img1 = batch_img[2].permute(0, 3, 1, 2)  # (1, 6, 192, 192)
            batch_inputx_img2 = batch_img[4].permute(0, 3, 1, 2)
            batch_inputy_img = batch_img[1].permute(0, 3, 1, 2)  # (1, 4, 384, 384)
            batch_inputy_img1 = batch_img[3].permute(0, 3, 1, 2)  # (1, 6, 192, 192)
            batch_inputy_img2 = batch_img[5].permute(0, 3, 1, 2)

            batch_inputx_img1_upsampled = F.interpolate(batch_inputx_img1, size=(384, 384), mode='bilinear', align_corners=False)
            batch_inputx_img2_upsampled = F.interpolate(batch_inputx_img2, size=(384, 384), mode='bilinear', align_corners=False)
            batch_inputy_img1_upsampled = F.interpolate(batch_inputy_img1, size=(384, 384), mode='bilinear', align_corners=False)
            batch_inputy_img2_upsampled = F.interpolate(batch_inputy_img2, size=(384, 384), mode='bilinear', align_corners=False)

            # 在通道维度上拼接
            combined_real_a = torch.cat([batch_inputx_img, batch_inputx_img1_upsampled, batch_inputx_img2_upsampled], dim=1).cuda()  # (1, 13, 384, 384)
            combined_real_b = torch.cat([batch_inputy_img, batch_inputy_img1_upsampled, batch_inputy_img2_upsampled], dim=1).cuda()  # (1, 13, 384, 384)

            y1 = nn.functional.interpolate(combined_real_b, scale_factor=0.5, mode='bicubic')
            y2 = nn.functional.interpolate(combined_real_b, scale_factor=0.25, mode='bicubic')
            y3 = nn.functional.interpolate(combined_real_b, scale_factor=0.125, mode='bicubic')

            opt_gen.zero_grad()
            gen.train()
            scaler = GradScaler()

            with autocast():
                y_list, var_list = gen(combined_real_a)

            loss_psnr = criterion_psnr(y_list[1], combined_real_b) + criterion_psnr(y_list[2], y1) + criterion_psnr(y_list[3],
                                                                                                           y2) + criterion_psnr(
                y_list[4], y3)
            loss_psnr = 0.5 * (loss_psnr / 4.0) + criterion_psnr(y_list[0], combined_real_b)
            loss_ssim = 1 - criterion_ssim(y_list[0], combined_real_b)

            loss_sobel = 0.01* sobel_loss(y_list[1], combined_real_b) + 0.01* sobel_loss(y_list[2], y1) + 0.01* sobel_loss(y_list[3], y2) + 0.01* sobel_loss(y_list[4], y3)
            loss_sobel = 0.5 * (loss_sobel/4.0) + 0.01* sobel_loss(y_list[0], combined_real_b)
            
            s = torch.exp(-var_list[0])
            sr_ = torch.mul(y_list[0], s)
            hr_ = torch.mul(s, combined_real_b)
            loss_uncertarinty0 = criterionL1(sr_, hr_) + 0.5 * torch.mean(var_list[0])
            s1 = torch.exp(-var_list[1])
            sr_1 = torch.mul(y_list[1], s1)
            hr_1 = torch.mul(s1, combined_real_b)
            loss_uncertarinty1 = criterionL1(sr_1, hr_1) + 0.5 * torch.mean(var_list[1])
            s2 = torch.exp(-var_list[2])
            sr_2 = torch.mul(y_list[2], s2)
            hr_2 = torch.mul(s2, y1)
            loss_uncertarinty2 = criterionL1(sr_2, hr_2) + 0.5 * torch.mean(var_list[2])
            s3 = torch.exp(-var_list[3])
            sr_3 = torch.mul(y_list[3], s3)
            hr_3 = torch.mul(s3, y2)
            loss_uncertarinty3 = criterionL1(sr_3, hr_3) + 0.5 * torch.mean(var_list[3])
            s4 = torch.exp(-var_list[4])
            sr_4 = torch.mul(y_list[4], s4)
            hr_4 = torch.mul(s4, y3)
            loss_uncertarinty4 = criterionL1(sr_4, hr_4) + 0.5 * torch.mean(var_list[4])
            loss_uncertarinty = (loss_uncertarinty0 + loss_uncertarinty1 + loss_uncertarinty2 + loss_uncertarinty3 + loss_uncertarinty4) / 5.0

            loss_g_l1 = criterionL1(y_list[0], combined_real_b) * config.lamb
            loss_g = loss_g_l1 + loss_psnr + loss_uncertarinty + loss_ssim + loss_sobel

            scaler.scale(loss_g).backward()
            scaler.step(opt_gen)
            scaler.update()

            # log
            if iteration % 500 == 0:
                print("===> Epoch[{}]({}/{}): loss_L1: {:.4f} loss_psnr: {:.4f} loss_uncertarinty: {:.4f} loss_sobel: {:.4f}".format(
                    epoch, iteration, len(train_dataloader), loss_g_l1.item(), loss_psnr.item(), loss_uncertarinty.item(), loss_sobel.item()))

                log = {}
                log['epoch'] = epoch
                log['iteration'] = len(train_dataloader) * (epoch - 1) + iteration
                log['gen/loss'] = loss_g.item()

                logreport(log)

        print('epoch', epoch, 'finished, use time', time.time() - epoch_start_time)
        with torch.no_grad():
            log_validation = test_1(config, val_dataloader, gen, criterionL1, epoch)
            validationreport(log_validation)
        print('validation finished')
        if epoch % config.snapshot_interval == 0:
            checkpoint(config, epoch, gen)

        logreport.save_lossgraph()
        validationreport.save_lossgraph()
    print('training time:', time.time() - start_time)


if __name__ == '__main__':
    with open('config.yml', 'r', encoding='UTF-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = AttrMap(config)

    utils.make_manager()
    n_job = utils.job_increment()
    config.out_dir = os.path.join(config.out_dir, '{:06}'.format(n_job))
    os.makedirs(config.out_dir)
    print('Job number: {:04d}'.format(n_job))

    # 保存本次训练时的配置
    shutil.copyfile('config.yml', os.path.join(config.out_dir, 'config.yml'))

    train(config)

