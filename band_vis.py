# -- coding: utf-8 --
# @Time : 2024/11/12 16:19
# @Author : Stephanie
# @Email : sunc696@gmail.com
# @File : band_vis.py
import torch
import random
import torch.nn.functional as F
import matplotlib
from tqdm import tqdm

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from data_whus2crv_manager import *
from models.cvae.network import VAE
from models.HSFormer_cr import Transformer
from models.UNet import UNet
from models.pix2pix import Generator
from models.TCME_arch import TCME
from models.restormer import Restormer
from models.M3SNet import M3SNet
# from models.models_spagan.gen.SPANet import Generator
# from models.models_amgan_cr.gen.AMGAN import Generator
from models.cloud_gan import Generator, Discriminator
from models.MPRNet import MPRNet
import os


# def visualize_each_band(img, save_path=None):
#     img = img.squeeze(0)
#     num_bands = img.shape[0]  # 波段数量
#     plt.figure(figsize=(15, 10))
#     for i in range(num_bands):
#         plt.subplot(3, 5, i + 1)
#         plt.imshow(img[i, :, :].cpu().numpy(), cmap='gray')  # 单独显示每个波段
#         plt.title(f'Band {i + 1}')
#         plt.axis('off')
#     plt.tight_layout()
#
#     if save_path:
#         plt.savefig(os.path.join(save_path, "each_band_visualization.png"))
#     plt.close()


def visualize_each_band(img, save_path=None, prefix="image", image_index=0):
    img = img.squeeze(0)
    num_bands = img.shape[0]  # 波段数量

    if save_path:
        os.makedirs(save_path, exist_ok=True)

    for i in range(num_bands):
        plt.figure()
        plt.imshow(img[i, :, :].cpu().numpy(), cmap='gray')  # 单独显示每个波段
        # plt.title(f'Band {i + 1}')
        plt.axis('off')

        if save_path:
            file_name = f"{prefix}_image_{image_index}_band_{i + 1}.png"
            plt.savefig(os.path.join(save_path, file_name), dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()


def visualize_true_color(img, save_path=None, file_name="true_color.tif"):
    img = img.squeeze(0)

    true_color_img = torch.stack([img[2], img[1], img[0]], dim=-1)  # 按 RGB 顺序排列
    true_color_img = true_color_img / true_color_img.max()  # 确保像素值在 0-1 范围内

    gamma = 2.2
    true_color_img = torch.pow(true_color_img, 1 / gamma)

    true_color_img = true_color_img.cpu().numpy()

    plt.figure(figsize=(6, 6))
    plt.imshow(true_color_img)
    # plt.title("True Color Composite (Bands 2, 3, 4)")
    plt.axis('off')
    # plt.show()

    if save_path:
        plt.savefig(os.path.join(save_path, file_name), dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()



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

base_path_test = "E:/WHUS2-CRv/test"
image_test_paths = get_image_paths(base_path_test)

x_test_datalists_10m = image_test_paths["cloudDNclips"]["10m"]
x_test_datalists_20m = image_test_paths["cloudDNclips"]["20m"]
x_test_datalists_60m = image_test_paths["cloudDNclips"]["60m"]
y_test_datalists_10m = image_test_paths["clearDNclips"]["10m"]
y_test_datalists_20m = image_test_paths["clearDNclips"]["20m"]
y_test_datalists_60m = image_test_paths["clearDNclips"]["60m"]

file_list_10m = list(zip(x_test_datalists_10m, y_test_datalists_10m))
file_list_20m = list(zip(x_test_datalists_20m, y_test_datalists_20m))
file_list_60m = list(zip(x_test_datalists_60m, y_test_datalists_60m))

file_test_list_10m = list(zip(x_test_datalists_10m, y_test_datalists_10m))
file_test_list_20m = list(zip(x_test_datalists_20m, y_test_datalists_20m))
file_test_list_60m = list(zip(x_test_datalists_60m, y_test_datalists_60m))

test_dataloader = iterate_paired_img_test(file_list_10m, file_list_20m, file_list_60m, batch_size=1, num_workers=0, pin_memory=True)
total_batches = len(test_dataloader)


# gen = VAE(in_channels=13).cuda()
gen = Transformer(img_size=(384,384)).cuda()
# gen = UNet().cuda()
# gen = Generator().cuda()  # pix2pix
# gen = TCME().cuda()
# gen = Restormer().cuda()
# gen = M3SNet().cuda()
# gen = Generator(gpu_ids=1, channel=13).cuda()  # sapgan
# gen = Generator(in_ch=13, out_ch=13, gpu_ids=1) # amgan
# gen = Generator().cuda()  # cloudgan
# gen = MPRNet(in_c=13, out_c=13).cuda()  # MPRNet

# param = torch.load('D:/SpA-GAN_for_cloud_removal-master/results/000426/models/gen_model_epoch_46.pth')  # cvae
# param = torch.load('D:/SpA-GAN_for_cloud_removal-master/results/000457/models/gen_model_epoch_1.pth')   # hsformer
param = torch.load('D:/SpA-GAN_for_cloud_removal-master/results/000458/models/gen_model_epoch_7.pth')   # hsformer_1
# param = torch.load('D:/SpA-GAN_for_cloud_removal-master/results/000469/models/gen_model_epoch_15.pth')   # UNet
# param = torch.load('D:/SpA-GAN_for_cloud_removal-master/results/000472/models/gen_model_epoch_20.pth')   # STGAN
# param = torch.load('D:/SpA-GAN_for_cloud_removal-master/results/pcl_cloud/TCME_whu_410_gen_model_epoch_1.pth')   # TCME
# param = torch.load('D:/SpA-GAN_for_cloud_removal-master/results/pcl_cloud/TCME_whu_410_gen_model_epoch_3.pth')   # TCME_1
# param = torch.load('D:/SpA-GAN_for_cloud_removal-master/results/pcl_cloud/restromer_whu_408_gen_model_epoch_8.pth')   # Restormer
# param = torch.load('D:/SpA-GAN_for_cloud_removal-master/results/pcl_cloud/M3SNet_whu_445_gen_model_epoch_9.pth')   # M3SNet
# param = torch.load('D:/SpA-GAN_for_cloud_removal-master/results/000480/models/gen_model_epoch_2.pth')   # SPAGAN
# param = torch.load('D:/SpA-GAN_for_cloud_removal-master/results/pcl_cloud/amgan_456_gen_model_epoch_4.pth')   # AMGAN
# param = torch.load('D:/SpA-GAN_for_cloud_removal-master/results/pcl_cloud/cloudgan_whus2crv_gen_model_epoch_7.pth')   # cloudgan
# param = torch.load('D:/SpA-GAN_for_cloud_removal-master/results/000499/models/gen_model_epoch_2.pth')   # MPRNet
gen.load_state_dict(param)
gen = gen.cuda(0)

num_images_to_infer = 50
specific_indices = [468, 471,470, 480, 490, 606, 666, 900, 1030,1000, 1100, 1117, 1150, 1155, 1173, 1174, 1202, 1206, 1475, 1500,  1597, 1600,1666, 1673, 1700, 2138, 2234,
                    2800, 3812, 4300]
processed_images = 0
# output_dir = './fig_whus2crv/cvae'
# output_dir = './fig_whus2crv/hsformer'
output_dir = './fig_whus2crv/hsformer_1'
# output_dir = './fig_whus2crv/UNet'
# output_dir = './fig_whus2crv/STGAN'
# output_dir = './fig_whus2crv/TCME'
# output_dir = './fig_whus2crv/TCME_1'
# output_dir = './fig_whus2crv/restormer'
# output_dir = './fig_whus2crv/M3SNet'
# output_dir = './fig_whus2crv/SPAGAN'
# output_dir = './fig_whus2crv/AMGAN'
# output_dir = './fig_whus2crv/cloudgan'
# output_dir = './fig_whus2crv/MPRNet'

progress_bar = tqdm(enumerate(test_dataloader), total=total_batches)

for i, batch_img in progress_bar:
    if i in specific_indices:
        batch_inputx_img = batch_img[0].permute(0, 3, 1, 2)  # (1, 4, 384, 384)
        batch_inputx_img1 = batch_img[2].permute(0, 3, 1, 2)  # (1, 6, 192, 192)
        batch_inputx_img2 = batch_img[4].permute(0, 3, 1, 2)
        batch_inputy_img = batch_img[1].permute(0, 3, 1, 2)  # (1, 4, 384, 384)
        batch_inputy_img1 = batch_img[3].permute(0, 3, 1, 2)  # (1, 6, 192, 192)
        batch_inputy_img2 = batch_img[5].permute(0, 3, 1, 2)

        batch_inputx_img1_upsampled = F.interpolate(batch_inputx_img1, size=(384, 384), mode='bilinear',
                                                    align_corners=False)
        batch_inputx_img2_upsampled = F.interpolate(batch_inputx_img2, size=(384, 384), mode='bilinear',
                                                    align_corners=False)
        batch_inputy_img1_upsampled = F.interpolate(batch_inputy_img1, size=(384, 384), mode='bilinear',
                                                    align_corners=False)
        batch_inputy_img2_upsampled = F.interpolate(batch_inputy_img2, size=(384, 384), mode='bilinear',
                                                    align_corners=False)

        # 在通道维度上拼接
        combined_real_a = torch.cat([batch_inputx_img, batch_inputx_img1_upsampled, batch_inputx_img2_upsampled],
                                    dim=1).cuda()
        combined_real_b = torch.cat([batch_inputy_img, batch_inputy_img1_upsampled, batch_inputy_img2_upsampled],
                                    dim=1).cuda()

        with torch.no_grad():
            # out= gen(combined_real_a)
            out, _ = gen(combined_real_a)

        save_path = os.path.join(output_dir, f"image_{i}")
        os.makedirs(save_path, exist_ok=True)

        visualize_each_band(out[0], save_path=save_path, prefix="decloud", image_index=i)
        visualize_true_color(out[0], save_path=save_path, file_name=f"decloud_image_{i}.tif")
        visualize_each_band(combined_real_a, save_path=save_path, prefix="cloud", image_index=i)
        visualize_true_color(combined_real_a, save_path=save_path, file_name=f"cloud_image_{i}.tif")
        visualize_each_band(combined_real_b, save_path=save_path, prefix="clear", image_index=i)
        visualize_true_color(combined_real_b, save_path=save_path, file_name=f"clear_image_{i}.tif")

        processed_images += 1

    # 提前结束循环
    if processed_images >= len(specific_indices):
        break
