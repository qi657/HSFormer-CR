# -- coding: utf-8 --
# @Time : 2024/3/16 21:39
# @Author : Stephanie
# @Email : sunc696@gmail.com
# @File : uncertain_vis.py

import numpy as np
import argparse
import cv2
import time

import torch

from utils import gpu_manage
# from models.DRDB import AHDR
from models.UDR_S2Former import Transformer

import matplotlib.pyplot as plt
from skimage import exposure


def show(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def predict(args):
    gpu_manage(args)
    ### MODELS LOAD ###
    print('===> Loading models')

    # gen = Generator(gpu_ids=args.gpu_ids)
    # gen = M3SNet().cuda()
    # gen = AHDR(args).cuda()
    gen = Transformer(img_size=(512, 512)).cuda()

    param = torch.load(args.pretrained)
    gen.load_state_dict(param)

    if args.cuda:
        gen = gen.cuda(0)

    print('<=== Model loaded')

    print('===> Loading test image')
    img = cv2.imread(args.test_filepath, 1).astype(np.float32)
    img = img / 255
    img = img.transpose(2, 0, 1)
    img = img[None]

    # print('===> Loading nir image')
    # nir = cv2.imread(args.nir_filepath, 1).astype(np.float32)
    # nir = nir / 255
    # nir = nir.transpose(2, 0, 1)
    # nir = nir[None]
    print('<=== test image loaded')

    with torch.no_grad():
        x = torch.from_numpy(img)
        # n = torch.from_numpy(nir)
        if args.cuda:
            x = x.cuda()
            # n = n.cuda()

        print('===> Removing the cloud...')
        start_time = time.time()
        # att, out = gen(x)
        # out = gen(x,n)
        out = gen(x)
        print('<=== finish! %.3fs cost.' % (time.time() - start_time))

        x_ = x.cpu().numpy()[0]
        x_rgb = x_ * 255
        x_rgb = x_rgb.transpose(1, 2, 0).astype('uint8')
        # out_ = out.cpu().numpy()[0]
        out_ = out[0][0].cpu().numpy()[0]
        out_rgb = np.clip(out_[:3], 0, 1) * 255
        out_rgb = out_rgb.transpose(1, 2, 0).astype('uint8')
        # att_ = att.cpu().numpy()[0] * 255
        # att_heatmap = heatmap(att_.astype('uint8'))[0]
        # att_heatmap = att_heatmap.transpose(1, 2, 0)

        # allim = np.hstack((x_rgb, out_rgb, att_heatmap))
        allim = np.hstack((x_rgb, out_rgb))
        # show(allim)
        # exit(0)


        # # visualize uncertainly map
        uncertainty = torch.exp(-out[1][0])
        uncertainty = torch.mul(uncertainty, x) # uncertainly map on original img
        if uncertainty.ndim > 2:
            uncertainty = uncertainty[0]

        uncertainty_numpy = uncertainty.cpu().detach().numpy()

        uncertainty_single_channel = uncertainty_numpy[0]

        eq_img = exposure.equalize_hist(uncertainty_single_channel)

        fig, ax = plt.subplots(figsize=(6, 6))
        cax = ax.imshow(eq_img, cmap='viridis')

        # cbar = plt.colorbar(cax, ax=ax, orientation='vertical', fraction=0.046, pad=0.01)  # 水平放置
        # cbar.ax.set_aspect(20) # 0.05
        plt.axis('off')
        plt.savefig(f'./paper_fig/uncertainly map-rice577-1.tiff', bbox_inches='tight', dpi=300, pad_inches=0.0)
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_filepath', type=str, default='D:/datasets/RICE_DATASET/RICE2/cloudy_image/577.png')
    # parser.add_argument('--test_filepath', type=str, default='D:/datasets/C-CUHK/CUHK-CR2/test/cloud/38.png')
    # parser.add_argument('--nir_filepath', type=str, default='D:/datasets/C-CUHK/CUHK-CR2/test/nir/38.png')
    # parser.add_argument('--pretrained', type=str, default='D:/SpA-GAN_for_cloud_removal-master/results/000083/models/gen_model_epoch_86.pth')
    # parser.add_argument('--pretrained', type=str, default='D:/SpA-GAN_for_cloud_removal-master/DRDB/gen_model_epoch_200.pth')
    parser.add_argument('--pretrained', type=str, default='D:/SpA-GAN_for_cloud_removal-master/UDR/36_gen_model_epoch_196.pth') # UDR_S2Former
    # parser.add_argument('--pretrained', type=str, default='D:/SpA-GAN_for_cloud_removal-master/AHDR/gen_model_epoch_101.pth')
    parser.add_argument('--cuda', default=True, action='store_true')
    parser.add_argument('--gpu_ids', type=int, default=[0])
    parser.add_argument('--manualSeed', type=int, default=0)
    parser.add_argument('--nChannel', type=int, default=3)
    parser.add_argument('--nDenselayer', type=int, default=10)
    parser.add_argument('--nFeat', type=int, default=64)
    parser.add_argument('--growthRate', type=int, default=32)

    args = parser.parse_args()

    predict(args)