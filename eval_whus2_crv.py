# -- coding: utf-8 --
# @Time : 2024/11/13 16:24
# @Author : Stephanie
# @Email : sunc696@gmail.com
# @File : eval_whus2_crv.py
import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim

from torch.autograd import Variable

from utils import save_image
import lpips
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM

loss_fn = lpips.LPIPS(net='vgg').cuda()


def caculate_lpips(img0, img1):
    im1 = np.copy(img0.cpu().numpy())
    im2 = np.copy(img1.cpu().numpy())
    im1 = torch.from_numpy(im1.astype(np.float32))
    im2 = torch.from_numpy(im2.astype(np.float32))
    im1.unsqueeze_(0)
    im2.unsqueeze_(0)
    current_lpips_distance = loss_fn.forward(im1, im2)
    return current_lpips_distance


def caculate_ssim(imgA, imgB):
    imgA1 = np.tensordot(imgA.cpu().numpy().transpose(1, 2, 0), [0.298912, 0.586611, 0.114478], axes=1)
    imgB1 = np.tensordot(imgB.cpu().numpy().transpose(1, 2, 0), [0.298912, 0.586611, 0.114478], axes=1)
    score = SSIM(imgA1, imgB1, data_range=255)
    return score


def calculate_multiband_ssim(img1, img2):
    assert img1.shape == img2.shape, "Images must have the same shape"
    num_bands = img1.shape[0]
    ssim_total = 0
    for i in range(num_bands):
        band_ssim = SSIM(img1[i], img2[i], data_range=img2[i].max() - img2[i].min())
        ssim_total += band_ssim
    avg_ssim = ssim_total / num_bands
    return avg_ssim

def caculate_psnr(imgA, imgB):
    imgA1 = imgA.cpu().numpy().transpose(1, 2, 0)
    imgB1 = imgB.cpu().numpy().transpose(1, 2, 0)
    psnr = PSNR(imgA1, imgB1, data_range=255)
    return psnr


def test(config, test_data_loader, gen, criterionMSE, epoch):
    avg_mse = 0.0
    avg_psnr = 0.0
    avg_ssim = 0.0
    for i, batch_img in enumerate(test_data_loader):
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

        combined_real_a = torch.cat([batch_inputx_img, batch_inputx_img1_upsampled, batch_inputx_img2_upsampled],
                                    dim=1).cuda()  # (1, 13, 384, 384)
        combined_real_b = torch.cat([batch_inputy_img, batch_inputy_img1_upsampled, batch_inputy_img2_upsampled],
                                    dim=1).cuda()  # (1, 13, 384, 384)

        x, t = Variable(combined_real_a), Variable(combined_real_b)
        if config.cuda:
            x = x.cuda(0)
            t = t.cuda(0)

        out = gen(x)

        mse = criterionMSE(out, t)
        psnr = 10 * np.log10(1 / mse.item())

        img1 = out.cpu().numpy()[0]  # 假设 out 是 13 个波段的输出
        img2 = t.cpu().numpy()[0]

        ssim_value = calculate_multiband_ssim(img1, img2)
        avg_mse += mse.item()
        avg_psnr += psnr
        avg_ssim += ssim_value


    avg_mse = avg_mse / len(test_data_loader)
    avg_psnr = avg_psnr / len(test_data_loader)
    avg_ssim = avg_ssim / len(test_data_loader)

    print("===> Avg. MSE: {:.4f}".format(avg_mse))
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr))
    print("===> Avg. SSIM: {:.4f} dB".format(avg_ssim))

    log_test = {}
    log_test['epoch'] = epoch
    log_test['mse'] = avg_mse
    log_test['psnr'] = avg_psnr
    log_test['ssim'] = avg_ssim

    return log_test

def test_1(config, test_data_loader, gen, criterionMSE, epoch):
    avg_mse = 0.0
    avg_psnr = 0.0
    avg_ssim = 0.0
    for i, batch_img in enumerate(test_data_loader):
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

        combined_real_a = torch.cat([batch_inputx_img, batch_inputx_img1_upsampled, batch_inputx_img2_upsampled],
                                    dim=1).cuda()  # (1, 13, 384, 384)
        combined_real_b = torch.cat([batch_inputy_img, batch_inputy_img1_upsampled, batch_inputy_img2_upsampled],
                                    dim=1).cuda()  # (1, 13, 384, 384)

        x, t = Variable(combined_real_a), Variable(combined_real_b)
        if config.cuda:
            x = x.cuda(0)
            t = t.cuda(0)

        y_list, var_list = gen(x)

        mse = criterionMSE(y_list[0], t)
        psnr = 10 * np.log10(1 / mse.item())

        img1 = y_list[0].cpu().numpy()[0]  # 假设 out 是 13 个波段的输出
        img2 = t.cpu().numpy()[0]

        ssim_value = calculate_multiband_ssim(img1, img2)
        avg_mse += mse.item()
        avg_psnr += psnr
        avg_ssim += ssim_value


    avg_mse = avg_mse / len(test_data_loader)
    avg_psnr = avg_psnr / len(test_data_loader)
    avg_ssim = avg_ssim / len(test_data_loader)

    print("===> Avg. MSE: {:.4f}".format(avg_mse))
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr))
    print("===> Avg. SSIM: {:.4f} dB".format(avg_ssim))

    log_test = {}
    log_test['epoch'] = epoch
    log_test['mse'] = avg_mse
    log_test['psnr'] = avg_psnr
    log_test['ssim'] = avg_ssim

    return log_test

def test_2(config, test_data_loader, gen, criterionMSE, epoch):
    avg_mse = 0.0
    avg_psnr = 0.0
    avg_ssim = 0.0
    for i, batch_img in enumerate(test_data_loader):
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

        combined_real_a = torch.cat([batch_inputx_img, batch_inputx_img1_upsampled, batch_inputx_img2_upsampled],
                                    dim=1).cuda()  # (1, 13, 384, 384)
        combined_real_b = torch.cat([batch_inputy_img, batch_inputy_img1_upsampled, batch_inputy_img2_upsampled],
                                    dim=1).cuda()  # (1, 13, 384, 384)

        x, t = Variable(combined_real_a), Variable(combined_real_b)
        if config.cuda:
            x = x.cuda(0)
            t = t.cuda(0)

        y_list = gen(x)

        mse = criterionMSE(y_list[0], t)
        psnr = 10 * np.log10(1 / mse.item())

        img1 = y_list[0].cpu().numpy()[0]  # 假设 out 是 13 个波段的输出
        img2 = t.cpu().numpy()[0]

        ssim_value = calculate_multiband_ssim(img1, img2)
        avg_mse += mse.item()
        avg_psnr += psnr
        avg_ssim += ssim_value


    avg_mse = avg_mse / len(test_data_loader)
    avg_psnr = avg_psnr / len(test_data_loader)
    avg_ssim = avg_ssim / len(test_data_loader)

    print("===> Avg. MSE: {:.4f}".format(avg_mse))
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr))
    print("===> Avg. SSIM: {:.4f} dB".format(avg_ssim))

    log_test = {}
    log_test['epoch'] = epoch
    log_test['mse'] = avg_mse
    log_test['psnr'] = avg_psnr
    log_test['ssim'] = avg_ssim

    return log_test
