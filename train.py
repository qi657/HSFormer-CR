import os
import random
import shutil
import yaml
from attrdict import AttrMap
import time

import torch
from torch import nn
from torch.backends import cudnn
from torch import optim
from torch.utils.data import DataLoader

from torch.autograd import Variable
from torch.nn import functional as F

from data_manager import TrainDataset
from models.HSFormer import Transformer
import utils
from utils import gpu_manage, save_image, checkpoint
from eval import test
from log_report import LogReport
from log_report import TestReport
from pytorch_msssim import SSIM
import lpips
from loss_utils.sobel_loss import *


def train(config):
    gpu_manage(config)

    ### DATASET LOAD ###
    print('===> Loading datasets')

    dataset = TrainDataset(config)
    print('dataset:', len(dataset))
    train_size = int((1 - config.validation_size) * len(dataset))
    validation_size = len(dataset) - train_size
    train_dataset, validation_dataset = torch.utils.data.random_split(dataset, [train_size, validation_size])
    print('train dataset:', len(train_dataset))
    print('validation dataset:', len(validation_dataset))
    training_data_loader = DataLoader(dataset=train_dataset, num_workers=config.threads, batch_size=config.batchsize, shuffle=True)
    validation_data_loader = DataLoader(dataset=validation_dataset, num_workers=config.threads, batch_size=config.validation_batchsize, shuffle=False)
    
    ### MODELS LOAD ###
    print('===> Loading models')
    gen = Transformer(img_size=(512,512)).cuda()
#     param = torch.load('/code/SpA-GAN_for_cloud_removal-master/results/000072/models/gen_model_epoch_197.pth')
#     gen1.load_state_dict(param)
    # print(gen)
    # exit(0)

    if config.gen_init is not None:
        param = torch.load(config.gen_init)
        gen.load_state_dict(param)
        print('load {} as pretrained model'.format(config.gen_init))


    # setup optimizer
    opt_gen = optim.Adam(gen.parameters(), lr=config.lr, betas=(config.beta1, 0.999), weight_decay=0.00001)

    real_a = torch.FloatTensor(config.batchsize, config.in_ch, config.width, config.height)
    real_b = torch.FloatTensor(config.batchsize, config.out_ch, config.width, config.height)

    criterionL1 = nn.L1Loss()
    criterionMSE = nn.MSELoss()
    criterion_psnr = nn.MSELoss()
    criterion_psnr = criterion_psnr.cuda()
    criterion_ssim = SSIM(data_range=1, size_average=True, channel=3).cuda()
    loss_fn_vgg = lpips.LPIPS(net='vgg').cuda()
    sobel_loss = sobel_l1loss_range_1()

    def ColorLoss(x1, x2):
        return torch.sum(torch.pow((x1 - x2), 2)).div(2 * x1.size()[0])
    

    criterionL1 = criterionL1.cuda()
    criterionMSE = criterionMSE.cuda()
    real_a = real_a.cuda()
    real_b = real_b.cuda()

    logreport = LogReport(log_dir=config.out_dir)
    validationreport = TestReport(log_dir=config.out_dir)

    print('===> begin')
    start_time=time.time()
    # main
    for epoch in range(1, config.epoch + 1):
        epoch_start_time = time.time()
        for iteration, batch in enumerate(training_data_loader, 1):
            opt_gen.zero_grad()
            gen.train()
            real_a, real_b, M = batch[0].cuda(), batch[1].cuda(), batch[2].cuda()

            y1 = nn.functional.interpolate(real_b, scale_factor=0.5, mode='bicubic')
            y2 = nn.functional.interpolate(real_b, scale_factor=0.25, mode='bicubic')
            y3 = nn.functional.interpolate(real_b, scale_factor=0.125, mode='bicubic')

            fake_b = gen(real_a)
    
            loss_g_l1 = criterionL1(y_list[0], real_b) * config.lamb
            loss_psnr = criterion_psnr(y_list[1], real_b) + criterion_psnr(y_list[2], y1) + criterion_psnr(y_list[3], y2) + criterion_psnr(y_list[4], y3)
            loss_psnr = 0.5*(loss_psnr/4.0) + criterion_psnr(y_list[0],real_b)

            loss_ssim = 1 - criterion_ssim(y_list[0], real_b)
            loss_lpips = torch.mean(loss_fn_vgg(y_list[0], real_b))
            
            loss_sobel = 0.01* sobel_loss(y_list[1], real_b) + 0.01* sobel_loss(y_list[2], y1) + 0.01* sobel_loss(y_list[3], y2) + 0.01* sobel_loss(y_list[4], y3)
            loss_sobel = 0.5 * (loss_sobel/4.0) + 0.01* sobel_loss(y_list[0], real_b)
            
#             loss_color = 0.00001* ColorLoss(y_list[1], real_b) + 0.00001* ColorLoss(y_list[2], y1) + 0.00001* ColorLoss(y_list[3], y2) + 0.00001* ColorLoss(y_list[4], y3)
#             loss_color = 0.5 * (loss_color/4.0) + 0.00001* ColorLoss(y_list[0], real_b)

            s = torch.exp(-var_list[0])
            sr_ = torch.mul(y_list[0], s)
            hr_ = torch.mul(s, real_b)
            loss_uncertarinty0 = criterionL1(sr_, hr_) + 0.5 * torch.mean(var_list[0])
            s1 = torch.exp(-var_list[1])
            sr_1 = torch.mul(y_list[1], s1)
            hr_1 = torch.mul(s1, real_b)
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

            loss_g_l1 = criterionL1(fake_b, real_b) * config.lamb
            loss_psnr = criterion_psnr(fake_b, real_b)
            loss_ssim = 1 - criterion_ssim(fake_b, real_b)
            loss_lpips = torch.mean(loss_fn_vgg(y_list[0], real_b))
            loss_g = loss_psnr + loss_g_l1 + loss_ssim  + 0.1 * loss_uncertarinty + loss_sobel

            loss_g.backward()

            opt_gen.step()

            # log
            if iteration % 10 == 0:
                print("===> Epoch[{}]({}/{}): loss_l1: {:.4f} loss_psnr: {:.4f} loss_ssim: {:.4f} loss_sobel: {:.4f} loss_uncertarinty: {:.4f}".format(
                epoch, iteration, len(training_data_loader), loss_g_l1.item(), loss_psnr.item(), loss_ssim.item(), loss_sobel.item(), loss_uncertarinty.item()))
                
                log = {}
                log['epoch'] = epoch
                log['iteration'] = len(training_data_loader) * (epoch-1) + iteration
                log['gen/loss'] = loss_g.item()

                logreport(log)

        print('epoch', epoch, 'finished, use time', time.time() - epoch_start_time)
        with torch.no_grad():
            log_validation = test(config, validation_data_loader, gen, criterionMSE, epoch)
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

    shutil.copyfile('config.yml', os.path.join(config.out_dir, 'config.yml'))

    train(config)
