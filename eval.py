import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim

from torch.autograd import Variable

from utils import save_image
import lpips
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM

loss_fn = lpips.LPIPS(net='vgg').cuda()

def caculate_lpips(img0,img1):
    im1=np.copy(img0.cpu().numpy())
    im2=np.copy(img1.cpu().numpy())
    im1=torch.from_numpy(im1.astype(np.float32))
    im2 = torch.from_numpy(im2.astype(np.float32))
    im1.unsqueeze_(0)
    im2.unsqueeze_(0)
    current_lpips_distance  = loss_fn.forward(im1, im2)
    return current_lpips_distance

def caculate_ssim(imgA, imgB):
    imgA1 = np.tensordot(imgA.cpu().numpy().transpose(1, 2, 0), [0.298912, 0.586611, 0.114478], axes=1)
    imgB1 = np.tensordot(imgB.cpu().numpy().transpose(1, 2, 0), [0.298912, 0.586611, 0.114478], axes=1)
    score = SSIM(imgA1, imgB1, data_range=255)
    return score

def caculate_psnr( imgA, imgB):
    imgA1 = imgA.cpu().numpy().transpose(1, 2, 0)
    imgB1 = imgB.cpu().numpy().transpose(1, 2, 0)
    psnr = PSNR(imgA1, imgB1, data_range=255)
    return psnr

def test(config, test_data_loader, gen, criterionMSE, epoch):
    avg_mse = 0.0
    avg_psnr = 0.0
    avg_ssim = 0.0
    # avg_lpips = 0.0
    for i, batch in enumerate(test_data_loader):
        x, t, n = Variable(batch[0]), Variable(batch[1]), Variable(batch[3])  # val, test Variable(batch[2])
        if config.cuda:
            x = x.cuda(0)
            t = t.cuda(0)
            n = n.cuda(0)

        # att, out = gen(x)
        out = gen(x,n)
        # y_list, var_list = gen(x, n)

        if epoch % config.snapshot_interval == 0:
            h = 1
            w = 3
            c = 3
            width = config.width
            height = config.height

            allim = np.zeros((h, w, c, width, height))
            x_ = x.cpu().numpy()[0]
            t_ = t.cpu().numpy()[0]
            out_ = out.cpu().numpy()[0]
            # out_ = y_list[0].cpu().numpy()[0]
            in_rgb = x_[:3]
            t_rgb = t_[:3]
            out_rgb = np.clip(out_[:3], 0, 1)
            allim[0, 0, :] = in_rgb * 255
            allim[0, 1, :] = out_rgb * 255
            allim[0, 2, :] = t_rgb * 255
            
            allim = allim.transpose(0, 3, 1, 4, 2)
            allim = allim.reshape((h*height, w*width, c))

            save_image(config.out_dir, allim, i, epoch)

        mse = criterionMSE(out, t)
        # mse = criterionMSE(y_list[0], t)
        psnr = 10 * np.log10(1 / mse.item())

        img1 = np.tensordot(out.cpu().numpy()[0, :3].transpose(1, 2, 0), [0.298912, 0.586611, 0.114478], axes=1)
        # img1 = np.tensordot(y_list[0].cpu().numpy()[0, :3].transpose(1, 2, 0), [0.298912, 0.586611, 0.114478], axes=1)
        img2 = np.tensordot(t.cpu().numpy()[0, :3].transpose(1, 2, 0), [0.298912, 0.586611, 0.114478], axes=1)
        
        ssim_value = ssim(img1, img2)
        # lpips = caculate_lpips(img1, img2)
        avg_mse += mse.item()
        avg_psnr += psnr
        avg_ssim += ssim_value
        # avg_lpips += lpips
    avg_mse = avg_mse / len(test_data_loader)
    avg_psnr = avg_psnr / len(test_data_loader)
    avg_ssim = avg_ssim / len(test_data_loader)
    # avg_lpips = avg_lpips / len(test_data_loader)

    print("===> Avg. MSE: {:.4f}".format(avg_mse))
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr))
    print("===> Avg. SSIM: {:.4f} dB".format(avg_ssim))
    # print("===> Avg. Lpips: {:.4f} dB".format(avg_lpips.item()))
    
    log_test = {}
    log_test['epoch'] = epoch
    log_test['mse'] = avg_mse
    log_test['psnr'] = avg_psnr
    log_test['ssim'] = avg_ssim

    return log_test
