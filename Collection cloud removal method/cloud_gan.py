import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            #Initial Convolution
            nn.ReflectionPad2d(3),
            nn.Conv2d(13, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace = True),

            #Encoder
            nn.Conv2d(64, 128, 3, stride = 2, padding = 1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace = True),
            nn.Conv2d(128, 256, 3, stride = 2, padding = 1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace = True),

            #Residual Block Sequence
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),

            #Decoder
            nn.ConvTranspose2d(256, 128, 3, stride = 2, padding = 1, output_padding = 1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace = True),
            nn.ConvTranspose2d(128, 64, 3, stride = 2, padding = 1, output_padding = 1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace = True),

            #Output
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 13, 7),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()

        self.res = nn.Sequential(nn.ReflectionPad2d(1),
                                 nn.Conv2d(in_channels, in_channels, 3),
                                 nn.InstanceNorm2d(in_channels),
                                 nn.ReLU(inplace = True),
                                 nn.ReflectionPad2d(1),
                                 nn.Conv2d(in_channels, in_channels, 3),
                                 nn.InstanceNorm2d(in_channels))

    def forward(self, x):
        return x + self.res(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(13, 64, 4, stride = 2, padding = 1),
            nn.LeakyReLU(0.2, inplace = True),

            nn.Conv2d(64, 128, 4, stride = 2, padding = 1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace = True),

            nn.Conv2d(128, 256, 4, stride = 2, padding = 1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace = True),

            nn.Conv2d(256, 512, 4, padding = 1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace = True),

            nn.Conv2d(512, 1, 13, padding = 1),
            )

    def forward(self, x):
        x = self.main(x)
        x = F.avg_pool2d(x, x.size()[2:])
        x = torch.flatten(x, 1)
        #x = torch.sigmoid(x)
        return x


# from ptflops import get_model_complexity_info
# import time
# import numpy as np
#
# models = Generator().cuda()
# H,W=256,256
# flops_t, params_t = get_model_complexity_info(models, (3, H,W), as_strings=True, print_per_layer_stat=True)
# print(models)
# print(f"net flops:{flops_t} parameters:{params_t}")
# # models = nn.DataParallel(models)
# x = torch.ones([1,3,H,W]).cuda()
# b = models(x)
# steps=25
# # print(b)
# time_avgs=[]
# memory_avgs=[]
# with torch.no_grad():
#     for step in range(steps):
#         torch.cuda.synchronize()
#         start = time.time()
#         result = models(x)
#         torch.cuda.synchronize()
#         time_interval = time.time() - start
#         memory = torch.cuda.max_memory_allocated()  # 获取每步的最大内存分配
#         torch.cuda.reset_peak_memory_stats()  # 重置峰值内存统计，以便每次迭代都是从零开始
#         if step > 5:
#             time_avgs.append(time_interval)
#             memory_avgs.append(memory)
#
# memory_mean = np.mean(memory_avgs)  # 计算平均内存使用
# print(
#     f'avg time: {np.mean(time_avgs)}, fps: {1 / np.mean(time_avgs)}, memory: {memory_mean / (1024 ** 2)} MB, size: {H, W}')
#
# total_params = sum(p.numel() for p in models.parameters())
# memory_usage = total_params * 4  # Assuming 32-bit floats
# print(f"Total memory for model parameters: {memory_usage / (1024 ** 2):.2f} MB")
#
# with torch.no_grad():  # Ensure we're not tracking gradients for this test
#     models(x)
#     print(torch.cuda.memory_summary(device='cuda', abbreviated=True))