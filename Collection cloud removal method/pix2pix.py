# -- coding: utf-8 --
# @Time : 2024/4/24 11:37
# @Author : Stephanie
# @Email : sunc696@gmail.com
# @File : pix2pix.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.conv import ConvTranspose2d


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, act="relu", use_dropout=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False, padding_mode="reflect")
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2)
        )
        self.use_dropout = use_dropout
        self.dropput = nn.Dropout(0.5)
        # self.down = down

    def forward(self, x):
        x = self.conv(x)
        return self.dropput(x) if self.use_dropout else x


class Generator(nn.Module):
    def __init__(self, in_channels=13, features=64):
        super().__init__()
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )
        self.down1 = Block(features, features * 2, down=True, act="leaky", use_dropout=False)
        self.down2 = Block(features * 2, features * 4, down=True, act="leaky", use_dropout=False)
        self.down3 = Block(features * 4, features * 8, down=True, act="leaky", use_dropout=False)
        self.down4 = Block(features * 8, features * 8, down=True, act="leaky", use_dropout=False)
        self.down5 = Block(features * 8, features * 8, down=True, act="leaky", use_dropout=False)
        self.down6 = Block(features * 8, features * 8, down=True, act="leaky", use_dropout=False)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, 4, 2, 1),
            nn.ReLU(),
        )

        self.up1 = Block(features * 8, features * 8, down=False, act="relu", use_dropout=True)
        self.up2 = Block(features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True)
        self.up3 = Block(features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True)
        self.up4 = Block(features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True)
        self.up5 = Block(features * 8 * 2, features * 4, down=False, act="relu", use_dropout=True)
        self.up6 = Block(features * 4 * 2, features * 2, down=False, act="relu", use_dropout=True)
        self.up7 = Block(features * 2 * 2, features * 1, down=False, act="relu", use_dropout=True)
        self.final_up = nn.Sequential(
            ConvTranspose2d(features * 2, in_channels, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        d1 = self.initial_down(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)
        d7 = self.down6(d6)
        bottleneck = self.bottleneck(d7)

        # 上采样和拼接
        up1 = self.up1(bottleneck)

        # 调整 up1 和 d7 的尺寸
        if up1.shape[2:] != d7.shape[2:]:
            up1 = F.interpolate(up1, size=d7.shape[2:], mode="bilinear", align_corners=True)

        up2 = self.up2(torch.cat([up1, d7], 1))

        # 同样地，调整 up2 和 d6 的尺寸
        if up2.shape[2:] != d6.shape[2:]:
            up2 = F.interpolate(up2, size=d6.shape[2:], mode="bilinear", align_corners=True)

        up3 = self.up3(torch.cat([up2, d6], 1))

        # 重复上述操作，确保拼接前的尺寸匹配
        if up3.shape[2:] != d5.shape[2:]:
            up3 = F.interpolate(up3, size=d5.shape[2:], mode="bilinear", align_corners=True)

        up4 = self.up4(torch.cat([up3, d5], 1))

        if up4.shape[2:] != d4.shape[2:]:
            up4 = F.interpolate(up4, size=d4.shape[2:], mode="bilinear", align_corners=True)

        up5 = self.up5(torch.cat([up4, d4], 1))

        if up5.shape[2:] != d3.shape[2:]:
            up5 = F.interpolate(up5, size=d3.shape[2:], mode="bilinear", align_corners=True)

        up6 = self.up6(torch.cat([up5, d3], 1))

        if up6.shape[2:] != d2.shape[2:]:
            up6 = F.interpolate(up6, size=d2.shape[2:], mode="bilinear", align_corners=True)

        up7 = self.up7(torch.cat([up6, d2], 1))

        # 最后一次拼接
        if up7.shape[2:] != d1.shape[2:]:
            up7 = F.interpolate(up7, size=d1.shape[2:], mode="bilinear", align_corners=True)

        final = self.final_up(torch.cat([up7, d1], 1))
        return final


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride, 1, bias=False, padding_mode="reflect"),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.conv(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels=13):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels * 2, 64, 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),

            CNNBlock(64, 128, 2),
            CNNBlock(128, 256, 2),
            CNNBlock(256, 512, 1),

            nn.Conv2d(512, 1, 4, 1, 1, padding_mode="reflect"),
        )

    def forward(self, x):
        #         x1 = torch.cat([x, y], dim=1)
        x2 = self.model(x)
        return x2  # 输出的30*30的patch，而不是1*1


# def test():
#     x = torch.randn((1, 3, 256, 256))
#     model = Generator()
#     preds = model(x)
#
#     print(model)
#     print(preds.shape)
#
#
# if __name__ == "__main__":
#     test()