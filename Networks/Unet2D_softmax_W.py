import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.sparse import coo_matrix
import scipy.io as io
import scipy as sp

eps = torch.tensor(1e-9)
kerpar = torch.tensor(1)

class double_conv(nn.Module):
    """convolution => [BN] => ReLU) * 2, no downsample"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=False),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=False)
        )


    def forward(self, x):
        return self.double_conv(x)

class Convout(nn.Module):
    """out image using convolution and relu"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv1(x)

class down_conv(nn.Module):
    """stride convolution => [BN] => ReLU, downsample"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=False)
        )

    def forward(self, x):
        return self.down_conv(x)


class Up(nn.Module):
    """Upscaling using bilinear or deconv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            #self.up = nn.ConvTranspose3d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(negative_slope=0.01, inplace=False)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        x1 = self.relu(self.bn(self.conv(x1)))
        x1 = x1 + x2
        return x1

class BuildK(nn.Module):
    def __init__(self, kerpar, eps):
        super(BuildK, self).__init__()
        self.kerpar = kerpar
        self.eps = eps

    def forward(self, input1, input2):
        ## compute the weight
        input2 = input2.long()
        a, b, c = input1.shape[1], input1.shape[2], input1.shape[3]
        UU = input1.contiguous().view(a, b * c)
        UU = UU.t()
        J = input2.shape[0]
        #W = torch.zeros(input2.shape, requires_grad=True).double()
        D = torch.zeros(input2.shape, requires_grad=True).double()
        for i in range(input2.shape[1]):
            data_UU = (UU[0:J, :] - UU[input2[:, i], :])
            D[:, i] = -((torch.mean(torch.pow(data_UU,  2), 1) + self.eps) ** 0.5)
        W = torch.nn.functional.softmax(D, dim = 1)
        return W

class UNet(nn.Module):
    def __init__(self, in_channels, inter_channels, out_channels,  bilinear=True):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.inter_channels = inter_channels
        self.bilinear = bilinear
        self.relu = nn.ReLU()

        self.inc1 = double_conv(in_channels, inter_channels)
        self.down1 = down_conv(inter_channels, inter_channels)
        self.inc2 = double_conv(inter_channels, inter_channels*2)
        self.down2 = down_conv(inter_channels*2, inter_channels*2)
        self.inc3 = double_conv(inter_channels*2, inter_channels*4)
        self.down3 = down_conv(inter_channels*4, inter_channels*4)
        self.inc4 = double_conv(inter_channels*4, inter_channels*8)
        self.up1 = Up(inter_channels*8, inter_channels*4, bilinear)
        self.inc5 = double_conv(inter_channels*4, inter_channels*4)
        self.up2 = Up(inter_channels*4, inter_channels*2, bilinear)
        self.inc6 = double_conv(inter_channels*2, inter_channels*2)
        self.up3 = Up(inter_channels*2, inter_channels, bilinear)
        self.inc7 = double_conv(inter_channels, inter_channels)
        self.out = Convout(inter_channels, out_channels)
        self.K = BuildK(kerpar, eps)

    def forward(self, x, x_noise, N):
        high = x.shape[-1]
        x1 = self.inc1(x)
        x1_down = self.down1(x1)
        x2 = self.inc2(x1_down)
        x2_down = self.down2(x2)
        x3 = self.inc3(x2_down)
        x3_down = self.down3(x3)
        x4 = self.inc4(x3_down)

        x5 = self.up1(x4,x3)
        x5 = self.inc5(x5)
        x6 = self.up2(x5,x2)
        x6 = self.inc6(x6)
        x7 = self.up3(x6,x1)
        x7 = self.inc7(x7)
        res = self.out(x7)
        f = res
        f = f[:,:,0:high-1,0:high-1]
        W = self.K(f, N)
        W = W.to(torch.float32)

        a, b, c = x_noise.shape[1], x_noise.shape[2], x_noise.shape[3]
        UU = x_noise.view(a, b * c)
        UU = UU.t()
        W = W.cuda()
        out = torch.zeros(UU.shape, requires_grad=True).double()
        for i in range(UU.shape[1]):
            UU_r = UU[:,i]
            inter_r = W * UU_r[N.long()]
            out[:,i] = torch.sum(inter_r, axis=1)
        out = out.t()
        out = out.view(-1, a, b, c)
        out = out.float().cuda()

        return W, out



