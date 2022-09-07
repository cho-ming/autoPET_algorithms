import torch
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt

from .layers import BasicConv3d, FastSmoothSeNormConv3d, RESseNormConv3d, UpConv, ITN3D
from skimage import exposure, io, util

import torch.nn as nn
from torch.nn import functional as F
import torch
import numpy as np

class BaselineUNet(nn.Module):
    def __init__(self, in_channels, n_cls, n_filters):
        super(BaselineUNet, self).__init__()
        self.in_channels = in_channels
        self.n_cls = 1 if n_cls == 2 else n_cls
        self.n_filters = n_filters

        self.block_1_1_left = BasicConv3d(in_channels, n_filters, kernel_size=3, stride=1, padding=1)
        self.block_1_2_left = BasicConv3d(n_filters, n_filters, kernel_size=3, stride=1, padding=1)

        self.pool_1 = nn.MaxPool3d(kernel_size=2, stride=2)  # 64, 1/2
        self.block_2_1_left = BasicConv3d(n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_2_2_left = BasicConv3d(2 * n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)

        self.pool_2 = nn.MaxPool3d(kernel_size=2, stride=2)  # 128, 1/4
        self.block_3_1_left = BasicConv3d(2 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_3_2_left = BasicConv3d(4 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)

        self.pool_3 = nn.MaxPool3d(kernel_size=2, stride=2)  # 256, 1/8
        self.block_4_1_left = BasicConv3d(4 * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_4_2_left = BasicConv3d(8 * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)

        self.upconv_3 = nn.ConvTranspose3d(8 * n_filters, 4 * n_filters, kernel_size=3, stride=2, padding=1,
                                           output_padding=1)
        self.block_3_1_right = BasicConv3d((4 + 4) * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_3_2_right = BasicConv3d(4 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)

        self.upconv_2 = nn.ConvTranspose3d(4 * n_filters, 2 * n_filters, kernel_size=3, stride=2, padding=1,
                                           output_padding=1)
        self.block_2_1_right = BasicConv3d((2 + 2) * n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_2_2_right = BasicConv3d(2 * n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)

        self.upconv_1 = nn.ConvTranspose3d(2 * n_filters, n_filters, kernel_size=3, stride=2, padding=1,
                                           output_padding=1)
        self.block_1_1_right = BasicConv3d((1 + 1) * n_filters, n_filters, kernel_size=3, stride=1, padding=1)
        self.block_1_2_right = BasicConv3d(n_filters, n_filters, kernel_size=3, stride=1, padding=1)

        self.conv1x1 = nn.Conv3d(n_filters, self.n_cls, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        ds0 = self.block_1_2_left(self.block_1_1_left(x))
        ds1 = self.block_2_2_left(self.block_2_1_left(self.pool_1(ds0)))
        ds2 = self.block_3_2_left(self.block_3_1_left(self.pool_2(ds1)))
        x = self.block_4_2_left(self.block_4_1_left(self.pool_3(ds2)))
        x = self.block_3_2_right(self.block_3_1_right(torch.cat([self.upconv_3(x), ds2], 1)))
        x = self.block_2_2_right(self.block_2_1_right(torch.cat([self.upconv_2(x), ds1], 1)))
        x = self.block_1_2_right(self.block_1_1_right(torch.cat([self.upconv_1(x), ds0], 1)))

        x = self.conv1x1(x)

        if self.n_cls == 1:
            return torch.sigmoid(x)
        else:
            return F.softmax(x, dim=1)


class Hybrid_SENorm(nn.Module):
    def __init__(self, in_channels, n_cls, n_filters, reduction=2):
        super(Hybrid_SENorm, self).__init__()
        self.in_channels = in_channels
        self.n_cls = 1 if n_cls == 2 else n_cls
        self.n_filters = n_filters

        #PET
        self.block_1_1_left = RESseNormConv3d(in_channels, n_filters, reduction, kernel_size=7, stride=1, padding=3)
        self.block_1_2_left = RESseNormConv3d(n_filters, n_filters, reduction, kernel_size=3, stride=1, padding=1)
        self.pool_1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.block_2_1_left = RESseNormConv3d(n_filters, 2 * n_filters, reduction, kernel_size=3, stride=1, padding=1)
        self.block_2_2_left = RESseNormConv3d(2 * n_filters, 2 * n_filters, reduction, kernel_size=3, stride=1,padding=1)
        self.pool_2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.block_3_1_left = RESseNormConv3d(2 * n_filters, 4 * n_filters, reduction, kernel_size=3, stride=1,padding=1)
        self.block_3_2_left = RESseNormConv3d(4 * n_filters, 4 * n_filters, reduction, kernel_size=3, stride=1,padding=1)
        self.pool_3 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.block_4_1_left = RESseNormConv3d(4 * n_filters, 8 * n_filters, reduction, kernel_size=3, stride=1,padding=1)
        self.block_4_2_left = RESseNormConv3d(8 * n_filters, 8 * n_filters, reduction, kernel_size=3, stride=1,padding=1)
        self.pool_4 = nn.MaxPool3d(kernel_size=2, stride=2)



        # CT
        self.block_1_1_leftct = RESseNormConv3d(in_channels, n_filters, reduction, kernel_size=7, stride=1, padding=3)
        self.block_1_2_leftct = RESseNormConv3d(n_filters, n_filters, reduction, kernel_size=3, stride=1, padding=1)
        self.pool_1ct = nn.MaxPool3d(kernel_size=2, stride=2)

        self.block_2_1_leftct = RESseNormConv3d(n_filters, 2 * n_filters, reduction, kernel_size=3, stride=1, padding=1)
        self.block_2_2_leftct = RESseNormConv3d(2 * n_filters, 2 * n_filters, reduction, kernel_size=3, stride=1,padding=1)
        self.pool_2ct = nn.MaxPool3d(kernel_size=2, stride=2)

        self.block_3_1_leftct = RESseNormConv3d(2 * n_filters, 4 * n_filters, reduction, kernel_size=3, stride=1,padding=1)
        self.block_3_2_leftct = RESseNormConv3d(4 * n_filters, 4 * n_filters, reduction, kernel_size=3, stride=1,padding=1)
        self.pool_3ct = nn.MaxPool3d(kernel_size=2, stride=2)

        self.block_4_1_leftct = RESseNormConv3d(4 * n_filters, 8 * n_filters, reduction, kernel_size=3, stride=1,padding=1)
        self.block_4_2_leftct = RESseNormConv3d(8 * n_filters, 8 * n_filters, reduction, kernel_size=3, stride=1,padding=1)
        self.pool_4ct = nn.MaxPool3d(kernel_size=2, stride=2)


        self.colearn_1 = nn.Conv3d(2*n_filters, n_filters, kernel_size=(2,3,3),padding=1,dilation=(2,1,1))
        self.colearn_2 = nn.Conv3d(4 * n_filters, 2* n_filters, kernel_size=(2, 3, 3),padding=1,dilation=(2,1,1))
        self.colearn_3 = nn.Conv3d(8 * n_filters, 4 * n_filters, kernel_size=(2, 3, 3),padding=1,dilation=(2,1,1))
        self.colearn_4 = nn.Conv3d(16 * n_filters, 8 * n_filters, kernel_size=(2, 3, 3),padding=1,dilation=(2,1,1))

        #decoder
        # self.upconv_4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.block_4_1_right = FastSmoothSeNormConv3d((8 + 8 + 8) * n_filters, 8 * n_filters, reduction, kernel_size=3,stride=1, padding=1)
        self.block_4_2_right = FastSmoothSeNormConv3d(8 * n_filters, 4 * n_filters, reduction, kernel_size=3, stride=1,padding=1)

        self.upconv_3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.block_3_1_right = FastSmoothSeNormConv3d((4 + 4 + 4 + 4) * n_filters, 4 * n_filters, reduction, kernel_size=3,stride=1, padding=1)
        self.block_3_2_right = FastSmoothSeNormConv3d(4 * n_filters, 2 * n_filters, reduction, kernel_size=3, stride=1,padding=1)

        self.upconv_2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.block_2_1_right = FastSmoothSeNormConv3d((2 + 2 + 2 + 2) * n_filters, 2 * n_filters, reduction, kernel_size=3,stride=1, padding=1)
        self.block_2_2_right = FastSmoothSeNormConv3d(2 * n_filters,  n_filters, reduction, kernel_size=3, stride=1,padding=1)

        self.upconv_1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.block_1_1_right = FastSmoothSeNormConv3d((1 + 1 + 1 + 1) * n_filters, n_filters, reduction, kernel_size=3,stride=1, padding=1)
        self.block_1_2_right = FastSmoothSeNormConv3d( n_filters, n_filters, reduction, kernel_size=3, stride=1,padding=1)
        self.upconv_0 = nn.Upsample(scale_factor=2, mode='nearest')

        self.conv1x1 = nn.Conv3d(1 * n_filters, self.n_cls, kernel_size=1, stride=1, padding=0)

    def forward(self,x):
        pet = x[:, 0, :, :, :]
        pet = pet.unsqueeze(0)

        ct = x[:, 1, :, :, :]
        ct = ct.unsqueeze(0)

        pet1 = self.pool_1(self.block_1_2_left(self.block_1_1_left(pet)))
        pet2 = self.pool_2(self.block_2_2_left(self.block_2_1_left(pet1)))
        pet3 = self.pool_3(self.block_3_2_left(self.block_3_1_left(pet2)))
        pet4 = self.pool_4(self.block_4_2_left(self.block_4_1_left(pet3)))


        ct1 = self.pool_1ct(self.block_1_2_leftct(self.block_1_1_leftct(ct)))
        ct2 = self.pool_2ct(self.block_2_2_leftct(self.block_2_1_leftct(ct1)))
        ct3 = self.pool_3ct(self.block_3_2_leftct(self.block_3_1_leftct(ct2)))
        ct4 = self.pool_4ct(self.block_4_2_leftct(self.block_4_1_leftct(ct3)))


        ds1 = self.colearn_1(torch.concat([pet1,ct1], dim=1))
        ds2 = self.colearn_2(torch.concat([pet2, ct2], dim=1))
        ds3 = self.colearn_3(torch.concat([pet3, ct3], dim=1))
        ds4 = self.colearn_4(torch.concat([pet4, ct4], dim=1))


        x = self.upconv_3(self.block_4_2_right(self.block_4_1_right(torch.concat([ds4,pet4,ct4],dim=1))))
        x = self.upconv_2(self.block_3_2_right(self.block_3_1_right(torch.concat([x,ds3,pet3,ct3],dim=1))))
        x = self.upconv_1(self.block_2_2_right(self.block_2_1_right(torch.concat([x, ds2, pet2, ct2], dim=1))))
        x = self.upconv_0(self.block_1_2_right(self.block_1_1_right(torch.concat([x, ds1, pet1, ct1], dim=1))))

        x = self.conv1x1(x)

        return torch.sigmoid(x)

class Hybrid_SENorm2(nn.Module):
    def __init__(self, in_channels, n_cls, n_filters, reduction=2):
        super(Hybrid_SENorm2, self).__init__()
        self.in_channels = in_channels
        self.n_cls = 1 if n_cls == 2 else n_cls
        self.n_filters = n_filters

        #PET
        self.block_1_1_left = RESseNormConv3d(in_channels, n_filters, reduction, kernel_size=7, stride=1, padding=3)
        self.block_1_2_left = RESseNormConv3d(n_filters, n_filters, reduction, kernel_size=3, stride=1, padding=1)
        self.block_1_3_left = RESseNormConv3d(n_filters, n_filters, reduction, kernel_size=3, stride=1, padding=1)
        self.pool_1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.block_2_1_left = RESseNormConv3d(n_filters, 2 * n_filters, reduction, kernel_size=3, stride=1, padding=1)
        self.block_2_2_left = RESseNormConv3d(2 * n_filters, 2 * n_filters, reduction, kernel_size=3, stride=1,padding=1)
        self.block_2_3_left = RESseNormConv3d(2 * n_filters, 2 * n_filters, reduction, kernel_size=3, stride=1,
                                              padding=1)
        self.pool_2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.block_3_1_left = RESseNormConv3d(2 * n_filters, 4 * n_filters, reduction, kernel_size=3, stride=1,padding=1)
        self.block_3_2_left = RESseNormConv3d(4 * n_filters, 4 * n_filters, reduction, kernel_size=3, stride=1,padding=1)
        self.block_3_3_left = RESseNormConv3d(4 * n_filters, 4 * n_filters, reduction, kernel_size=3, stride=1,
                                              padding=1)
        self.pool_3 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.block_4_1_left = RESseNormConv3d(4 * n_filters, 8 * n_filters, reduction, kernel_size=3, stride=1,padding=1)
        self.block_4_2_left = RESseNormConv3d(8 * n_filters, 8 * n_filters, reduction, kernel_size=3, stride=1,padding=1)
        self.block_4_3_left = RESseNormConv3d(8 * n_filters, 8 * n_filters, reduction, kernel_size=3, stride=1,padding=1)


        self.pool_4 = nn.MaxPool3d(kernel_size=2, stride=2)



        # CT
        self.block_1_1_leftct = RESseNormConv3d(in_channels, n_filters, reduction, kernel_size=7, stride=1, padding=3)
        self.block_1_2_leftct = RESseNormConv3d(n_filters, n_filters, reduction, kernel_size=3, stride=1, padding=1)
        self.block_1_3_leftct = RESseNormConv3d(n_filters, n_filters, reduction, kernel_size=3, stride=1, padding=1)

        self.pool_1ct = nn.MaxPool3d(kernel_size=2, stride=2)

        self.block_2_1_leftct = RESseNormConv3d(n_filters, 2 * n_filters, reduction, kernel_size=3, stride=1, padding=1)
        self.block_2_2_leftct = RESseNormConv3d(2 * n_filters, 2 * n_filters, reduction, kernel_size=3, stride=1,padding=1)
        self.block_2_3_leftct = RESseNormConv3d(2 * n_filters, 2 * n_filters, reduction, kernel_size=3, stride=1,padding=1)

        self.pool_2ct = nn.MaxPool3d(kernel_size=2, stride=2)

        self.block_3_1_leftct = RESseNormConv3d(2 * n_filters, 4 * n_filters, reduction, kernel_size=3, stride=1,padding=1)
        self.block_3_2_leftct = RESseNormConv3d(4 * n_filters, 4 * n_filters, reduction, kernel_size=3, stride=1,padding=1)
        self.block_3_3_leftct = RESseNormConv3d(4 * n_filters, 4 * n_filters, reduction, kernel_size=3, stride=1,padding=1)

        self.pool_3ct = nn.MaxPool3d(kernel_size=2, stride=2)

        self.block_4_1_leftct = RESseNormConv3d(4 * n_filters, 8 * n_filters, reduction, kernel_size=3, stride=1,padding=1)
        self.block_4_2_leftct = RESseNormConv3d(8 * n_filters, 8 * n_filters, reduction, kernel_size=3, stride=1,padding=1)
        self.block_4_3_leftct = RESseNormConv3d(8 * n_filters, 8 * n_filters, reduction, kernel_size=3, stride=1,padding=1)

        self.pool_4ct = nn.MaxPool3d(kernel_size=2, stride=2)


        self.colearn_1 = nn.Conv3d(2*n_filters, n_filters, kernel_size=(2,3,3),padding=1,dilation=(2,1,1))
        self.colearn_2 = nn.Conv3d(4 * n_filters, 2* n_filters, kernel_size=(2, 3, 3),padding=1,dilation=(2,1,1))
        self.colearn_3 = nn.Conv3d(8 * n_filters, 4 * n_filters, kernel_size=(2, 3, 3),padding=1,dilation=(2,1,1))
        self.colearn_4 = nn.Conv3d(16 * n_filters, 8 * n_filters, kernel_size=(2, 3, 3),padding=1,dilation=(2,1,1))

        #decoder
        # self.upconv_4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.block_4_1_right = FastSmoothSeNormConv3d((8 + 8 + 8) * n_filters, 8 * n_filters, reduction, kernel_size=3,stride=1, padding=1)
        self.block_4_2_right = FastSmoothSeNormConv3d(8 * n_filters, 4 * n_filters, reduction, kernel_size=3, stride=1,padding=1)
        self.block_4_3_right = FastSmoothSeNormConv3d(4 * n_filters, 4 * n_filters, reduction, kernel_size=3, stride=1,
                                                      padding=1)

        self.upconv_3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.block_3_1_right = FastSmoothSeNormConv3d((4 + 4 + 4 + 4) * n_filters, 4 * n_filters, reduction, kernel_size=3,stride=1, padding=1)
        self.block_3_2_right = FastSmoothSeNormConv3d(4 * n_filters, 2 * n_filters, reduction, kernel_size=3, stride=1,padding=1)
        self.block_3_3_right = FastSmoothSeNormConv3d(2 * n_filters, 2 * n_filters, reduction, kernel_size=3, stride=1,padding=1)


        self.upconv_2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.block_2_1_right = FastSmoothSeNormConv3d((2 + 2 + 2 + 2) * n_filters, 2 * n_filters, reduction, kernel_size=3,stride=1, padding=1)
        self.block_2_2_right = FastSmoothSeNormConv3d(2 * n_filters,  n_filters, reduction, kernel_size=3, stride=1,padding=1)
        self.block_2_3_right = FastSmoothSeNormConv3d(n_filters,  n_filters, reduction, kernel_size=3, stride=1,padding=1)


        self.upconv_1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.block_1_1_right = FastSmoothSeNormConv3d((1 + 1 + 1 + 1) * n_filters, n_filters, reduction, kernel_size=3,stride=1, padding=1)
        self.block_1_2_right = FastSmoothSeNormConv3d( n_filters, n_filters, reduction, kernel_size=3, stride=1,padding=1)

        self.upconv_0 = nn.Upsample(scale_factor=2, mode='nearest')

        self.conv1x1 = nn.Conv3d(1 * n_filters, self.n_cls, kernel_size=1, stride=1, padding=0)

    def forward(self,x):
        pet = x[:, 0, :, :, :]
        pet = pet.unsqueeze(0)

        ct = x[:, 1, :, :, :]
        ct = ct.unsqueeze(0)

        pet1 = self.pool_1(self.block_1_3_left(self.block_1_2_left(self.block_1_1_left(pet))))
        pet2 = self.pool_2(self.block_2_3_left(self.block_2_2_left(self.block_2_1_left(pet1))))
        pet3 = self.pool_3(self.block_3_3_left(self.block_3_2_left(self.block_3_1_left(pet2))))
        pet4 = self.pool_4(self.block_4_3_left(self.block_4_2_left(self.block_4_1_left(pet3))))


        ct1 = self.pool_1ct(self.block_1_3_leftct(self.block_1_2_leftct(self.block_1_1_leftct(ct))))
        ct2 = self.pool_2ct(self.block_2_3_leftct(self.block_2_2_leftct(self.block_2_1_leftct(ct1))))
        ct3 = self.pool_3ct(self.block_3_3_leftct(self.block_3_2_leftct(self.block_3_1_leftct(ct2))))
        ct4 = self.pool_4ct(self.block_4_3_leftct(self.block_4_2_leftct(self.block_4_1_leftct(ct3))))


        ds1 = self.colearn_1(torch.concat([pet1,ct1], dim=1))
        ds2 = self.colearn_2(torch.concat([pet2, ct2], dim=1))
        ds3 = self.colearn_3(torch.concat([pet3, ct3], dim=1))
        ds4 = self.colearn_4(torch.concat([pet4, ct4], dim=1))


        x = self.upconv_3(self.block_4_3_right(self.block_4_2_right(self.block_4_1_right(torch.concat([ds4,pet4,ct4],dim=1)))))
        x = self.upconv_2(self.block_3_3_right(self.block_3_2_right(self.block_3_1_right(torch.concat([x,ds3,pet3,ct3],dim=1)))))
        x = self.upconv_1(self.block_2_3_right(self.block_2_2_right(self.block_2_1_right(torch.concat([x, ds2, pet2, ct2], dim=1)))))
        x = self.upconv_0(self.block_1_2_right(self.block_1_1_right(torch.concat([x, ds1, pet1, ct1], dim=1))))

        x = self.conv1x1(x)

        return torch.sigmoid(x)
