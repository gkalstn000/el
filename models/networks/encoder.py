import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks import modules
from models.networks.base_network import BaseNetwork
from models.networks.normalization import get_nonspade_norm_layer
import math
import numpy as np


class ELEncoder(BaseNetwork):
    def __init__(self, opt):
        super(ELEncoder, self).__init__()
        self.opt = opt

        kw = 3
        nf = opt.nef
        input_nc = opt.image_nc

        self.layer1 = nn.Sequential(nn.Conv2d(input_nc, nf * 1, kw, stride=1, padding=1),
                                    nn.BatchNorm2d(nf * 1),
                                    nn.LeakyReLU(0.2, False),
                                    nn.Conv2d(nf * 1, nf * 1, kw, stride=1, padding=1),
                                    nn.BatchNorm2d(nf * 1),
                                    nn.LeakyReLU(0.2, False),
                                    nn.MaxPool2d(kernel_size=2)
                                    )

        self.layer2 = nn.Sequential(nn.Conv2d(nf * 1, nf * 2, kw, stride=1, padding=1),
                                    nn.BatchNorm2d(nf * 2),
                                    nn.LeakyReLU(0.2, False),
                                    nn.Conv2d(nf * 2, nf * 2, kw, stride=1, padding=1),
                                    nn.BatchNorm2d(nf * 2),
                                    nn.LeakyReLU(0.2, False),
                                    nn.MaxPool2d(kernel_size=2)
                                    )


        self.layer3 = nn.Sequential(nn.Conv2d(nf * 2, nf * 4, kw, stride=1, padding=1),
                                    nn.BatchNorm2d(nf * 4),
                                    nn.LeakyReLU(0.2, False),
                                    nn.Conv2d(nf * 4, nf * 4, kw, stride=1, padding=1),
                                    nn.BatchNorm2d(nf * 4),
                                    nn.LeakyReLU(0.2, False),
                                    nn.Conv2d(nf * 4, nf * 4, kw, stride=1, padding=1),
                                    nn.BatchNorm2d(nf * 4),
                                    nn.LeakyReLU(0.2, False),
                                    nn.Conv2d(nf * 4, nf * 4, kw, stride=1, padding=1),
                                    nn.BatchNorm2d(nf * 4),
                                    nn.LeakyReLU(0.2, False),
                                    nn.MaxPool2d(kernel_size=2)
                                    )

        self.layer4 = nn.Sequential(nn.Conv2d(nf * 4, nf * 8, kw, stride=1, padding=1),
                                    nn.BatchNorm2d(nf * 8),
                                    nn.LeakyReLU(0.2, False),
                                    nn.Conv2d(nf * 8, nf * 8, kw, stride=1, padding=1),
                                    nn.BatchNorm2d(nf * 8),
                                    nn.LeakyReLU(0.2, False),
                                    nn.Conv2d(nf * 8, nf * 8, kw, stride=1, padding=1),
                                    nn.BatchNorm2d(nf * 8),
                                    nn.LeakyReLU(0.2, False),
                                    nn.Conv2d(nf * 8, nf * 8, kw, stride=1, padding=1),
                                    nn.BatchNorm2d(nf * 8),
                                    nn.LeakyReLU(0.2, False),
                                    nn.MaxPool2d(kernel_size=2)
                                    )

        self.layer5 = nn.Sequential(nn.Conv2d(nf * 8, nf * 8, kw, stride=1, padding=1),
                                    nn.BatchNorm2d(nf * 8),
                                    nn.LeakyReLU(0.2, False),
                                    nn.Conv2d(nf * 8, nf * 8, kw, stride=1, padding=1),
                                    nn.BatchNorm2d(nf * 8),
                                    nn.LeakyReLU(0.2, False),
                                    nn.Conv2d(nf * 8, nf * 8, kw, stride=1, padding=1),
                                    nn.BatchNorm2d(nf * 8),
                                    nn.LeakyReLU(0.2, False),
                                    nn.Conv2d(nf * 8, nf * 8, kw, stride=1, padding=1),
                                    nn.BatchNorm2d(nf * 8),
                                    nn.LeakyReLU(0.2, False),
                                    nn.MaxPool2d(kernel_size=2)
                                    )
        dh, dw = 4, 4
        self.GAP = nn.AdaptiveAvgPool2d((dh, dw))
        self.fc_layer1 = nn.Sequential(nn.Linear(nf * 8 * dh * dw, 4096),
                                       nn.LeakyReLU(0.2, False),
                                       nn.BatchNorm1d(4096),
                                       nn.Dropout(0.2),
                                       )
        self.mu = nn.Sequential(nn.Linear(4096, self.opt.z_dim),
                                       nn.LeakyReLU(0.2, False),
                                       nn.BatchNorm1d(self.opt.z_dim),
                                       )
        self.logvar = nn.Sequential(nn.Linear(4096, self.opt.z_dim),
                                       nn.LeakyReLU(0.2, False),
                                       nn.BatchNorm1d(self.opt.z_dim),
                                       )

    def forward(self, x):

        x = self.layer1(x)  # 256x256 -> 128x128
        x = self.layer2(x)  # 128x128 -> 64x64
        x = self.layer3(x)  # 64x64 -> 32x32
        x = self.layer4(x)  # 32x32 -> 16x16
        x = self.layer5(x)  # 16x16 -> 8x8
        x = self.GAP(x) # 8x8 -> 4x4

        x = x.view(x.size(0), -1)
        x = self.fc_layer1(x)
        mu = self.mu(x)
        logvar = self.logvar(x)

        return mu, logvar
