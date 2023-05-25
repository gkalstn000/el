"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torch.nn as nn
from models.networks.base_network import BaseNetwork
import torch.nn.functional as F
from models.networks.normalization import get_nonspade_norm_layer

class BaseGenerator(BaseNetwork) :
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--activation', type=str, default='LeakyReLU', help='type of activation function')
        parser.add_argument('--z_dim', type=int, default=128, help="dimension of the latent z vector")
        parser.set_defaults(img_f=512)
        return parser
    def __init__(self):
        super(BaseGenerator, self).__init__()
    def compute_latent_vector_size(self, opt):
        if opt.num_upsampling_layers == 'normal':
            num_up_layers = 5
        elif opt.num_upsampling_layers == 'more':
            num_up_layers = 6
        elif opt.num_upsampling_layers == 'most':
            num_up_layers = 7
        else:
            raise ValueError('opt.num_upsampling_layers [%s] not recognized' %
                             opt.num_upsampling_layers)

        sw = opt.crop_size // (2**num_up_layers)
        sh = round(sw / (opt.load_size[1] / opt.load_size[0]))

        return sw, sh
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std) + mu


class ELGenerator(BaseGenerator) :

    def __init__(self, opt):
        super(ELGenerator, self).__init__()
        self.opt = opt
        nf = opt.ngf
        norm_layer = get_nonspade_norm_layer(opt, opt.norm_E)

        self.sw = self.sh = 8

        self.fc = nn.Linear(opt.z_dim, 16 * nf * self.sw * self.sh)

        self.head_0 = norm_layer(nn.Conv2d(16 * nf, 16 * nf, 3, stride=1, padding=1))

        self.G_middle_0 = norm_layer(nn.Conv2d(16 * nf, 16 * nf, 3, stride=1, padding=1))

        self.up_0 = norm_layer(nn.Conv2d(16 * nf, 8 * nf, 3, stride=1, padding=1))
        self.up_1 = norm_layer(nn.Conv2d(8 * nf, 4 * nf, 3, stride=1, padding=1))
        self.up_2 = norm_layer(nn.Conv2d(4 * nf, 2 * nf, 3, stride=1, padding=1))



        final_nc = 2 * nf
        self.conv_img = nn.Conv2d(final_nc, 1, 3, padding=1)

        self.up = nn.Upsample(scale_factor=2)


    def forward(self, texture_param):
        mu, var = texture_param
        z = self.reparameterize(mu, var)

        x = self.fc(z)
        x = x.view(-1, 16 * self.opt.ngf, self.sh, self.sw) # 8x8

        x = self.head_0(x)
        x = self.up(x) # 16x16

        x = self.G_middle_0(x)

        x = self.up(x) # 32x32
        x = self.up_0(x)
        x = self.up(x) # 64x64
        x = self.up_1(x)
        x = self.up(x) # 64x64
        x = self.up_2(x)

        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = F.tanh(x)

        return x


