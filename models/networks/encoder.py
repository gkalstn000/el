import torch.nn as nn
from models.networks.base_network import BaseNetwork

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
        dh, dw = 4, 8
        self.GAP = nn.AdaptiveAvgPool2d((4, 8))
        self.fc_layer1 = nn.Sequential(nn.Linear(nf * 8 * dh * dw, 4096),
                                       nn.LeakyReLU(0.2, False),
                                       nn.BatchNorm1d(4096),
                                       nn.Dropout(0.2),
                                       )
        self.fc_layer2 = nn.Sequential(nn.Linear(4096, 2048),
                                       nn.LeakyReLU(0.2, False),
                                       nn.BatchNorm1d(2048),
                                       nn.Dropout(0.2),
                                       )
        self.final_layer = nn.Linear(2048, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        x = self.layer1(x)  # 256x512 -> 128x256
        x = self.layer2(x)  # 128x256 -> 64x128
        x = self.layer3(x)  # 64x128 -> 32x64
        x = self.layer4(x)  # 32x64 -> 16x32
        x = self.layer5(x)  # 16x32 -> 8x16
        x = self.GAP(x)

        x = x.view(x.size(0), -1)
        x = self.fc_layer1(x)
        x = self.fc_layer2(x)
        x = self.final_layer(x)
        x = self.sigmoid(x)

        return x

