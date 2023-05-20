import torch
import torch.nn as nn

import models.networks as networks
import util.util as util
from models.networks import loss

class AnoGANModel(nn.Module) :
    @staticmethod
    def modify_commandline_options(parser, is_train):
        networks.modify_commandline_options(parser, is_train)
        return parser
    def __init__(self, opt):
        super(AnoGANModel, self).__init__()
        self.opt = opt
        self.FloatTensor = torch.cuda.FloatTensor if self.use_gpu() \
            else torch.FloatTensor
        self.ByteTensor = torch.cuda.ByteTensor if self.use_gpu() \
            else torch.ByteTensor

        self.netG, self.netE, self.netD = self.initialize_networks(opt)

        # set loss functions
        if opt.isTrain:
            self.GANloss = loss.GANLoss(opt.gan_mode, tensor=self.FloatTensor, opt=self.opt).cuda()
            self.L1loss = torch.nn.L1Loss()
            self.Vggloss = loss.VGGLoss()
            self.L2loss = torch.nn.MSELoss()
            self.KLDLoss = loss.KLDLoss()


    def forward(self, data, mode):
        el_tensors = self.preprocess_input(data)
        if mode == 'generator':
            g_loss, fake_t = self.compute_generator_loss(el_tensors)
            return g_loss, fake_t
        elif mode == 'discriminator':
            d_loss = self.compute_discriminator_loss(el_tensors)
            return d_loss
        elif mode == 'inference' :
            self.netG.eval()
            self.netE.eval()
            with torch.no_grad():
                mu, logvar, _ = self.encoding(el_tensors, loss=False)
                texture_information = [mu, logvar]
                fake_image_t = self.generate_fake(texture_information)
            return fake_image_t
    def create_optimizers(self, opt):
        G_params = list(self.netG.parameters()) + list(self.netE.parameters())
        D_params = list(self.netD.parameters())

        beta1, beta2 = opt.beta1, opt.beta2
        if opt.no_TTUR:
            G_lr, D_lr = opt.lr, opt.lr
        else:
            G_lr, D_lr = opt.lr / 2, opt.lr * 2

        optimizer_G = torch.optim.Adam(G_params, lr=G_lr, betas=(beta1, beta2))
        optimizer_D = torch.optim.Adam(D_params, lr=D_lr, betas=(beta1, beta2))

        return optimizer_G, optimizer_D

    def save(self, epoch):
        util.save_network(self.netG, 'G', epoch, self.opt)
        util.save_network(self.netD, 'D', epoch, self.opt)
        util.save_network(self.netE, 'E', epoch, self.opt)

    def initialize_networks(self, opt):
        netG = networks.define_G(opt)
        netE = networks.define_E(opt)
        netD = networks.define_D(opt) if opt.isTrain else None

        if not opt.isTrain or opt.continue_train:
            netG = util.load_network(netG, 'G', opt.which_epoch, opt)
            netE = util.load_network(netE, 'E', opt.which_epoch, opt)
            if opt.isTrain:
                netD = util.load_network(netD, 'D', opt.which_epoch, opt)

        return netG, netE, netD
    def preprocess_input(self, data):
        if self.use_gpu():
            data = data.float().cuda()
            b, g, c, h, w = data.size()
            h, w = self.opt.load_size
            data = data.view(b * g, c, h, w)

        return data
    def feature_matching(self, pred_fake, pred_real):
        num_D = len(pred_fake)
        GAN_Feat_loss = self.FloatTensor(1).fill_(0)
        for i in range(num_D):  # for each discriminator
            # last output is the final prediction, so we exclude it
            num_intermediate_outputs = len(pred_fake[i]) - 1
            for j in range(num_intermediate_outputs):  # for each layer output
                unweighted_loss = self.L1loss(
                    pred_fake[i][j], pred_real[i][j].detach())
                GAN_Feat_loss += unweighted_loss * self.opt.lambda_feat / num_D
        return GAN_Feat_loss
    def compute_generator_loss(self, image):
        self.netG.train()
        self.netE.train()
        self.netD.train()
        G_losses = {}

        mu, logvar, kld_loss = self.encoding(image)
        texture_information = [mu, logvar]

        fake_image = self.generate_fake(texture_information)

        pred_fake, pred_real = self.discriminate(fake_image, image)

        G_losses['GAN'] = self.GANloss(pred_fake, True, for_discriminator=False)
        G_losses['VGG_loss'] =  self.Vggloss(fake_image, image)  * self.opt.lambda_vgg
        G_losses['KLD_loss'] = kld_loss
        G_losses['L1_loss'] = self.L1loss(fake_image, image)  * self.opt.lambda_rec

        if not self.opt.no_ganFeat_loss:
            f_matching = self.feature_matching(pred_fake, pred_real)
            G_losses['GAN_Feat'] = f_matching

        return G_losses, fake_image

    def discriminate(self, fake, real):

        fake_and_real = torch.cat([fake, real], dim=0)

        discriminator_out = self.netD(fake_and_real)
        pred_fake, pred_real = self.divide_pred(discriminator_out)

        return pred_fake, pred_real
    def compute_discriminator_loss(self, image):
        self.netG.train()
        self.netE.train()
        self.netD.train()
        D_losses = {}
        with torch.no_grad():
            mu, logvar, _ = self.encoding(image, loss=False)
            texture_information = [mu, logvar]

            fake_image = self.generate_fake(texture_information)
            fake_image = fake_image.detach()
            fake_image.requires_grad_()

        pred_fake, pred_real = self.discriminate(fake_image, image)

        D_losses['D_fake'] = self.GANloss(pred_fake, False, for_discriminator=True)
        D_losses['D_real'] = self.GANloss(pred_real, True, for_discriminator=True)

        return D_losses
    def encoding(self, image, loss = True):
        mu, logvar = self.netE(image)
        kld_loss = None
        if loss :
            kld_loss = self.KLDLoss(mu, logvar) * self.opt.lambda_kld

        return mu, logvar, kld_loss
    def generate_fake(self, texture_information):
        fake_image_t = self.netG(texture_information)

        return fake_image_t
    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0

    def divide_pred(self, pred):
        # the prediction contains the intermediate outputs of multiscale GAN,
        # so it's usually a list
        if type(pred) == list:
            fake = []
            real = []
            for p in pred:
                fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
                real.append([tensor[tensor.size(0) // 2:] for tensor in p])
        else:
            fake = pred[:pred.size(0) // 2]
            real = pred[pred.size(0) // 2:]

        return fake, real