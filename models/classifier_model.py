import torch
import torch.nn as nn

import models.networks as networks
import util.util as util
from models.networks import loss

class ClassifierModel(nn.Module) :
    @staticmethod
    def modify_commandline_options(parser, is_train):
        networks.modify_commandline_options(parser, is_train)
        return parser
    def __init__(self, opt):
        super(ClassifierModel, self).__init__()
        self.opt = opt
        self.netE = self.initialize_networks(opt)
        # set loss functions
        if opt.isTrain:
            self.CELoss = torch.nn.CrossEntropyLoss()

    def forward(self, data, mode):
        img_tensor, label = self.preprocess_input(data)
        if mode == 'encoder':
            e_loss, logit = self.compute_encoder_loss(img_tensor, label)
            return e_loss, logit

        elif mode == 'inference':
            self.netE.eval()
            with torch.no_grad():
                logit = self.netE(img_tensor)
            self.netE.train()
            return logit
    def create_optimizers(self, opt):
        E_params = list(self.netE.parameters())

        beta1, beta2 = opt.beta1, opt.beta2
        E_lr = opt.lr

        optimizer_E = torch.optim.Adam(E_params, lr=E_lr, betas=(beta1, beta2))

        return optimizer_E

    def save(self, epoch):
        util.save_network(self.netE, 'E', epoch, self.opt)

    def initialize_networks(self, opt):
        netE = networks.define_E(opt)
        if not opt.isTrain or opt.continue_train:
            netE = util.load_network(netE, 'E', opt.which_epoch, opt)

        return netE
    def preprocess_input(self, data):
        if self.use_gpu():
            data['img_tensor'] = data['img_tensor'].float().cuda()
            data['label'] = data['label'].float().cuda()

        return data['img_tensor'], data['label']
    def compute_encoder_loss(self, img_tensor, label):
        self.netE.train()
        E_losses = {}

        logit = self.netE(img_tensor)
        E_losses['CrossEntropy'] = self.CELoss(logit.squeeze(), label) * self.opt.lambda_ce
        return E_losses, logit

    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0
