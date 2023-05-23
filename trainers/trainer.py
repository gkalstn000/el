import models
import torch
from models.sync_batchnorm import DataParallelWithCallback



class Trainer() :
    def __init__(self, opt):
        super(Trainer, self).__init__()
        self.opt = opt
        self.model = models.create_model(opt)
        self.model = DataParallelWithCallback(self.model, device_ids = opt.gpu_ids)

        self.generated = None
        if opt.isTrain :
            self.optimizer_E = self.model.module.create_optimizers(opt)
            self.old_lr = opt.lr


    def run_encoder_one_step(self, data):
        self.optimizer_E.zero_grad()
        e_losses, logit = self.model(data, mode='encoder')
        e_loss = sum(e_losses.values()).mean()
        e_loss.backward()
        self.optimizer_E.step()
        self.e_losses = e_losses
        self.latest_logit = logit

    def get_latest_losses(self):
        return {**self.e_losses}

    def get_latest_logit(self):
        return self.latest_logit
    def update_learning_rate(self, epoch):
        self.update_learning_rate(epoch)

    def save(self, epoch):
        self.model.module.save(epoch)

    ##################################################################
    # Helper functions
    ##################################################################

    def update_learning_rate(self, epoch):
        if epoch > self.opt.niter:
            lrd = self.opt.lr / self.opt.niter_decay
            new_lr = self.old_lr - lrd
        else:
            new_lr = self.old_lr

        if new_lr != self.old_lr:
            new_lr_E = new_lr

            for param_group in self.optimizer_E.param_groups:
                param_group['lr'] = new_lr_E
            print('update learning rate: %f -> %f' % (self.old_lr, new_lr))
            self.old_lr = new_lr
