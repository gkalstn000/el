import sys
from collections import OrderedDict
from tqdm import tqdm

from options.train_options import TrainOptions
from trainers.trainer import Trainer
from util.iter_counter import IterationCounter
from util.visualizer import Visualizer

import torch
import data
import time


# parse options
opt = TrainOptions().parse()

# print options to help debugging
print(' '.join(sys.argv))

# load the dataset
dataloader = data.create_dataloader(opt)

opt.phase = 'test'
dataloader_val = data.create_dataloader(opt, valid = True)
opt.phase = 'train'

trainer = Trainer(opt)

iter_counter = IterationCounter(opt, len(dataloader))
visualizer = Visualizer(opt)



for epoch in iter_counter.training_epochs():
    iter_counter.record_epoch_start(epoch)
    for i, data_i in enumerate(tqdm(dataloader), start=iter_counter.epoch_iter):
        iter_counter.record_one_iteration()
        if i % opt.D_steps_per_G == 0:
            trainer.run_generator_one_step(data_i)

        # train discriminator
        trainer.run_discriminator_one_step(data_i)

        if iter_counter.needs_printing():
            losses = trainer.get_latest_losses()
            visualizer.print_current_errors(epoch, iter_counter.epoch_iter,
                                            losses, iter_counter.time_per_iter)
            visualizer.plot_current_errors(losses, iter_counter.total_steps_so_far)

        if iter_counter.needs_displaying():
            fake_image = trainer.get_latest_generated()
            b, g, c, h, w = data_i.size()

            visuals = OrderedDict([('train_1_original', data_i.view(b*g, c, h, w)),
                                   ('train_2_fake', fake_image),
                                   ('train_3_diff', torch.abs(data_i.view(b*g, c, h, w) - fake_image.detach().cpu()))
                                   ])
            visualizer.display_current_results(visuals, epoch, iter_counter.total_steps_so_far)

        if iter_counter.needs_saving():
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, iter_counter.total_steps_so_far))
            trainer.save('latest')
            iter_counter.record_current_iter()
        # break

    for i, data_i in tqdm(enumerate(dataloader_val), desc='Validation images generating') :
        b, g, c, h, w = data_i.size()

        fake_image = trainer.model(data_i, mode='inference')
        valid_losses = {}
        valid_losses['valid_L1'] = trainer.model.module.L1loss(fake_image, data_i.cuda().view(b*g, c, h, w)) * opt.lambda_rec
        visualizer.print_current_errors(epoch, iter_counter.epoch_iter,
                                        valid_losses, iter_counter.time_per_iter)
        visualizer.plot_current_errors(valid_losses, iter_counter.total_steps_so_far)

        visuals = OrderedDict([('valid_1_original', data_i.view(b*g, c, h, w)),
                               ('valid_2_fake', fake_image),
                               ('valid_3_diff', torch.abs(data_i.view(b*g, c, h, w) - fake_image.detach().cpu())),
                               ])

        visualizer.display_current_results(visuals, epoch, iter_counter.total_steps_so_far)
        break