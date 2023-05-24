import sys
from collections import OrderedDict
from tqdm import tqdm
import numpy as np

import util.util as util
from options.train_options import TrainOptions
from trainers.trainer import Trainer
from util.iter_counter import IterationCounter
from util.visualizer import Visualizer
from models.networks.loss import Accuracy
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
cal_acc = Accuracy()

for epoch in iter_counter.training_epochs():
    iter_counter.record_epoch_start(epoch)
    for i, data_i in enumerate(tqdm(dataloader), start=iter_counter.epoch_iter):
        iter_counter.record_one_iteration()
        trainer.run_encoder_one_step(data_i)


        if iter_counter.needs_printing():
            losses = trainer.get_latest_losses()
            logit = trainer.get_latest_logit()
            acc = cal_acc(logit, data_i['label'])
            losses['Acc'] = acc
            visualizer.print_current_errors(epoch, iter_counter.epoch_iter,
                                            losses, iter_counter.time_per_iter)
            visualizer.plot_current_errors(losses, iter_counter.total_steps_so_far)

        if iter_counter.needs_displaying():
            logit = trainer.get_latest_logit()
            img_tensors = util.write_text_to_img(data_i['img_tensor'], logit, data_i['label'])

            visuals = OrderedDict([('Image with logit', img_tensors),
                                   ])
            visualizer.display_current_results(visuals, epoch, iter_counter.total_steps_so_far)

        if iter_counter.needs_saving():
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, iter_counter.total_steps_so_far))
            trainer.save('latest')
            iter_counter.record_current_iter()
        # break

    preds = []
    trues = []
    for i, data_i in enumerate(tqdm(dataloader_val, desc='Validation images generating')) :
        logit = trainer.model(data_i, mode='inference')
        target = data_i['label']
        _, pred = logit.max(1)
        _, true = target.max(1)

        preds += pred.cpu().tolist()
        trues += true.cpu().tolist()

    pred_vector = torch.Tensor(preds)
    true_vector = torch.Tensor(trues)

    tp = torch.sum((pred_vector == 1) & (true_vector == 1))
    tn = torch.sum((pred_vector == 0) & (true_vector == 0))
    fp = torch.sum((pred_vector == 1) & (true_vector == 0))
    fn = torch.sum((pred_vector == 0) & (true_vector == 1))

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)

    valid_losses = {}
    valid_losses['valid_Acc'] = accuracy
    valid_losses['valid_precision'] = precision
    valid_losses['valid_recall'] = recall
    valid_losses['valid_f1'] = 2 * (precision * recall) / (precision + recall)

    visualizer.plot_current_errors(valid_losses, iter_counter.total_steps_so_far)

    if epoch % opt.save_epoch_freq == 0 or \
       epoch == iter_counter.total_epochs:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, iter_counter.total_steps_so_far))
        trainer.save('latest')
        trainer.save(epoch)