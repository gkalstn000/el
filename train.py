import sys
from collections import OrderedDict
from tqdm import tqdm

import util.util as util
from options.train_options import TrainOptions
from trainers.trainer import Trainer
from util.iter_counter import IterationCounter
from util.visualizer import Visualizer
from models.networks.loss import Scores
import torch
import data
import time


# parse options
opt = TrainOptions().parse()

# print options to help debugging
print(' '.join(sys.argv))

# load the training dataset
dataloader = data.create_dataloader(opt)
# load the testing dataset
opt.phase = 'test'
dataloader_val = data.create_dataloader(opt, valid = True)
opt.phase = 'train'

trainer = Trainer(opt)

iter_counter = IterationCounter(opt, len(dataloader))
visualizer = Visualizer(opt)
get_score = Scores() # Acc, Precision, Recall, F1 score 계산함수

for epoch in iter_counter.training_epochs():
    iter_counter.record_epoch_start(epoch)
    train_true = []
    train_pred = []
    for i, data_i in enumerate(tqdm(dataloader), start=iter_counter.epoch_iter):
        iter_counter.record_one_iteration()
        trainer.run_encoder_one_step(data_i) # Encoder 학습 1 iteration
        # logit, label list에 저장
        logit = trainer.get_latest_logit()
        train_true.extend(data_i['label'].tolist())
        train_pred.extend(((logit > 0.5) * 1).squeeze().detach().cpu().tolist())

        # opt.print_freq 배수마다 학습 loss 출력 및 tensorboard에 plot
        if iter_counter.needs_printing():
            losses = trainer.get_latest_losses()
            visualizer.print_current_errors(epoch, iter_counter.epoch_iter,
                                            losses, iter_counter.time_per_iter)
            visualizer.plot_current_errors(losses, iter_counter.total_steps_so_far)
        # opt.display_freq 배수마다 logit 이 써진 image tensorboard에 plot
        if iter_counter.needs_displaying():
            img_tensors = util.write_text_to_img(data_i['img_tensor'], logit, data_i['label'])
            visuals = OrderedDict([('Image with logit', img_tensors),])
            visualizer.display_current_results(visuals, epoch, iter_counter.total_steps_so_far)
        # opt.save_latest_freq 배수마다 latest model 파라미터 저장
        if iter_counter.needs_saving():
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, iter_counter.total_steps_so_far))
            trainer.save('latest')
            iter_counter.record_current_iter()
        # break
    # Train score tensorboard에 plot
    train_scores = get_score(train_pred, train_true, 'train')
    visualizer.plot_current_errors(train_scores, iter_counter.total_steps_so_far)

    # Test 결과물 확인
    test_true = []
    test_pred = []
    for data_i in tqdm(dataloader_val) :
        # logit, label list에 저장
        logit = trainer.model(data_i, mode='inference')
        test_true.extend(data_i['label'].tolist())
        test_pred.extend(((logit > 0.5) * 1).squeeze().detach().cpu().tolist())
        # break
    # Test 결과 이미지 tensorboard에 plot
    img_tensors = util.write_text_to_img(data_i['img_tensor'], logit, data_i['label'])
    visuals = OrderedDict([('[Valid] Image with logit ', img_tensors)])
    visualizer.display_current_results(visuals, epoch, iter_counter.total_steps_so_far)
    # Test score tensorboard에 plot
    valid_scores = get_score(test_pred, test_true, 'test')
    visualizer.plot_current_errors(valid_scores, iter_counter.total_steps_so_far)

    # opt.save_epoch_freq 배수마다 Model save
    if epoch % opt.save_epoch_freq == 0 or \
       epoch == iter_counter.total_epochs:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, iter_counter.total_steps_so_far))
        trainer.save('latest')
        trainer.save(epoch)