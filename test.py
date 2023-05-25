import data
from options.test_options import TestOptions
import models
import torch
import torchvision.transforms as T
from tqdm import tqdm
from util import html, util
import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
from collections import defaultdict

opt = TestOptions().parse()
dataloader = data.create_dataloader(opt)



trans = T.ToPILImage()
result_path = os.path.join(opt.results_dir, opt.id)
util.mkdirs(result_path)

listdir = os.listdir(os.path.join(opt.checkpoints_dir, opt.id))
score_df_list = []

preds = defaultdict(list)
trues = defaultdict(list)

model_dict = {}
for pre_train in listdir:
    if 'net_E' not in pre_train: continue
    epoch = pre_train.split('_')[0]
    opt.which_epoch = epoch
    util.mkdirs(os.path.join(result_path, f'vis_{epoch}'))

    model = models.create_model(opt)
    model.eval()
    model_dict[epoch] = model


for i, data_i in enumerate(tqdm(dataloader)) :
    # test

    for epoch, model in model_dict.items() :

        logit = model(data_i, mode='inference')
        target = data_i['label']

        _, pred = logit.max(1)
        _, true = target.max(1)

        preds[f'{epoch}'] += pred.cpu().tolist()
        trues[f'{epoch}'] += true.cpu().tolist()

        img_tensors = util.write_text_to_img(data_i['img_tensor'], logit, data_i['label'])
        img_tensors = (img_tensors + 1) / 2
        filename = data_i['filename']
        for k in range(img_tensors.shape[0]) :
            el_image = trans(img_tensors[k].cpu())
            el_image.save(os.path.join(result_path, f'vis_{epoch}' , filename[k]))


for (epoch, pred), (_, true) in zip(preds.items(), trues.items()) :


    pred_vector = np.array(pred)
    true_vector = np.array(true)

    # 이진 분류 평가 지표 계산
    tp = np.sum((pred_vector == 1) & (true_vector == 1))
    tn = np.sum((pred_vector == 0) & (true_vector == 0))
    fp = np.sum((pred_vector == 1) & (true_vector == 0))
    fn = np.sum((pred_vector == 0) & (true_vector == 1))

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)

    cm = confusion_matrix(true_vector, pred_vector)

    classes = ['Negative', 'Positive']
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.savefig(os.path.join(result_path, f'confusion_matrix_{epoch}.png'))
    plt.close()

    df_score = pd.DataFrame({f'Score_{epoch}' : [round(accuracy, 3), round(precision, 3), round(recall, 3), round(f1, 3)]})
    score_df_list.append(df_score)

df = pd.concat(score_df_list, axis=1)
df.index = ['Acc', 'Precision', 'Recall', 'F1']
df.to_csv(os.path.join(result_path,f'score.csv'))