import data
from options.test_options import TestOptions
import models
import torch
import torchvision.transforms as T
from tqdm import tqdm
from util import html, util
import os
from PIL import Image
import pandas as pd

def reconstruct(img_tensor):
    nrow = 6
    ncol = 26
    b, c, h, w = img_tensor.size()

    reshaped = img_tensor.reshape(nrow, ncol, h, w)
    reshaped = reshaped.permute(0, 2, 1, 3)
    reshaped = reshaped.reshape(nrow * h, ncol * w)

    return reshaped

def get_concat_v(im1, im2):
    dst = Image.new('L', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst

opt = TestOptions().parse()
dataloader = data.create_dataloader(opt)

trans = T.ToPILImage()
result_path = os.path.join(opt.results_dir, opt.id)
util.mkdirs(result_path)
vis_result_path = os.path.join(result_path, 'vis')
util.mkdirs(vis_result_path)


model = models.create_model(opt)
model.eval()

dis_dict = {'filename' : [],
            'label' : [],
            'distance L2' : [],
            'distance L1' : []}

L1loss = torch.nn.L1Loss()
L2loss = torch.nn.MSELoss()
toPIL = T.ToPILImage()

for i, data_i in enumerate(tqdm(dataloader)) :
    fake_image = model(data_i, mode='inference')
    true_image = data_i['img_tensor']

    L1 = L1loss(fake_image, true_image)
    L2 = L2loss(fake_image, true_image)

    dis_dict['filename'].append(data_i['filename'][0])
    dis_dict['label'].append(data_i['label'].int().item())
    dis_dict['distance L1'].append(round(L1.cpu().item(), 3))
    dis_dict['distance L2'].append(round(L2.cpu().item(), 3))

    diff_image = torch.abs(true_image - fake_image)

    fake_image = reconstruct(fake_image)
    true_image = reconstruct(true_image)
    diff_image = reconstruct(diff_image)

    fake_image = toPIL((fake_image + 1) / 2)
    true_image = toPIL((true_image + 1) / 2)
    diff_image = toPIL(diff_image)

    img = get_concat_v(true_image, diff_image)
    img = get_concat_v(img, fake_image)

    # img.save(os.path.join(vis_result_path, data_i['filename'][0]))



df = pd.DataFrame.from_dict(dis_dict)
fault = df[df['label'] == 1]['distance L1']
non_fault = df[df['label'] == 0]['distance L1']

df.to_csv(os.path.join(result_path, 'dist.csv'))

