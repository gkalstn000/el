from data.base_dataset import BaseDataset
from PIL import Image
import util.util as util
import os
import cv2

import pandas as pd
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import torch
import random
from tqdm import tqdm, trange
import numpy as np
import time


class ELDataset(BaseDataset) :

    @staticmethod
    def modify_commandline_options(parser, is_train) :
        parser.set_defaults(load_size=(256, 512))
        parser.add_argument('--no_augment', action='store_true')
        # parser.set_defaults(image_nc=3)
        # # parser.add_argument(pose_nc=41)
        # parser.set_defaults(display_winsize=256)
        # parser.set_defaults(crop_size=256)
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.data_mode = opt.data_mode
        self.dataroot = os.path.join(opt.dataroot, opt.data_mode)
        self.phase = opt.phase
        self.df = self.get_paths(opt)

        size = len(self.df)
        self.dataset_size = size

        if isinstance(opt.load_size, int):
            self.load_size = (opt.load_size, opt.load_size)
        else:
            self.load_size = opt.load_size

        transform_list = []
        # transform_list.append(transforms.Resize(size=self.load_size))
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize(0.5, 0.5))
        self.trans = transforms.Compose(transform_list)


    def get_paths(self, opt):
        root = os.path.join(opt.dataroot, 'classification', self.phase)
        df = pd.read_csv(os.path.join(root, f'data_df_{self.data_mode}.csv'))
        return df

    def __getitem__(self, index):
        filename, label = self.df.iloc[index]
        label_str = 'fault' if label == 1 else 'non_fault'
        image_path = os.path.join(self.dataroot, label_str, filename)

        img = Image.open(image_path).convert("L")
        img = self.crop_edge(img)
        if self.phase == 'train' and not self.opt.no_augment and random.random() > 0.5 :
            img = augment(img)
        img = F.resize(img, self.load_size)

        img_tensor = self.trans(img)

        input_dict = {'img_tensor' : img_tensor,
                      'label' : label,
                      'filename': filename}

        return input_dict


    def crop_edge(self, img):
        img_array = np.array(img)
        if self.data_mode == 'first':
            # edge 여백 crop
            img_array = img_array[:-58]
        else:
            left_start = [50, 55]
            right_start = [50, 3978]
            size = [3612, 3888]

            left_array = img_array[left_start[0] : left_start[0] + size[0], left_start[1] : left_start[1] + size[1]]
            right_array = img_array[right_start[0] : right_start[0] + size[0], right_start[1] : right_start[1] + size[1]]
            img_array = np.concatenate([left_array, right_array], 1)
        return Image.fromarray(img_array)

    def postprocess(self, input_dict):
        return input_dict

    def __len__(self):
        return self.dataset_size



def augment(img):
    # edge crop
    img_array = np.array(img)
    h, w = img_array.shape

    # 수직 수평 한번에 섞기
    if random.random() > 0.5 :
        dw = w // 4
        dh = h // 3

        grid = []
        for i in range(3) : # 수직
            for j in range(4) : # 수형
                grid.append(img_array[i * dh : (i + 1) * dh, j * dw : (j+1) * dw])
        random.shuffle(grid)
        output = []
        for i in range(3) :
            tmp = []
            for j in range(4) :
                tmp.append(grid[i * 4 + j])
            output.append(np.concatenate(tmp, 1))
        img_array = np.array(np.concatenate(output, 0))
    # 수직 수평 따로 섞기
    else :
        # 수직 섞기
        if random.random() > 0.5 :
            dw = w // 4
            grid = [img_array[:, i * dw : (i+1) * dw] for i in range(4)]
            random.shuffle(grid)
            img_array = np.concatenate(grid, 1)
        # 수형 섞기
        if random.random() > 0.5 :
            dh = h // 3
            grid = [img_array[i * dh: (i + 1) * dh, :] for i in range(3)]
            random.shuffle(grid)
            img_array = np.concatenate(grid, 0)

    return Image.fromarray(img_array)




# def augment(img):
#
#     w, h = img.size
#     if random.random() > 0.5 :
#         left = img.crop((0, 0, w // 2, h))
#         right = img.crop((w//2, 0, w, h))
#
#         new_img = Image.new("L", (w, h))
#         new_img.paste(right, (0, 0))
#         new_img.paste(left, (w//2, 0))
#
#         img = new_img
#
#     return img
#
# def get_outlier(x) :
#     mean = x.mean()
#     std = x.std()
#
#     z_score = (x - mean) / std
#
#     threshold = 1
#
#     outlier_index = np.abs(z_score) > threshold
#
#     return outlier_index
# def get_concat_h(im1, im2):
#     dst = Image.new('L', (im1.width + im2.width, im1.height))
#     dst.paste(im1, (0, 0))
#     dst.paste(im2, (im1.width, 0))
#     return dst
