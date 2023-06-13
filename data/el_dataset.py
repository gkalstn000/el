from data.base_dataset import BaseDataset, augment
from PIL import Image
import os

import pandas as pd
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import random
import numpy as np


class ELDataset(BaseDataset) :

    @staticmethod
    def modify_commandline_options(parser, is_train) :
        parser.set_defaults(load_size=(256, 512))
        parser.add_argument('--no_augment', action='store_true')
        # parser.set_defaults(image_nc=3)
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.data_mode = opt.data_mode
        self.dataroot = os.path.join(opt.dataroot, opt.data_mode)
        self.phase = opt.phase
        self.df = self.get_paths(opt)
        self.dataset_size = len(self.df)
        self.load_size = opt.load_size
        # PIL image -> torch.Tensor type
        transform_list = []
        transform_list.append(transforms.ToTensor()) # Channel (H, W, C) -> (1, H, W)
        transform_list.append(transforms.Normalize(0.5, 0.5)) # Normalize [0,255] -> [-1, 1]
        self.trans = transforms.Compose(transform_list)

    def get_paths(self, opt):
        '''
        :param opt: option
        :return: data_mode(first|second) 에 맞는 dataframe
        '''
        root = os.path.join(opt.dataroot, 'classification', self.phase)
        df = pd.read_csv(os.path.join(root, f'data_df_{self.data_mode}.csv'))
        return df

    def __getitem__(self, index):
        filename, label = self.df.iloc[index]
        label_str = 'fault' if label == 1 else 'non_fault'
        image_path = os.path.join(self.dataroot, label_str, filename)
        img = Image.open(image_path).convert("L")
        # Image Pre-processing
        img = self.crop_edge(img)
        if self.phase == 'train' and not self.opt.no_augment and random.random() > 0.5 : # 50% 확률로 Aug 할지 말지
            img = augment(img)
        img = F.resize(img, self.load_size)
        img_tensor = self.trans(img)

        input_dict = {'img_tensor' : img_tensor,
                      'label' : label,
                      'filename': filename}

        return input_dict
    # edge 여백 crop
    def crop_edge(self, img):
        img_array = np.array(img)
        if self.data_mode == 'first':
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


