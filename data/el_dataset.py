from data.base_dataset import BaseDataset
from PIL import Image
import util.util as util
import os
import pandas as pd
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import torch

from tqdm import tqdm, trange
import numpy as np
import time


class ELDataset(BaseDataset) :

    @staticmethod
    def modify_commandline_options(parser, is_train) :
        parser.set_defaults(load_size=(256, 512))
        parser.set_defaults(old_size=(7780, 3600))
        # parser.set_defaults(image_nc=3)
        # # parser.add_argument(pose_nc=41)
        # parser.set_defaults(display_winsize=256)
        # parser.set_defaults(crop_size=256)
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.dataroot = opt.dataroot
        self.phase = opt.phase
        self.df = self.get_paths(opt)

        size = len(self.df)
        self.dataset_size = size

        if isinstance(opt.load_size, int):
            self.load_size = (opt.load_size, opt.load_size)
        else:
            self.load_size = opt.load_size

        transform_list=[]
        # transform_list.append(transforms.Resize(size=self.load_size))
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize(0.5,0.5))
        self.trans = transforms.Compose(transform_list)

    def get_paths(self, opt):
        root = os.path.join(opt.dataroot, 'classification', self.phase)
        df = pd.read_csv(os.path.join(root, 'data_df.csv'))
        return df

    def __getitem__(self, index):
        filename, label = self.df.iloc[index]
        label_str = 'fault' if label == 1 else 'non_fault'
        image_path = os.path.join(self.dataroot, label_str, filename)

        img = Image.open(image_path).convert("L")
        img = self.edge_corp(img)
        img = F.resize(img, self.load_size)
        img_tensor = self.trans(img)
        label = torch.nn.functional.one_hot(torch.tensor(label), 2)
        input_dict = {'img_tensor' : img_tensor,
                      'label' : label,
                      'filename': filename}

        return input_dict

    def edge_corp(self, img):
        w, h = img.size
        left, top, right, bot = [61, 66, 58, 123]
        img = img.crop([left, top, w-right, h-bot])
        w_crop, h_crop = img.size

        img_left = img.crop([0, 0, w_crop // 2 - (24 - 3), h_crop])
        img_right = img.crop([w_crop // 2 + (24 - 3), 0, w_crop, h_crop])

        img = self.get_concat_h(img_left, img_right)

        return img
    def postprocess(self, input_dict):
        return input_dict

    def __len__(self):
        return self.dataset_size

    def get_concat_h(self, im1, im2):
        dst = Image.new('L', (im1.width + im2.width, im1.height))
        dst.paste(im1, (0, 0))
        dst.paste(im2, (im1.width, 0))
        return dst


