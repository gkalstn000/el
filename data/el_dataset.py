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
        parser.set_defaults(load_size=(128, 128))
        parser.set_defaults(old_size=(7780, 3600))
        # parser.set_defaults(image_nc=3)
        # # parser.add_argument(pose_nc=41)
        # parser.set_defaults(display_winsize=256)
        # parser.set_defaults(crop_size=256)
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.phase = opt.phase
        self.df, self.fault_name_list, self.nonfault_name_list = self.get_paths(opt)
        size = len(self.nonfault_name_list) if self.phase == 'train' else len(self.fault_name_list)
        self.dataset_size = size

        if isinstance(opt.load_size, int):
            self.load_size = (opt.load_size, opt.load_size)
        else:
            self.load_size = opt.load_size

        transform_list=[]
        # transform_list.append(transforms.Resize(size=self.load_size))
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5)))
        self.trans = transforms.Compose(transform_list)

    def get_paths(self, opt):
        root = opt.dataroot
        df = pd.read_csv(os.path.join(root, 'df_label.csv'))

        self.fault_image_dir = os.path.join(root, f'fault')
        self.nonfault_image_dir = os.path.join(root, f'non_fault')
        fault_name_list = os.listdir(self.fault_image_dir)
        nonfault_name_list = os.listdir(self.nonfault_image_dir)

        return df, fault_name_list, nonfault_name_list



    def __getitem__(self, index):
        if self.phase == 'train' :
            image_name = self.nonfault_name_list[index]
            image_path = os.path.join(self.nonfault_image_dir, image_name)
        else :
            image_name = self.fault_name_list[index]
            image_path = os.path.join(self.fault_image_dir, image_name)

        img = Image.open(image_path)
        img = self.edge_corp(img)
        # 테두리 crop

        w, h = img.size
        dw, dh = w / 26, h / 6
        cells = []
        for i in range(6) :
            for j in range(26) :
                cell = img.crop([j * dw, i * dh, (j+1) * dw, (i+1) * dh])
                cell = F.resize(cell, self.load_size)
                cell_tensor = self.trans(cell)
                cells.append(cell_tensor)
        cells = torch.stack(cells)

        return cells

    def edge_corp(self, img):
        w, h = img.size
        left, top, right, bot = [61, 66, 58, 123]
        img = img.crop([left, top, w-right, h-bot])
        w_crop, h_crop = img.size

        img_left = img.crop([0, 0, w_crop // 2 - 25, h_crop])
        img_right = img.crop([w_crop // 2 + 25, 0, w_crop, h_crop])

        img = self.get_concat_h(img_left, img_right)

        return img
    def postprocess(self, input_dict):
        return input_dict

    def __len__(self):
        return self.dataset_size

    def get_concat_h(self, im1, im2):
        dst = Image.new('RGB', (im1.width + im2.width, im1.height))
        dst.paste(im1, (0, 0))
        dst.paste(im2, (im1.width, 0))
        return dst


