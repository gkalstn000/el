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
        parser.set_defaults(load_size=(256, 256))
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
        csv_filename = 'data_df_aug.csv' if self.phase == 'train' else 'data_df.csv'
        df = pd.read_csv(os.path.join(root, csv_filename))
        return df

    def __getitem__(self, index):
        filename, label = self.df.iloc[index]
        label_str = 'fault' if label == 1 else 'non_fault'
        image_path = os.path.join(self.dataroot, label_str, filename)

        img = Image.open(image_path).convert("L")
        if 'aug' not in filename :
            img = self.edge_corp(img)
            img_tensor = self.split(img)
            img_tensor = self.reconstruct(img_tensor)
        else :
            img = F.resize(img, (256, 512))
            img_tensor = self.trans(img)

        label = torch.nn.functional.one_hot(torch.tensor(label), 2)

        input_dict = {'img_tensor' : img_tensor,
                      'label' : label,
                      'filename': filename,
                      'img_original': self.trans(F.resize(img, (256, 512)))}

        return input_dict
    def reconstruct(self, img_tensor):
        nrow = 6
        ncol = 26
        toPIL = transforms.ToPILImage()
        c, h, w = img_tensor.size()

        reshaped = img_tensor.reshape(nrow, ncol, h, w)
        reshaped = reshaped.permute(0, 2, 1, 3)
        reshaped = reshaped.reshape(nrow*h, ncol*w)

        pil_img = toPIL((reshaped + 1) / 2)
        pil_img = F.resize(pil_img, (256, 512))


        return self.trans(pil_img)

    def edge_corp(self, img):
        w, h = img.size
        left, top, right, bot = [59, 62, 60, 119]
        img = img.crop([left, top, w-right, h-bot])

        w_croped, h_croped = img.size
        img_left = img.crop([0, 0, 3878 , h_croped])
        img_right = img.crop([3927, 0, w_croped, h_croped])
        img = self.get_concat_h(img_left, img_right)

        return img
    def split(self, img):
        widths = [296, 295, 295, 293, 294, 293, 296, 291, 292, 292, 286, 286, 284, # left
                  286, 286, 286, 292, 292, 291, 294, 292, 292, 292, 292, 293, 292] #right
        space = [6, 7, 4, 8, 6, 6, 8, 8, 7, 8, 7, 7, 0,
                 6, 7, 8, 7, 9, 7, 7, 8, 8, 8, 8, 8, 0]
        width, height = img.size
        cell_height = 597
        util.mkdirs('./results/tmp')
        img_stacks = []
        for h in range(6) :
            from_ = 0
            for i, (w, s) in enumerate(zip(widths, space)) :
                to_ = from_ + w if from_ + w < width else width
                single_cell = img.crop([from_, h * cell_height, to_, (h+1) * cell_height])
                from_ = to_ + s
                single_cell = F.resize(single_cell, self.load_size)
                img_stacks.append(self.trans(single_cell))
        return torch.cat(img_stacks, 0)

    def postprocess(self, input_dict):
        return input_dict

    def __len__(self):
        return self.dataset_size

    def get_concat_h(self, im1, im2):
        dst = Image.new('L', (im1.width + im2.width, im1.height))
        dst.paste(im1, (0, 0))
        dst.paste(im2, (im1.width, 0))
        return dst


