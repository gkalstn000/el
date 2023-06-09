"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch.utils.data as data

class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        pass


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

