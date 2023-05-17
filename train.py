import sys
from collections import OrderedDict
from tqdm import tqdm

from options.train_options import TrainOptions

import torch
import data
import time


# parse options
opt = TrainOptions().parse()

# print options to help debugging
print(' '.join(sys.argv))

# load the dataset
dataloader = data.create_dataloader(opt)


