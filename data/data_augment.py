from PIL import Image
import util.util as util
import os
import pandas as pd
import torch
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import random
from tqdm import tqdm


def split(img):
    widths = [296, 295, 295, 293, 294, 293, 296, 291, 292, 292, 286, 286, 284,  # left
              286, 286, 286, 292, 292, 291, 294, 292, 292, 292, 292, 293, 292]  # right
    space = [6, 7, 4, 8, 6, 6, 8, 8, 7, 8, 7, 7, 0,
             6, 7, 8, 7, 9, 7, 7, 8, 8, 8, 8, 8, 0]
    width, height = img.size
    cell_height = 597
    img_stacks = []
    for h in range(6):
        from_ = 0
        for i, (w, s) in enumerate(zip(widths, space)):
            to_ = from_ + w if from_ + w < width else width
            single_cell = img.crop([from_, h * cell_height, to_, (h + 1) * cell_height])
            from_ = to_ + s
            single_cell = F.resize(single_cell, load_size)
            img_stacks.append(trans(single_cell))
    return torch.cat(img_stacks, 0)

def reshape(img_tensor):
    nrow = 6
    ncol = 26
    h, w = img_tensor.shape[1:]

    reshaped = img_tensor.reshape(nrow, ncol, h, w)
    reshaped = reshaped.permute(0, 2, 1, 3)
    reshaped = reshaped.reshape(nrow*h, ncol*w)

    return reshaped
def get_concat_h( im1, im2):
    dst = Image.new('L', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst
def edge_corp( img):
    w, h = img.size
    left, top, right, bot = [59, 62, 60, 119]
    img = img.crop([left, top, w-right, h-bot])

    w_croped, h_croped = img.size
    img_left = img.crop([0, 0, 3878 , h_croped])
    img_right = img.crop([3927, 0, w_croped, h_croped])
    img = get_concat_h(img_left, img_right)

    return img

def generate_shuffled_indices(num_samples, num_to_select):
    # 모든 인덱스 생성
    indices = list(range(num_samples))
    sampled_indices = []
    # 인덱스 섞기
    for i in range(num_to_select) :
        random.seed(i)
        sampled_indices.append(random.sample(indices, len(indices)))

    return sampled_indices


root = '/datasets/msha/el'
img_root = os.path.join(root, 'fault')
df_path = os.path.join(root, 'classification/train/data_df.csv')
# Image options
load_size = (256, 256)
transform_list = []
transform_list.append(transforms.ToTensor())
transform_list.append(transforms.Normalize(0.5, 0.5))
trans = transforms.Compose(transform_list)
toPIL = transforms.ToPILImage()

df = pd.read_csv(df_path)

fault_list = df[df['label']==1].filename.to_list()

permute_list = generate_shuffled_indices(156, 200)

aug_dict = {'filename' : [],
            'label' : []}

for filename in fault_list :
    img_module = Image.open(os.path.join(img_root, filename))
    img_module = edge_corp(img_module)
    img_cells = split(img_module)
    for i, indices in enumerate(tqdm(permute_list)) :
        aug_filename = filename.replace('.jpg', '_aug_{:03d}.jpg'.format(i+1))
        img_cell_shuffle = img_cells[indices]
        img_module_shuffle = reshape(img_cell_shuffle)
        img_module_shuffle = toPIL((img_module_shuffle + 1) / 2)

        aug_dict['filename'].append(aug_filename)
        aug_dict['label'].append(1)

        img_module_shuffle.save(os.path.join(img_root, aug_filename))

aug_df = pd.DataFrame.from_dict(aug_dict)
new_df = pd.concat([df, aug_df], axis = 0).reset_index(drop=True)
new_df.to_csv(os.path.join(root, 'classification/train/data_df_aug.csv'), index=False)