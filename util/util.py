"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import torch
from torchvision.utils import make_grid
import torchvision.transforms as transforms

import numpy as np
import os

from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

# ============== I/O 관련 함수들 ==============
def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# model parameter I/O
def load_network(net, label, epoch, opt):
    save_filename = '%s_net_%s.pth' % (epoch, label)
    save_dir = os.path.join(opt.checkpoints_dir, opt.id)
    save_path = os.path.join(save_dir, save_filename)
    weights = torch.load(save_path)
    net.load_state_dict(weights)
    return net
def save_network(net, label, epoch, opt):
    save_filename = '%s_net_%s.pth' % (epoch, label)
    save_path = os.path.join(opt.checkpoints_dir, opt.id, save_filename)
    torch.save(net.cpu().state_dict(), save_path)
    if len(opt.gpu_ids) and torch.cuda.is_available():
        net.cuda()
# model parameter I/O

def save_image(image_numpy, image_path, create_dir=False):
    if create_dir:
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
    if len(image_numpy.shape) == 2:
        image_numpy = np.expand_dims(image_numpy, axis=2)
    if image_numpy.shape[2] == 1:
        image_numpy = np.repeat(image_numpy, 3, 2)
    if image_numpy.shape[0] == 3 :
        image_numpy = np.transpose(image_numpy, (1, 2, 0))
    image_pil = Image.fromarray(image_numpy)#.resize((176, 256))

    # save image
    image_pil.save(image_path)
# ============== I/O 관련 함수들 ==============

# ============== 이미지 처리 관련 함수 ==============
def get_concat_h(imgs):
    '''
    PIL image list를 입력으로 받아 수평으로 concat 된 이미지 return 하는 함수
    :param imgs: PIL type image list
    :return: PIL type image
    '''
    width, height = imgs[0].size
    dst = Image.new('RGB', (width * len(imgs), height))
    for i, img in enumerate(imgs) :
        dst.paste(img, (i*width, 0))
    return dst

def get_concat_v(im1, im2):
    '''
    im1 이미지 아래에 im2 붙이는 함수
    :param im1: PIL type image
    :param im2: PIL type image
    :return:
    '''
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst

def write_text_to_img(img_tensor, logit, label) :
    '''
    입력으로 받은 image tensor 상단에 "Prob: {round(logit, 3)} , True: {label}" 글씨 써주는 함수
    :param img_tensor: [B, C, H, W] size의 이미지 Tensor
    :param logit: [B, 1] size의 Logit Tensor
    :param label: [B, 1] size의 Label Tensor
    :return: [B, C, H, W] Size의 이미지 tensor
    '''
    toPIL = transforms.ToPILImage()
    trans = [transforms.ToTensor(),
             transforms.Normalize(0.5,0.5)]
    trans = transforms.Compose(trans)

    output = []
    for single_tensor, prob, true in zip(img_tensor, logit, label) :
        img = toPIL((single_tensor + 1) / 2)
        draw = ImageDraw.Draw(img)
        prob = prob.detach().cpu().item()
        true = true.cpu().item()
        text = f'Prob: {round(prob, 3)} , True: {true}'
        font_size = 40
        text_color = 0  # 검은색 (0은 흑백 이미지에서의 검은색 값)
        text_position = (int(img.width / 2), 10)  # 이미지 상단 중간 좌표 설정
        font = ImageFont.load_default()  # 시스템 기본 폰트로 설정
        text_width, text_height = draw.textsize(text, font=font)
        text_position = (text_position[0] - int(text_width / 2), text_position[1])

        draw.text(text_position, text, font_size=font_size, font=font, fill=text_color)
        output.append(trans(img))
    output = torch.cat(output, 0).unsqueeze(1)
    return output

def print_PILimg(img) :
    '''
    Array, torch Tensor, Pillow 타입 입력을 출력 해주는 함수
    :param img: [H, W] size의 Pillow Image or Numpy, Tensor
    :return:
    '''
    plt.imshow(img)
    plt.show()

# ============== 이미지 처리 관련 함수 ==============

# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8, normalize=True, tile=False):
    '''
    :param image_tensor: (B, C, H, W) size의 이미지 tensor
    :param imtype: image type
    :param normalize: image tensor가 [-1, 1] 사이 값으로 normalize 되어있는지 여부
    :param tile: tile 형식으로 배치 할지 여부
    :return: (H, W, C) size의 이미지 array 출력
    '''
    if isinstance(image_tensor, list):
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2im(image_tensor[i], imtype, normalize))
        return image_numpy

    if image_tensor.dim() == 4:
        # transform each image in the batch
        images_np = []
        for b in range(image_tensor.size(0)):
            one_image = image_tensor[b]
            one_image_np = tensor2im(one_image, normalize=normalize)
            images_np.append(one_image_np.reshape(1, *one_image_np.shape))
        images_np = np.concatenate(images_np, axis=0)
        if tile:
            images_tensor = torch.tensor(images_np.transpose((0, 3, 1, 2)))
            images_grid = make_grid(images_tensor, nrow= 6)
            return torch.permute(images_grid, (1, 2, 0)).numpy()
        else:
            return images_np[0].transpose((2, 0, 1))

    if image_tensor.dim() == 2:
        image_tensor = image_tensor.unsqueeze(0)
    image_numpy = image_tensor.detach().cpu().float().numpy()
    if normalize:
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
    image_numpy = np.clip(image_numpy, 0, 255)
    # if image_numpy.shape[2] == 1:
    #     image_numpy = image_numpy[:, :, 0]
    return image_numpy.astype(imtype)


