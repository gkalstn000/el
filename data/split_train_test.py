from PIL import Image
import pandas as pd
import os
import util.util as util
from tqdm import tqdm
import random
random.seed(1717)

def split_filelist(filelist):
    # 이미지 파일 이름 목록을 랜덤하게 섞음
    random.shuffle(filelist)

    # 분할 비율
    train_ratio = 0.7
    test_ratio = 0.3

    # 분할 인덱스 계산
    train_size = int(len(filelist) * train_ratio)
    test_size = len(filelist) - train_size

    # 훈련용 이미지 파일 이름 목록
    train_filenames = filelist[:train_size]

    # 테스트용 이미지 파일 이름 목록
    test_filenames = filelist[train_size:]

    return train_filenames, test_filenames

def make_list_to_dict(filename_list, label) :
    dict_ = {'filename' : filename_list,
             'label' : [label] * len(filename_list)}
    return dict_

def concat_dict(dict1, dict2) :
    dict_ = {'filename' : dict1['filename'] + dict2['filename'],
             'label' : dict1['label'] + dict2['label']}
    return dict_


dataroot = '/datasets/msha/el'
fault_list = os.listdir(os.path.join(dataroot, 'fault'))
non_fault_list = os.listdir(os.path.join(dataroot, 'non_fault'))

classification_root = os.path.join(dataroot, 'classification')
anodetect_root = os.path.join(dataroot, 'anodetect')


train_non_fault_list, test_non_fault_list = split_filelist(non_fault_list)
train_fault_list, test_fault_list = split_filelist(fault_list)

train_non_fault_dict = make_list_to_dict(train_non_fault_list, 0)
test_non_fault_dict = make_list_to_dict(test_non_fault_list, 0)
train_fault_dict = make_list_to_dict(train_fault_list, 1)
test_fault_dict = make_list_to_dict(test_fault_list, 1)

classification_train_dict = concat_dict(train_non_fault_dict, train_fault_dict)
classification_test_dict = concat_dict(test_non_fault_dict, test_fault_dict)
anodetect_train_dict = train_non_fault_dict
anodetect_test_dict = concat_dict(concat_dict(test_fault_dict, train_fault_dict), test_non_fault_dict)

pd.DataFrame.from_dict(classification_train_dict).to_csv(os.path.join(dataroot, 'classification/train/data_df.csv'), index=False)
pd.DataFrame.from_dict(classification_test_dict).to_csv(os.path.join(dataroot, 'classification/test/data_df.csv'), index=False)
pd.DataFrame.from_dict(anodetect_train_dict).to_csv(os.path.join(dataroot, 'anodetect/train/data_df.csv'), index=False)
pd.DataFrame.from_dict(anodetect_test_dict).to_csv(os.path.join(dataroot, 'anodetect/test/data_df.csv'), index=False)

