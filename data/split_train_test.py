from PIL import Image
import pandas as pd
import os
import util.util as util
from tqdm import tqdm
import random
from tqdm import tqdm
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

def integrity_check(root, filelist) :
    output = []
    for filepath in tqdm(filelist) :
        try :
            Image.open(os.path.join(root, filepath)).convert("L")
        except Exception as e :
            print(os.path.join(root, filepath), 'is error')

        else :
            output.append(filepath)
    return output

mode = 'first'
dataroot = '/datasets/msha/el'

fault_list = os.listdir(os.path.join(dataroot, mode, 'fault'))
non_fault_list = os.listdir(os.path.join(dataroot, mode, 'non_fault'))

fault_list = integrity_check(os.path.join(dataroot, mode, 'fault'), fault_list)
non_fault_list = integrity_check(os.path.join(dataroot, mode, 'non_fault'), non_fault_list)


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

classification_train = pd.DataFrame.from_dict(classification_train_dict)
classification_test = pd.DataFrame.from_dict(classification_test_dict)
anodetect_train = pd.DataFrame.from_dict(anodetect_train_dict)
anodetect_test = pd.DataFrame.from_dict(anodetect_test_dict)

classification_train.to_csv(os.path.join(dataroot, f'classification/train/data_df_{mode}.csv'), index=False)
classification_test.to_csv(os.path.join(dataroot, f'classification/test/data_df_{mode}.csv'), index=False)
anodetect_train.to_csv(os.path.join(dataroot, f'anodetect/train/data_df_{mode}.csv'), index=False)
anodetect_test.to_csv(os.path.join(dataroot, f'anodetect/test/data_df_{mode}.csv'), index=False)