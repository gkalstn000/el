from PIL import Image
import pandas as pd
import os
import util.util as util
from tqdm import tqdm

dataroot = '/datasets/msha/el'

df = pd.read_csv(os.path.join(dataroot, 'simulator.csv'))
df = df.dropna(axis=1)
label_dict = {'ID' : [],
              'label' : []}


fault_path = os.path.join(dataroot, 'fault')
non_fault_path = os.path.join(dataroot, 'non_fault')
# util.mkdirs(fault_path)
# util.mkdirs(non_fault_path)

for root, dirs, files in  tqdm(sorted(os.walk(os.path.join(dataroot, 'images')))) :
    label = root.split('/')[-1]
    if label not in ['OK', 'NG'] : continue
    fault = 1 if label == 'NG' else 0
    save_path = fault_path if fault == 1 else non_fault_path
    for filename in tqdm(files) :
        ext = filename.split('.')[-1]
        if ext != 'jpg' : continue

        # img = Image.open(os.path.join(root, filename))
        # img.save(os.path.join(save_path, filename))

        label_dict['ID'].append(filename.replace('.jpg', ''))
        label_dict['label'].append(fault)

label_df = pd.DataFrame.from_dict(label_dict)
df_merged = df.merge(label_df, on='ID', how='inner')
