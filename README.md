# el

## Prepare
github clone 
```Bash
git clone https://github.com/gkalstn000/el.git
mv el
```

가상환경
```Bash
conda create -n el python==3.8
conda activate el
```

package 설치
```Bash
pip install -r requirements.txt
```

## Dataset 구성
* `data_df_[mode].csv`는 학습에 필요한 이미지명과 label이 저장 된 파일
* 원래 `Anomal Detect`와 `Classification` 둘다 고려해서 짠 Framework라 Anodetect 폴더도 있음.
* `first`는 1차 검사 이미지(정상/비정장) 이미지 폴더이고 `second`는 2차 검사 이미지(정상/비정상) 이미지 폴더임.
```
el
├── anodetect
│   ├── test
│   │   ├── data_df_first.csv
│   │   └── data_df_second.csv
│   └── train
│       ├── data_df_first.csv
│       └── data_df_second.csv
├── classification
│   ├── test
│   │   ├── data_df_first.csv
│   │   └── data_df_second.csv
│   └── train
│       ├── data_df_first.csv
│       └── data_df_second.csv
├── df_label.csv
│
├── first
│   ├── fault
│   │   ├── 084013.jpg
│   │   ├── 084547.jpg
│   │   ├── BKDGGA11139.jpg
│   │   └── ...
│   └── non_fault
│       ├── 0105366018301696.jpg
│       ├── 0105748496182163_1.jpg
│       ├── 0105748496182163_2.jpg
│       ├── ...
├── second
│   ├── fault
│   │   ├── BJDGGA03197.jpg
│   │   ├── BKDGGA03363.jpg
│   │   ├── ...
│   └── non_fault
│       ├── BJDGGA01168.jpg
│       ├── BJDGGA06244.jpg
│       ├── BJDGGA06456.jpg
│       ├── ...
└── simulator.csv

```


## Training 예시
```Bash
python train.py --id [실험ID] --tf_log --gpu_ids 1 --batchSize 18 --num_workers 10 --dataroot /home/work/msha/el/
```

학습 결과 `./checkpoints/[실험ID]` 폴더내에 저장

### Tensorboard 
```Bash
# Remote Server에서
cd checkpoints/[실험ID]
nohup tensorboard --logdir logs --port [Remote_Port_Num] &

# Local Machine에서
ssh -N -f -L localhost:[Local_Port_Num]:localhost:[Remote_Port_Num] [Server_IP_Addr] -p [Server_Port_Num]
```
브라우저에서 `http://localhost:[Local_Port_Num]/` 으로 접속

```
[실험ID]/
    ├── N_net_E.pth
    ├── ...
    ├── iter.txt
    ├── latest_net_E.pth
    ├── logs
    │   └── events.out.tfevents.1686133342.red-WS-C621E-SAGE-Series.2087.0
    ├── loss_log.txt
    ├── opt.pkl
    ├── opt.txt
    └── web
        ├── images
        │   ├── epoch001_iter5000_Image with logit.png
        │   ├── epoch001_iter8560_[Valid] Image with logit .png
        │
        └── index.html
```


## Test 예시
```Bash
python test.py --id classifier_first_aug --gpu_ids 1 --batchSize 18 --num_workers 10 --dataroot /home/work/msha/el/
```
실험 결과 `results/[실험ID]` 폴더내에 저장
```
[실험ID]/
├── confusion_matrix.png
├── logit.csv
├── score.csv
└── vis
    ├── 0105748496182163_1.jpg
    ├── 0115214198159248.jpg
    ├── 084547.jpg
    ├── 091725.jpg
    ├── 114947.jpg
    ├── ...

```
