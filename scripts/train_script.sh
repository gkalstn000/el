nohup python train.py --id anogan --gpu_ids 3 --batchSize 2 --tf_log > anogan.out &

nohup python train.py --tf_log --id classifier_flat --gpu_ids 3 --batchSize 40 --num_workers 12  > classifier_flat.out &

nohup python train.py --tf_log --id classifier_flat --gpu_ids 2 --batchSize 40 --num_workers 15  > classifier_flat.out &

# RED
nohup python train.py --id classifier_first_noaug --tf_log --gpu_ids 1 --batchSize 8 --num_workers 10 --no_augment > classifier_first_noaug.out &
nohup python train.py --id classifier_first_aug --tf_log --gpu_ids 3 --batchSize 8 --num_workers 10 > classifier_first_aug.out &

nohup python train.py --id classifier_second_noaug_finetune --tf_log --gpu_ids 1 --batchSize 8 --num_workers 10 --no_augment --continue_train --data_mode second > classifier_second_noaug_finetune.out &
nohup python train.py --id classifier_second_aug_finetune --tf_log --gpu_ids 3 --batchSize 8 --num_workers 10 --continue_train --data_mode second > classifier_second_aug_finetune.out &





# NIPA
nohup python train.py --id classifier_first_noaug --tf_log --gpu_ids 1 --batchSize 18 --num_workers 10 --dataroot /home/work/msha/el/  --no_augment  > classifier_first_noaug.out &
nohup python train.py --id classifier_first_aug --tf_log --gpu_ids 1 --batchSize 18 --num_workers 10 --dataroot /home/work/msha/el/  > classifier_first_aug.out &

nohup python train.py --id classifier_first_aug_16 --tf_log --gpu_ids 1 --batchSize 10 --num_workers 7 --nef 16 --dataroot /home/work/msha/el/  > classifier_first_aug_16.out &