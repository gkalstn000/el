nohup python train.py --id anogan --gpu_ids 3 --batchSize 2 --tf_log > anogan.out &

nohup python train.py --tf_log --id classifier_flat --gpu_ids 3 --batchSize 40 --num_workers 12  > classifier_flat.out &

nohup python train.py --tf_log --id classifier_flat --gpu_ids 2 --batchSize 40 --num_workers 15  > classifier_flat.out &


nohup python train.py --id classifier_first_noaug --tf_log --gpu_ids 1 --batchSize 8 --num_workers 10 --no_augment > classifier_first_noaug.out &
nohup python train.py --id classifier_first_aug --tf_log --gpu_ids 3 --batchSize 8 --num_workers 10 > classifier_first_aug.out &