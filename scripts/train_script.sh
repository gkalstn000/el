nohup python train.py --id anogan --gpu_ids 3 --batchSize 2 --tf_log > anogan.out &

nohup python train.py --tf_log --id classifier_flat --gpu_ids 3 --batchSize 40 --num_workers 12  > classifier_flat.out &
nohup python train.py --tf_log --id classifier_stack --gpu_ids 3 --batchSize 40 --num_workers 12  > classifier_stack.out &


nohup python train.py --tf_log --id classifier_flat_aug --gpu_ids 3 --batchSize 40 --num_workers 12  > classifier_flat_aug.out &
