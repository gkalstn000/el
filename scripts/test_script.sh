#python test.py --id anodetect_stack --gpu_ids 2 --batchSize 1 --num_workers 1  --which_epoch 5
#
#
#python test.py --id classifier_both_aug --gpu_ids 1 --batchSize 20 --num_workers 10 --dataroot /home/work/msha/el/ --data_mode first
#python test.py --id classifier_both_aug --gpu_ids 1 --batchSize 20 --num_workers 10 --dataroot /home/work/msha/el/ --data_mode second
#python test.py --id classifier_both_noaug --gpu_ids 1 --batchSize 20 --num_workers 10 --dataroot /home/work/msha/el/ --data_mode first
#python test.py --id classifier_both_noaug --gpu_ids 1 --batchSize 20 --num_workers 10 --dataroot /home/work/msha/el/ --data_mode second

#python test.py --id classifier_both_noaug --gpu_ids 1 --batchSize 20 --num_workers 10 --dataroot /home/work/msha/el/ --data_mode second --which_epoch 20
#python test.py --id classifier_both_noaug --gpu_ids 1 --batchSize 20 --num_workers 10 --dataroot /home/work/msha/el/ --data_mode second --which_epoch 30
#python test.py --id classifier_both_noaug --gpu_ids 1 --batchSize 20 --num_workers 10 --dataroot /home/work/msha/el/ --data_mode second --which_epoch 40
#python test.py --id classifier_both_noaug --gpu_ids 1 --batchSize 20 --num_workers 10 --dataroot /home/work/msha/el/ --data_mode second --which_epoch 50
python test.py --id classifier_first_aug --gpu_ids 1 --batchSize 20 --num_workers 10 --dataroot /home/work/msha/el/ --data_mode first



