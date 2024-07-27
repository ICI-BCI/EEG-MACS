python train_MACS.py --epoch 30 --num_classes 2 --batch_size 60 --low_dim 128 --ua_ratio 0.3 \
--network "mAtt" --lr 0.1 --wd 1e-4  --batch_t 0.1 \
--memory_use 1 --memory_begin 3 --memory_per_class 100 --balance_crit "median" --discrepancy_corrected 1 --startLabelCorrection 5 \
--PredictiveCorrection 1 --k_val 16 --experiment_name PD \
--cuda_dev 0 --m 2 --dataset pd --uns_queue_k 100  --foldb 1 --folde 4 --channel 63 --dim 18

python3 train_MACS.py --epoch 30 --num_classes 2 --batch_size 128 --low_dim 128 --ua_ratio 0.3 \
--network "mAtt" --lr 0.1 --wd 1e-4  \
--batch_t 0.1 --memory_use 1 --memory_begin 3 --memory_per_class 300 \
--balance_crit "median" --discrepancy_corrected 1 \
--startLabelCorrection 5 --PredictiveCorrection 1 --k_val 25 \
--experiment_name MCI --cuda_dev 0 --m 4 --dataset mci --dim 32 --foldb 1 --folde 5 --uns_queue_k 300 --channel 62

