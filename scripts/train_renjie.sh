DATA_ROOT=/media/bjw/Disk
TRAIN_SET=/Disk_2/Jiahan/renjie_dataset
python -W ignore train.py $TRAIN_SET \
--resnet-layers 18 \
--num-scales 1 \
-b1 -s0.3 -c0.5 --epoch-size 100000 --sequence-length 3 \
--with-ssim 1 \
--with-mask 1 \
--with-auto-mask 1 \
--with-pretrain 1 \
--log-output  \
--name /Disk_2/Jiahan/renjie_dataset/checkpoints/first_time/ \
--lr "1e-4" \
--epochs 50
