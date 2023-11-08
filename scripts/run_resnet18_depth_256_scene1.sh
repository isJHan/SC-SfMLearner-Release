DATA_ROOT=/media/bjw/Disk
TRAIN_SET=/home/jiahan/jiahan/datasets/C3VD/dataset_cecum_t1_a_4SCDepth/scenes
python -W ignore train.py $TRAIN_SET \
--resnet-layers 18 \
--num-scales 1 \
-b2 -s0.4 -c0.2 --epoch-size 1000 --sequence-length 3 \
--with-ssim 1 \
--with-mask 1 \
--with-auto-mask 1 \
--with-pretrain 1 \
--log-output  \
--name /home/jiahan/jiahan/checkpoints/SC_Depth_on_C3VD/scene1 \
--lr "1e-5" 