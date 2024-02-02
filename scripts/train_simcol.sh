DATA_ROOT=/media/bjw/Disk
TRAIN_SET=/home/jiahan/jiahan/datasets/SimCol/SimCol3D4SC_Depth
python -W ignore train.py $TRAIN_SET \
--resnet-layers 18 \
--num-scales 1 \
-b2 -s0.3 -c0.5 --epoch-size 100000 --sequence-length 3 \
--with-ssim 1 \
--with-mask 1 \
--with-auto-mask 1 \
--with-pretrain 1 \
--log-output  \
--name /home/jiahan/jiahan/checkpoints/SC_Depth_on_SimCol3D/scenes_all_SimCol3D \
--lr "1e-4" \
--epochs 70 \
--folder-type simcol