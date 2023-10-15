DATA_ROOT=/media/bjw/Disk
TRAIN_SET=/root/autodl-tmp/dataset4SC_Depth
python train.py $TRAIN_SET \
--resnet-layers 18 \
--num-scales 1 \
-b2 -s0.1 -c0.2 --epoch-size 1000 --sequence-length 3 \
--with-ssim 1 \
--with-mask 1 \
--with-auto-mask 1 \
--with-pretrain 1 \
--log-output  \
--pretrained-disp  /root/autodl-tmp/SC_Depth_ckpts/pretrainDepth/10-13-03:47/dispnet_model_best.pth.tar \
--name /root/autodl-tmp/SC_Depth_ckpts/resnet18_depth_256_scene1_pretrainedDepthNet \
--lr "1e-5"