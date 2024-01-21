import sys
import os

train_set = "/home/jiahan/jiahan/datasets/C3VD/dataset_cecum_t1_a_4SCDepth/scenes"
batchs = [2]
# smooths = [0.2,0.1]
smooths = [0.3]
# consistancys = [0.3,0.4,0.6,0.7,0.8]
consistancys = [0.5]
lrs = ['1e-4','3e-4','5e-4','7e-4','7e-5','5e-5','3e-5','1e-5']


for b in batchs:
    for s in smooths:
        for c in consistancys:
            for lr in lrs:
                cmd = '''python -W ignore train.py {} \
--resnet-layers 18 \
--num-scales 1 \
--epochs 50 \
-b{} -s{} -c{} --epoch-size 1000 --sequence-length 3 \
--with-ssim 1 \
--with-mask 1 \
--with-auto-mask 1 \
--with-pretrain 1 \
--log-output  \
--name /home/jiahan/jiahan/checkpoints/SC_Depth_on_C3VD/scene1_lossMidas/hyper_4 \
--lr "{}" '''.format(train_set,b,s,c,lr)
                print(cmd)
                os.system(cmd)