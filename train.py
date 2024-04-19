import argparse
import time
import csv
import datetime
from path import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.nn.functional as F

import models

import custom_transforms
# from utils import tensor2array, save_checkpoint
from utils import save_checkpoint
from datasets.sequence_folders import SequenceFolder, SimCol3D
from datasets.pair_folders import PairFolder
from loss_functions import compute_smooth_loss, compute_photo_and_geometry_loss, compute_errors
from logger import TermLogger, AverageMeter
from tensorboardX import SummaryWriter


parser = argparse.ArgumentParser(description='Structure from Motion Learner training on KITTI and CityScapes Dataset',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--folder-type', type=str, choices=['sequence', 'pair', 'simcol'], default='sequence', help='the dataset dype to train')
parser.add_argument('--sequence-length', type=int, metavar='N', help='sequence length for training', default=3)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--epoch-size', default=0, type=int, metavar='N', help='manual epoch size (will match dataset size if not set)')
parser.add_argument('-b', '--batch-size', default=4, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum for sgd, alpha parameter for adam')
parser.add_argument('--beta', default=0.999, type=float, metavar='M', help='beta parameters for adam')
parser.add_argument('--weight-decay', '--wd', default=0, type=float, metavar='W', help='weight decay')
parser.add_argument('--print-freq', default=10, type=int, metavar='N', help='print frequency')
parser.add_argument('--seed', default=0, type=int, help='seed for random functions, and network initialization')
parser.add_argument('--log-summary', default='progress_log_summary.csv', metavar='PATH', help='csv where to save per-epoch train and valid stats')
parser.add_argument('--log-full', default='progress_log_full.csv', metavar='PATH', help='csv where to save per-gradient descent train stats')
parser.add_argument('--log-output', action='store_true', help='will log dispnet outputs at validation step')
parser.add_argument('--resnet-layers',  type=int, default=18, choices=[18, 50], help='number of ResNet layers for depth estimation.')
parser.add_argument('--num-scales', '--number-of-scales', type=int, help='the number of scales', metavar='W', default=1)
parser.add_argument('-p', '--photo-loss-weight', type=float, help='weight for photometric loss', metavar='W', default=1)
parser.add_argument('-s', '--smooth-loss-weight', type=float, help='weight for disparity smoothness loss', metavar='W', default=0.1)
parser.add_argument('-c', '--geometry-consistency-weight', type=float, help='weight for depth consistency loss', metavar='W', default=0.5)
parser.add_argument('--with-ssim', type=int, default=1, help='with ssim or not')
parser.add_argument('--with-mask', type=int, default=1, help='with the the mask for moving objects and occlusions or not')
parser.add_argument('--with-auto-mask', type=int,  default=0, help='with the the mask for stationary points')
parser.add_argument('--with-pretrain', type=int,  default=1, help='with or without imagenet pretrain for resnet')
parser.add_argument('--dataset', type=str, choices=['kitti', 'nyu'], default='kitti', help='the dataset to train')
parser.add_argument('--pretrained-disp', dest='pretrained_disp', default=None, metavar='PATH', help='path to pre-trained dispnet model')
parser.add_argument('--pretrained-pose', dest='pretrained_pose', default=None, metavar='PATH', help='path to pre-trained Pose net model')
parser.add_argument('--name', dest='name', type=str, required=True, help='name of the experiment, checkpoints are stored in checpoints/name')
parser.add_argument('--padding-mode', type=str, choices=['zeros', 'border'], default='zeros',
                    help='padding mode for image warping : this is important for photometric differenciation when going outside target image.'
                         ' zeros will null gradients outside target image.'
                         ' border will only null gradients of the coordinate outside (x or y)')
parser.add_argument('--with-gt', action='store_true', help='use ground truth for validation. \
                    You need to store it in npy 2D arrays see data/kitti_raw_loader.py for an example')


best_error = -1
n_iter = 0
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.autograd.set_detect_anomaly(True)

my_writers = {}

distNet = None
optimizer_dist = None
__alpha,__beta = None,None

def main():
    global best_error, n_iter, device
    global distortion_net, optimizer_distortion, skip_frame # by jiahan
    global distNet, optimizer_dist
    
    args = parser.parse_args()

    timestamp = datetime.datetime.now().strftime("%y-%m-%d-%H:%M")
    save_path = Path(args.name)
    args.save_path = 'checkpoints'/save_path/timestamp
    print('=> will save everything to {}'.format(args.save_path))
    args.save_path.makedirs_p()
    
    # 保存参数
    with open(args.save_path/'args.txt', 'w') as f:
        tmp = str(args)
        for t in tmp.split(','):
            f.write(t)
            f.write('\n')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = True

    training_writer = SummaryWriter(args.save_path)
    output_writers = []
    if args.log_output:
        for i in range(3):
            output_writers.append(SummaryWriter(args.save_path/'valid'/str(i)))
    my_writers['depth_range'] = SummaryWriter(args.save_path/'mine_writer')
    my_writers['rotation'] = SummaryWriter(args.save_path/'mine_writer')
    my_writers['translation'] = SummaryWriter(args.save_path/'mine_writer')
    my_writers['depth_L1'] = SummaryWriter(args.save_path/'mine_writer')
    my_writers['alphab'] = SummaryWriter(args.save_path/'mine_writer')
    
    # Data loading code
    normalize = custom_transforms.Normalize(mean=[0.58, 0.58, 0.58],
                                            std=[0.20, 0.20, 0.20])
    # normalize = custom_transforms.Normalize(mean=[0.58, 0.24, 0.20], # RGB
    #                                         std=[0.20, 0.12, 0.11])

    train_transform = custom_transforms.Compose([
        # custom_transforms.RandomHorizontalFlip(),
        # custom_transforms.RandomScaleCrop(),
        custom_transforms.ArrayToTensor(),
        # normalize
    ])

    valid_transform = custom_transforms.Compose([
        custom_transforms.ArrayToTensor(), 
        # normalize
    ])

    print("=> fetching scenes in '{}'".format(args.data))
    if args.folder_type == 'sequence':
        train_set = SequenceFolder(
            args.data,
            transform=train_transform,
            seed=args.seed,
            train=True,
            sequence_length=args.sequence_length,
            dataset=args.dataset,
            skip_frames=3
        )
    elif args.folder_type == 'simcol':
        train_set = SimCol3D(
            args.data,
            transform=train_transform,
            seed=args.seed,
            train=True,
            sequence_length=args.sequence_length,
            dataset=args.dataset,
            skip_frames=1
        )
    else:
        train_set = PairFolder(
            args.data,
            seed=args.seed,
            train=True,
            transform=train_transform
        )


    # if no Groundtruth is avalaible, Validation set is the same type as training set to measure photometric loss from warping
    if args.with_gt:
        from datasets.validation_folders import ValidationSet
        val_set = ValidationSet(
            args.data,
            transform=valid_transform,
            dataset=args.dataset
        )
    else:
        if args.folder_type == 'simcol':
            val_set = SimCol3D(
                args.data,
                transform=valid_transform,
                seed=args.seed,
                train=False,
                sequence_length=args.sequence_length,
                dataset=args.dataset,
                skip_frames=1
            )
        else:
            val_set = SequenceFolder(
                args.data,
                transform=valid_transform,
                seed=args.seed,
                train=False,
                sequence_length=args.sequence_length,
                dataset=args.dataset,
                skip_frames=3
            )
    print('{} samples found in {} train scenes'.format(len(train_set), len(train_set.scenes)))
    print('{} samples found in {} valid scenes'.format(len(val_set), len(val_set.scenes)))
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.epoch_size == 0:
        args.epoch_size = len(train_loader)

    # create model
    print("=> creating model")
    disp_net = models.DispResNet(args.resnet_layers, args.with_pretrain).to(device)
    pose_net = models.PoseResNet(18, args.with_pretrain).to(device)
    
    # by jiahan
    # distortion_net = models.Learn_Distortion(len(train_set), True, True) # by jiahan
    # distortion_net = torch.nn.DataParallel(distortion_net)
    # optimizer_distortion = torch.optim.Adam(distortion_net.parameters(), lr=5e-4)
    # skip_frame = train_set.k

    distNet = models.DistNet().to(device)
    
    # load parameters
    if args.pretrained_disp:
        print("=> using pre-trained weights for DispResNet")
        weights = torch.load(args.pretrained_disp)
        disp_net.load_state_dict(weights['state_dict'], strict=False)

    if args.pretrained_pose:
        print("=> using pre-trained weights for PoseResNet")
        weights = torch.load(args.pretrained_pose)
        pose_net.load_state_dict(weights['state_dict'], strict=False)

    disp_net = torch.nn.DataParallel(disp_net)
    pose_net = torch.nn.DataParallel(pose_net)
    distNet = torch.nn.DataParallel(distNet)

    print('=> setting adam solver')
    optim_params = [
        {'params': disp_net.parameters(), 'lr': args.lr},
        {'params': pose_net.parameters(), 'lr': args.lr},
        {'params':distNet.parameters(), 'lr':5e-6}
    ]
    optimizer = torch.optim.Adam(optim_params,
                                 betas=(args.momentum, args.beta),
                                 weight_decay=args.weight_decay)
    # optimizer_dist = torch.optim.Adam([{'params':distNet.parameters(), 'lr':1e-3}],
    #                                   betas=(args.momentum, args.beta),
    #                                   weight_decay=args.weight_decay) # by jiahan NOTE 使用单独的优化器优化 DistNet, 只与 几何一致性损失有关系

    with open(args.save_path/args.log_summary, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(['train_loss', 'validation_loss'])

    with open(args.save_path/args.log_full, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(['train_loss', 'photo_loss', 'smooth_loss', 'geometry_consistency_loss'])

    logger = TermLogger(n_epochs=args.epochs, train_size=min(len(train_loader), args.epoch_size), valid_size=len(val_loader))
    logger.epoch_bar.start()

    for epoch in range(args.epochs):
        
        # logger.reset_valid_bar()
        # if args.with_gt:
        #     errors, error_names = validate_with_gt(args, val_loader, disp_net, epoch, logger, output_writers)
        # else:
        #     errors, error_names = validate_without_gt(args, val_loader, disp_net, pose_net, epoch, logger, output_writers)

        logger.epoch_bar.update(epoch)

        # train for one epoch
        logger.reset_train_bar()
        train_loss = train(args, train_loader, disp_net, pose_net, optimizer, args.epoch_size, logger, training_writer, epoch)
        logger.train_writer.write(' * Avg Loss : {:.3f}'.format(train_loss))

        # evaluate on validation set
        logger.reset_valid_bar()
        if args.with_gt:
            errors, error_names = validate_with_gt(args, val_loader, disp_net, epoch, logger, output_writers)
        else:
            errors, error_names = validate_without_gt(args, val_loader, disp_net, pose_net, epoch, logger, output_writers)
        error_string = ', '.join('{} : {:.3f}'.format(name, error) for name, error in zip(error_names, errors))
        logger.valid_writer.write(' * Avg {}'.format(error_string))

        for error, name in zip(errors, error_names):
            training_writer.add_scalar(name, error, epoch)

        # Up to you to chose the most relevant error to measure your model's performance, careful some measures are to maximize (such as a1,a2,a3)
        decisive_error = errors[1]
        if best_error < 0:
            best_error = decisive_error

        # remember lowest error and save checkpoint
        is_best = decisive_error < best_error
        best_error = min(best_error, decisive_error)
        save_checkpoint(
            args.save_path, {
                'epoch': epoch + 1,
                'state_dict': disp_net.module.state_dict()
            }, {
                'epoch': epoch + 1,
                'state_dict': pose_net.module.state_dict()
            },
            is_best,
            distnet_state={
                'epoch': epoch+1,
                'state_dict': distNet.module.state_dict()
            }, epoch=epoch)

        with open(args.save_path/args.log_summary, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow([train_loss, decisive_error])
    logger.epoch_bar.finish()


def train(args, train_loader, disp_net, pose_net, optimizer, epoch_size, logger, train_writer, epoch):
    global n_iter, device, optimizer_dist
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter(precision=4)
    w1, w2, w3 = args.photo_loss_weight, args.smooth_loss_weight, args.geometry_consistency_weight

    # switch to train mode
    disp_net.train()
    pose_net.train()
    distNet.train()

    end = time.time()
    logger.train_bar.update(0)
    
    # from utils import Gradient_Net
    # gradNet = Gradient_Net().to(device)

    for i, (tgt_img, ref_imgs, intrinsics, intrinsics_inv, ref_poses, others) in enumerate(train_loader):
        log_losses = i > 0 and n_iter % args.print_freq == 0

        # tgt_img = tgt_img[::,0:1,...]
        # ref_imgs = [ref_img[::,0:1,...] for ref_img in ref_imgs]
        # measure data loading time
        data_time.update(time.time() - end)
        tgt_img = tgt_img.to(device)
        ref_imgs = [img.to(device) for img in ref_imgs]
        intrinsics = intrinsics.to(device)
        
        tgt_disp = others['tgt_depth'].to(device)
        ref_disps = [d.to(device) for d in others['ref_depths']]

        # compute output
        tgt_depth, ref_depths = compute_depth(disp_net, tgt_img, ref_imgs)
        # tgt_depth, ref_depths = compute_depth_distortion(tgt_disp, ref_disps, i, skip_frame)
        # tgt_depth, ref_depths = compute_depth_withRes(disp_net, tgt_img, ref_imgs, tgt_disp,ref_disps, epoch)
        # NOTE load disp from gt
        tgt_depth_gt = others['tgt_depth_gt'].to(device)*100
        ref_depths_gt = [d.to(device)*100 for d in others['ref_depths_gt']]
        # ! simcol
        # tgt_depth_gt = others['tgt_depth_gt'].to(device)*200
        # ref_depths_gt = [d.to(device)*200 for d in others['ref_depths_gt']]
        
        
        poses, poses_inv = compute_pose_with_inv(pose_net, tgt_img, ref_imgs)
        # poses,poses_inv = compute_pose_from_gt(ref_poses)
        # poses_gt,poses_inv_gt = compute_pose_from_gt(ref_poses)
        # oflows = compute_oflows(tgt_img,ref_imgs)

        # # jh 给出旋转真值
        # for ii in range(len(poses)):
        #     poses[ii][...,3:] = poses_gt[ii][...,3:]
            
        loss_1, loss_3 = compute_photo_and_geometry_loss(tgt_img, ref_imgs, intrinsics, tgt_depth, ref_depths,
                                                         poses, poses_inv, args.num_scales, args.with_ssim,
                                                         args.with_mask, args.with_auto_mask, args.padding_mode)
        # loss_1, loss_3 = compute_photo_and_geometry_loss(tgt_img, ref_imgs, intrinsics, [tgt_depth_gt], [ref_depths_gt],
        #                                                  poses, poses_inv, args.num_scales, args.with_ssim,
        #                                                  args.with_mask, args.with_auto_mask, args.padding_mode)

        loss_2 = compute_smooth_loss(tgt_depth, tgt_img, ref_depths, ref_imgs)
        
        # by jiahan reproj
        # from loss_functions import compute_reprojection_loss
        # w_reproj = 0.0
        # oflows = others['ref_oflows']
        # oflows = [t.to('cuda') for t in oflows]
        # loss_reproj = compute_reprojection_loss(tgt_depth[0],oflows,poses,intrinsics)

        # w4 = 1
        # loss_4 = torch.tensor(0.0)
        # loss_4 = compute_pose_loss(poses,poses_inv, poses_gt,poses_inv_gt)
        
        # w5 = 0
        # loss_5 = compute_depth_grad_loss(tgt_disp,(1/tgt_depth[0]-0.02)/10, gradNet)
        
        w6 = 1.0
        loss_6 = compute_midas_loss_aux(tgt_depth, tgt_disp) # ! 对齐损失
        # loss_6 = compute_midas_loss_pearson(tgt_depth,tgt_disp)
        # 计算ref和tgt
        # for ref_depth,ref_disp in zip(ref_depths, ref_disps):
        #     loss_6 += compute_midas_loss_pearson(ref_depth,ref_disp)
        # loss_6 /= 3
        # w6 = 1
        # if epoch < 10: loss_6 += 0.01*compute_midas_loss(tgt_depth,tgt_disp)
        # w6 = 1
        # loss_6 = compute_ssim_loss(tgt_depth, tgt_disp)
        # w6_aux = 0
        # loss_6_aux = compute_midas_loss_aux(tgt_depth,tgt_disp,tgt_img=tgt_img)
        
        # loss = w1*loss_1 + w2*loss_2 + w3*loss_3 + w4*loss_4 + w5*loss_5
        loss = w1*loss_1 + w2*loss_2 + w3*loss_3 + w6*loss_6
        # loss = w1*loss_1 + w2*loss_2 + w3*loss_3 
        
        # jh 计算与真值的差，放缩预测深度到 [0,100]，并用平均值计算放缩系数保持尺度一致
        def meanOnB(tensor):
            """[B,1,H,W]"""
            return torch.mean(torch.mean(torch.mean(tensor,axis=-1,keepdim=True),axis=-2,keepdim=True),axis=-3,keepdim=True)
        
        mean_diff = meanOnB(tgt_depth_gt) / meanOnB(tgt_depth[0])
        depth_L1 = abs( tgt_depth_gt - (tgt_depth[0]*mean_diff) ).mean()
        depth_absrel = abs( tgt_depth_gt - (tgt_depth[0]*mean_diff)/(tgt_depth_gt+1e-3) ).mean()
        depth_rmse = torch.pow( torch.pow(tgt_depth_gt - (tgt_depth[0]*mean_diff), 2).mean(), 0.5 )

        if log_losses:
            train_writer.add_scalar('loss/photometric_error', loss_1.item(), n_iter)
            train_writer.add_scalar('loss/disparity_smoothness_loss', loss_2.item(), n_iter)
            train_writer.add_scalar('loss/geometry_consistency_loss', loss_3.item(), n_iter)
            train_writer.add_scalar('loss/loss_6', loss_6.item(), n_iter)
            # train_writer.add_scalar('loss/loss_6_aux', loss_6_aux.item(), n_iter)
            # train_writer.add_scalar('loss/reprojection_loss', loss_reproj.item(), n_iter)
            # train_writer.add_scalar('loss/pose_loss', loss_4.item(), n_iter)
            # train_writer.add_scalar('loss/disp_grad_loss', loss_5.item(), n_iter)
            train_writer.add_scalar('loss/total_loss', loss.item(), n_iter)
            
            my_writers['depth_L1'].add_scalar('depth/depth_L1(MAE)', depth_L1.item(), n_iter)
            my_writers['depth_L1'].add_scalar('depth/Abs_rel', depth_absrel.item(), n_iter)
            my_writers['depth_L1'].add_scalar('depth/RMSE', depth_rmse.item(), n_iter)
            my_writers['depth_range'].add_scalars('depth',
                                                  {'depth_max': tgt_depth[0].max().item(),
                                                   'depth_min': tgt_depth[0].min().item(),
                                                   'depth_mean': tgt_depth[0].mean().item()}, n_iter)
            if __alpha is not None: 
                my_writers['alphab'].add_scalars('depth/alpha beta',
                                             {
                                                 'alpha': __alpha,
                                                 'beta': __beta
                                             }, n_iter)

        # record loss and EPE
        losses.update(loss.item(), args.batch_size)

        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # by jiahan NOTE 优化 DistNet
        # optimizer_dist.zero_grad()
        # loss_3.backward()
        # optimizer_dist.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        with open(args.save_path/args.log_full, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow([loss.item(), loss_1.item(), loss_2.item(), loss_3.item()])
        logger.train_bar.update(i+1)
        if i % args.print_freq == 0:
            logger.train_writer.write('Train: Time {} Data {} Loss {}'.format(batch_time, data_time, losses))
        if i >= epoch_size - 1:
            break

        n_iter += 1

    return losses.avg[0]

flag = True
@torch.no_grad()
def validate_without_gt(args, val_loader, disp_net, pose_net, epoch, logger, output_writers=[]):
    global device,flag
    batch_time = AverageMeter()
    losses = AverageMeter(i=4, precision=4)
    log_outputs = len(output_writers) > 0

    # switch to evaluate mode
    disp_net.eval()
    pose_net.eval()
    distNet.eval()

    end = time.time()
    logger.valid_bar.update(0)
    
    if flag: loss1s = []
    for i, (tgt_img, ref_imgs, intrinsics, intrinsics_inv, ref_poses, others) in enumerate(val_loader):
        # tgt_img = tgt_img[::,0:1,...]
        # ref_imgs = [ref_img[::,0:1,...] for ref_img in ref_imgs]
        
        tgt_img = tgt_img.to(device)
        ref_imgs = [img.to(device) for img in ref_imgs]
        intrinsics = intrinsics.to(device)
        intrinsics_inv = intrinsics_inv.to(device)

        # load disp from midas
        tgt_disp = others['tgt_depth'].to(device)
        ref_disps = [d.to(device) for d in others['ref_depths']]
        # NOTE load disp from gt
        tgt_depth_gt = others['tgt_depth_gt'].to(device) * 100
        ref_depths_gt = [d.to(device)*100 for d in others['ref_depths_gt']]
        
        # compute output
        tgt_depth, ref_depths = compute_depth(disp_net, tgt_img, ref_imgs, eval=True)
        # tgt_depth, ref_depths, residual_tgt, residual_refs = compute_depth_withRes(disp_net, tgt_img, ref_imgs, tgt_disp,ref_disps, epoch,eval=True)
        ref_depth,ref_img = ref_depths[0], ref_imgs[0]
        # tgt_depth = [1 / (torch.log2(1+(disp_net(tgt_img)+tgt_disp)))]
        # tgt_depth = [1 / disp_net(tgt_img)]
        # ref_depths = []
        # for _,ref_img in enumerate(ref_imgs):
        #     # ref_depth = [1 / (torch.log2(1+(disp_net(ref_img)+ref_disps[i])))]
        #     ref_depth = [1 / disp_net(ref_img)]
        #     ref_depths.append(ref_depth)

        if log_outputs and i < len(output_writers):
            # 保存残差
            # np.save("./tmp/residual_{}.npy".format(epoch), residual_tgt.detach().cpu().numpy())
            if epoch == 0:
                output_writers[i].add_image('val Input', tensor2array(tgt_img[0]), 0)
                output_writers[i].add_image('val Depth GT',
                                            tensor2array(tgt_depth_gt[0][0], max_value=100.0, colormap='magma'),
                                            epoch)

            output_writers[i].add_image('val Dispnet Output Normalized',
                                        tensor2array(1/tgt_depth[0][0], max_value=None, colormap='magma'),
                                        epoch)
            output_writers[i].add_image('val Depth Output',
                                        tensor2array(tgt_depth[0][0], max_value=None, colormap='magma'),
                                        epoch)
            from inverse_warp import inverse_warp2, inverse_warp
            poses, poses_inv = compute_pose_with_inv(pose_net, tgt_img, ref_imgs)
            # poses, poses_inv = compute_pose_from_gt(ref_poses)
            # poses_gt, poses_inv_gt = compute_pose_from_gt(ref_poses)

            # jh 给出旋转真值
            # for ii in range(len(poses)):
            #     poses[ii][...,3:] = poses_gt[ii][...,3:]
            
            ref_img_warped, valid_mask, projected_depth, computed_depth = inverse_warp2(ref_imgs[0], tgt_depth[0], ref_depth[0], poses[0], intrinsics, args.padding_mode)
            # ref_img_warped, valid_mask, projected_depth, computed_depth = inverse_warp2(ref_imgs[0], tgt_depth_gt, ref_depths_gt[0], poses[0], intrinsics, args.padding_mode)
            output_writers[i].add_image('warp image', tensor2array(ref_img_warped[0]),epoch)
            output_writers[i].add_image('warp image - ref_img', abs(tensor2array(ref_img_warped[0]-ref_img[0])),epoch)

            # jh 计算与真值的差，放缩预测深度到 [0,100]，并用平均值计算放缩系数保持尺度一致
            def meanOnB(tensor):
                """[B,1,H,W]"""
                return torch.mean(torch.mean(torch.mean(tensor,axis=-1,keepdim=True),axis=-2,keepdim=True),axis=-3,keepdim=True)

            tgt_depth_gt = tgt_depth_gt*2 # ! SimCol3D
            ratio = meanOnB(tgt_depth[0]) / meanOnB(tgt_depth_gt) # NOTE 放缩平移量的尺度
            ratio_depth = meanOnB(tgt_depth_gt[0]) / meanOnB(tgt_depth[0][0]) # NOTE 只取 batch 是0的位置的计算
            depth_L1 = abs( tgt_depth_gt[0] - (tgt_depth[0][0]*ratio_depth) ).mean()
            depth_absrel = abs( tgt_depth_gt[0] - (tgt_depth[0][0]*ratio_depth)/(tgt_depth_gt+1e-3) ).mean()
            depth_rmse = torch.pow( torch.pow(tgt_depth_gt[0] - (tgt_depth[0][0]*ratio_depth), 2).mean(), 0.5 )
            
            output_writers[i].add_image('depth diff',
                                        tensor2array(abs(tgt_depth[0][0]*ratio_depth-tgt_depth_gt[0][0]), max_value=None, colormap='gray'),
                                        epoch)
            
            my_writers['depth_L1'].add_scalar('transform/val_MAE', depth_L1.item(), epoch)
            my_writers['depth_L1'].add_scalar('transform/val_AbsRel', depth_absrel.item(), epoch)
            my_writers['depth_L1'].add_scalar('transform/val_RMSE', depth_rmse.item(), epoch)
            
            my_writers['depth_L1'].add_scalars('transform/depth_minmax',
                                                  {'depth_min': tgt_depth[0][0].min(),
                                                   'depth_max': tgt_depth[0][0].max(),
                                                   'depth_mean': tgt_depth[0][0].mean()
                                                   }, epoch)
            
            my_writers['rotation'].add_scalars('transform/rotation',
                                                  {'rx': poses[0][0][3].item()*180/3.1415926,
                                                   'ry': poses[0][0][4].item()*180/3.1415926,
                                                   'rz': poses[0][0][5].item()*180/3.1415926,
                                                #    'rx_gt': poses_gt[0][0][3].item()*180/3.1415926,
                                                #    'ry_gt': poses_gt[0][0][4].item()*180/3.1415926,
                                                #    'rz_gt': poses_gt[0][0][5].item()*180/3.1415926,
                                                   }, epoch)
            my_writers['translation'].add_scalars('transform/translation',
                                                  {'tx': poses[0][0][0].item(),
                                                   'ty': poses[0][0][1].item(),
                                                   'tz': poses[0][0][2].item(),
                                                #    'tx_gt': poses_gt[0][0][0].item()*ratio.squeeze()[0].item(),
                                                #    'ty_gt': poses_gt[0][0][1].item()*ratio.squeeze()[0].item(),
                                                #    'tz_gt': poses_gt[0][0][2].item()*ratio.squeeze()[0].item(),
                                                   }, epoch)
            if __alpha is not None: 
                my_writers['alphab'].add_scalars('transform/alpha beta',
                                             {
                                                 'alpha': __alpha,
                                                 'beta': __beta
                                             }, epoch)
            

        poses, poses_inv = compute_pose_with_inv(pose_net, tgt_img, ref_imgs)
        # poses, poses_inv = compute_pose_from_gt(ref_poses)
        # poses_gt, poses_inv_gt = compute_pose_from_gt(ref_poses)

        # jh 给出旋转真值
        # for ii in range(len(poses)):
        #     poses[ii][...,3:] = poses_gt[ii][...,3:]
            
        
        loss_1, loss_3 = compute_photo_and_geometry_loss(tgt_img, ref_imgs, intrinsics, tgt_depth, ref_depths,
                                                         poses, poses_inv, args.num_scales, args.with_ssim,
                                                         args.with_mask, False, args.padding_mode)
        # loss_1, loss_3 = compute_photo_and_geometry_loss(tgt_img, ref_imgs, intrinsics, [tgt_depth_gt], [ref_depths_gt],
        #                                                  poses, poses_inv, args.num_scales, args.with_ssim,
        #                                                  args.with_mask, False, args.padding_mode)

        loss_2 = compute_smooth_loss(tgt_depth, tgt_img, ref_depths, ref_imgs)

        loss_1 = loss_1.item()
        loss_2 = loss_2.item()
        loss_3 = loss_3.item()
        
        if flag: loss1s.append(loss_1)

        loss = loss_1
        losses.update([loss, loss_1, loss_2, loss_3])

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        logger.valid_bar.update(i+1)
        if i % args.print_freq == 0:
            logger.valid_writer.write('valid: Time {} Loss {}'.format(batch_time, losses))

    if flag: np.save("./loss1.npy", np.array(loss1s))
    flag = False
    
    logger.valid_bar.update(len(val_loader))
    return losses.avg, ['Total loss', 'Photo loss', 'Smooth loss', 'Consistency loss']


@torch.no_grad()
def validate_with_gt(args, val_loader, disp_net, epoch, logger, output_writers=[]):
    global device
    batch_time = AverageMeter()
    error_names = ['abs_diff', 'abs_rel', 'sq_rel', 'a1', 'a2', 'a3']
    errors = AverageMeter(i=len(error_names))
    log_outputs = len(output_writers) > 0

    # switch to evaluate mode
    disp_net.eval()

    end = time.time()
    logger.valid_bar.update(0)
    for i, (tgt_img, depth) in enumerate(val_loader):
        tgt_img = tgt_img.to(device)
        depth = depth.to(device)

        # check gt
        if depth.nelement() == 0:
            continue

        # compute output
        output_disp = disp_net(tgt_img)
        output_depth = 1/output_disp[:, 0]

        if log_outputs and i < len(output_writers):
            if epoch == 0:
                output_writers[i].add_image('val Input', tensor2array(tgt_img[0]), 0)
                depth_to_show = depth[0]
                output_writers[i].add_image('val target Depth',
                                            tensor2array(depth_to_show, max_value=10),
                                            epoch)
                depth_to_show[depth_to_show == 0] = 1000
                disp_to_show = (1/depth_to_show).clamp(0, 10)
                # output_writers[i].add_image('val target Disparity Normalized',
                #                             tensor2array(disp_to_show, max_value=None, colormap='magma'),
                #                             epoch)
                output_writers[i].add_image('val target Disparity Normalized',
                                            tensor2array(disp_to_show, max_value=None, colormap='bone'),
                                            epoch)

            # output_writers[i].add_image('val Dispnet Output Normalized',
            #                             tensor2array(output_disp[0], max_value=None, colormap='magma'),
            #                             epoch)
            output_writers[i].add_image('val Dispnet Output Normalized',
                                        tensor2array(output_disp[0], max_value=None, colormap='bone'),
                                        epoch)
            output_writers[i].add_image('val Depth Output',
                                        tensor2array(output_depth[0], max_value=None, colormap='bone'),
                                        epoch)

        if depth.nelement() != output_depth.nelement():
            b, h, w = depth.size()
            output_depth = torch.nn.functional.interpolate(output_depth.unsqueeze(1), [h, w]).squeeze(1)

        errors.update(compute_errors(depth, output_depth, args.dataset))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        logger.valid_bar.update(i+1)
        if i % args.print_freq == 0:
            logger.valid_writer.write('valid: Time {} Abs Error {:.4f} ({:.4f})'.format(batch_time, errors.val[0], errors.avg[0]))
    logger.valid_bar.update(len(val_loader))
    return errors.avg, error_names


def compute_depth(disp_net, tgt_img, ref_imgs, eval=False):
    if eval: tgt_depth = [1/disp for disp in disp_net(tgt_img).unsqueeze(0)]
    else: tgt_depth = [1/disp for disp in disp_net(tgt_img)]
    # tgt_depth = [disp for disp in disp_net(tgt_img)]

    ref_depths = []
    for ref_img in ref_imgs:
        if eval: ref_depth = [1/disp for disp in disp_net(ref_img).unsqueeze(0)]
        else: ref_depth = [1/disp for disp in disp_net(ref_img)]
        # ref_depth = [disp for disp in disp_net(ref_img)]
        ref_depths.append(ref_depth)

    return tgt_depth, ref_depths


def compute_depth_distortion(tgt_disp, ref_disps, idx, skip_frame):
    idx_pre, idx_nxt = idx-skip_frame, idx+skip_frame
    alpha_cur,beta_cur = distortion_net(idx)
    
    

def compute_depth_withRes(disp_net, tgt_img, ref_imgs, tgt_disp_midas,ref_disps_midas, epoch,eval=False):
    # disp2depth log
    # def disp2depth(base, residual):
    #     alpha,beta = 3, 1
    #     tmp = torch.log2(
    #         torch.clamp(
    #             base + residual, 0.01
    #         ) +1
    #     )
    #     tmp = alpha * tmp + beta
    #     return 1 / tmp
    # disp2depth linear
    
    # 使用 alpha 和 beta 的估计
    def disp2depth(base, residual, input):
        global __alpha, __beta
        
        ab = distNet(input)
        alpha, beta = ab[:,0:1], ab[:,1:]**2
        __alpha = alpha[0][0]
        __beta = beta[0][0]
        alpha, beta = alpha.unsqueeze(-1).unsqueeze(-1), beta.unsqueeze(-1).unsqueeze(-1)
        # tmp = alpha*(1-base+(1-base)**2 * residual)+beta
        # if epoch < 2e7: tmp = alpha*(1-base+(1-base)**2 * residual)+0
        # alpha = 1.0 # NOTE no alpha
        # if epoch < 1e10: tmp = alpha*(1-base)+ (2-base)**2 * residual # NOTE 将 res 移到括号外面
        # else: tmp = alpha*(1-base)+ (1-base)**2 * residual + beta
        if epoch < 20: tmp = 1-base # NOTE 将 res 移到括号外面
        else: tmp = alpha*(1-base)
        tmp = torch.clamp(tmp,0.001)
        return tmp
    
    # def disp2depth(base, residual):
    #     alpha,beta = 1,0
    #     tmp = alpha*(
    #         torch.clamp(
    #             1-base+residual,
    #             0.01)
    #         ) + beta
    #     return tmp
        
    # tgt_disp_midas_multiScale = []
    # for i in range(4):
    #     tgt_disp_midas_multiScale.append(F.interpolate(tgt_disp_midas[i],scale_factor=0.5**i,mode='bilinear',align_corners=False))
    # ref_disps_midas_multiScale = []
    # for i in range(4):
        
    # tgt_depth = [1/(disp+tgt_disp_midas) for disp in disp_net(tgt_img)]
    # if eval: tgt_depth = [ 1 / (torch.log2(1+torch.clamp(disp+F.interpolate(tgt_disp_midas,scale_factor=0.5**i),0.01,1) )) for i,disp in enumerate(disp_net(tgt_img).unsqueeze(0)) ]
    # else: tgt_depth = [ 1 / (torch.log2(1+torch.clamp(disp+F.interpolate(tgt_disp_midas,scale_factor=0.5**i),0.01,1) )) for i,disp in enumerate(disp_net(tgt_img)) ]
    multi_ratio = 100.0 # NOTE 对深度图放缩的系数
    residual_tgt, _feature_tgt = disp_net(tgt_img)
    feature_tgt = _feature_tgt[-1]
    # if eval: tgt_depth = [ disp2depth( F.interpolate(tgt_disp_midas,scale_factor=0.5**i), disp) for i,disp in enumerate(residual_tgt.unsqueeze(0)) ]
    # else: tgt_depth = [ disp2depth( F.interpolate(tgt_disp_midas,scale_factor=0.5**i), disp) for i,disp in enumerate(residual_tgt ) ]
    if eval: tgt_depth = [ multi_ratio*disp2depth( F.interpolate(tgt_disp_midas,scale_factor=0.5**i), disp, feature_tgt) for i,disp in enumerate(residual_tgt.unsqueeze(0)) ]
    else: tgt_depth = [ multi_ratio*disp2depth( F.interpolate(tgt_disp_midas,scale_factor=0.5**i), disp, feature_tgt) for i,disp in enumerate(residual_tgt ) ]

    residual_refs = []
    ref_depths = []
    for i,ref_img in enumerate(ref_imgs):
        # ref_depth = [1/(disp+ref_disps_midas[i]) for disp in disp_net(ref_img)]
        # if eval: ref_depth = [ 1 / (torch.log2(1+torch.clamp(disp+F.interpolate(ref_disps_midas[i],scale_factor=0.5**j), 0.01,1)) ) for j,disp in enumerate(disp_net(ref_img).unsqueeze(0)) ]
        # else: ref_depth = [ 1 / (torch.log2(1+torch.clamp(disp+F.interpolate(ref_disps_midas[i],scale_factor=0.5**j),0.01,1)) )for j,disp in enumerate(disp_net(ref_img)) ]
        residual_ref, _feature_tgt = disp_net(ref_img)
        feature_ref = _feature_tgt[-1]
        # if eval: ref_depth = [ disp2depth( F.interpolate(ref_disps_midas[i],scale_factor=0.5**j), disp) for j,disp in enumerate(residual_ref.unsqueeze(0)) ]
        # else: ref_depth = [ disp2depth( F.interpolate(ref_disps_midas[i],scale_factor=0.5**j), disp) for j,disp in enumerate(residual_ref) ]
        if eval: ref_depth = [ multi_ratio*disp2depth( F.interpolate(ref_disps_midas[i],scale_factor=0.5**j), disp, feature_ref) for j,disp in enumerate(residual_ref.unsqueeze(0)) ]
        else: ref_depth = [ multi_ratio*disp2depth( F.interpolate(ref_disps_midas[i],scale_factor=0.5**j), disp, feature_ref) for j,disp in enumerate(residual_ref) ]
        ref_depths.append(ref_depth)
        residual_refs.append(residual_ref)

    # if eval: return tgt_depth, ref_depths, residual_tgt, residual_refs
    # return tgt_depth, ref_depths
    # NOTE *100 让smooth损失项的值比较理想
    if eval: return tgt_depth, ref_depths, residual_tgt, residual_refs
    return tgt_depth, ref_depths


def compute_pose_with_inv(pose_net, tgt_img, ref_imgs):
    poses = []
    poses_inv = []
    for ref_img in ref_imgs:
        poses.append(pose_net(tgt_img, ref_img))
        poses_inv.append(pose_net(ref_img, tgt_img))

    return poses, poses_inv

from scipy.spatial.transform import Rotation
def compute_pose_from_gt(sample):
    # sample: list of [B,4,4] tensor
    rate = 1.0
    
    poses = []
    poses_inv = []
    for ref_poses in sample:
        pose,pose_inv = [],[]
        for ref_pose in ref_poses:
            ref_pose = ref_pose.numpy()
            ref_pose_inv = np.linalg.inv(ref_pose)
            t,rot = ref_pose[:3,3]/rate, ref_pose[:3,:3]
            euler = Rotation.from_matrix(rot).as_euler('xyz').astype(np.float32)
            pose.append(np.concatenate((t,euler),0))
            
            t_inv,rot_inv = ref_pose_inv[:3,3]/rate, ref_pose_inv[:3,:3]
            euler_inv = Rotation.from_matrix(rot_inv).as_euler('xyz').astype(np.float32)
            pose_inv.append(np.concatenate((t_inv,euler_inv),0))

        pose,pose_inv = torch.from_numpy(np.array(pose).astype(np.float32)),torch.from_numpy(np.array(pose_inv).astype(np.float32))
        pose,pose_inv = pose.to(device), pose_inv.to(device)
        poses.append(pose)
        poses_inv.append(pose_inv)       
    # poses,poses_inv = torch.from_numpy(np.array(poses).astype(np.float32)),torch.from_numpy(np.array(poses_inv).astype(np.float32))
    # poses,poses_inv = poses.to(device), poses_inv.to(device)
    # print(poses[0][0][:3])
    return poses, poses_inv # [2,B,6]

def compute_oflows(tgt_img,ref_imgs):
    """compute optical flows from tgt_img to ref_imgs

    Args:
        tgt_img (_type_): _description_
        ref_imgs (_type_): _description_
    """
    pass

def compute_pose_loss(poses,poses_inv, poses_gt,poses_inv_gt):
    loss_4 = 0
    for i in range(len(poses)):
        loss_4 += (poses[i]-poses_gt[i]).norm(p=1)
        # loss_4 += (poses_inv[i]-poses_inv_gt[i]).norm(p=1)
    return loss_4

def compute_depth_grad_loss(disp_dpt,disp_output, gradNet):
    loss = 0.0
    (gradX_dpt,gradY_dpt),(gradX_output,gradY_output) = gradNet(disp_dpt), gradNet(disp_output)
    loss += (gradX_dpt-gradX_output).abs().mean()
    loss += (gradY_dpt-gradY_output).abs().mean()
    
    return loss

@torch.no_grad()
def align_depth(ref_depth, tgt_depth):
    B,_,h,w = ref_depth.shape
    
    # 解方程
    A = torch.ones((B, h*w,2))
    A[...,0] = tgt_depth.view(B,-1)
    b = torch.ones((B, h*w,1))
    b[...,0] = ref_depth.view(B,-1)
    x = torch.matmul( torch.matmul( torch.inverse(torch.matmul(A.permute(0,2,1), A)), A.permute(0,2,1)),  b )
    
    alpha, beta = x[...,0,::].view(B,1,1,1).to(device), x[...,1,::].view(B,1,1,1).to(device)
    depth_midas = alpha*tgt_depth + beta
    return depth_midas

@torch.no_grad()
def align_depth_mean(ref_depth, tgt_depth):
    B,_,h,w = ref_depth.shape
    
    mean1,mean2 = ref_depth.view((B,-1)).mean(axis=1),tgt_depth.view((B,-1)).mean(axis=1)
    ratio = mean1/mean2
    ratio = ratio.view((B,1,1,1))
    return ratio * tgt_depth
    
def compute_midas_loss(tgt_depth,tgt_disp_midas):
    loss = 0.0
    B,_,h,w = tgt_depth[0].shape
    
    for i in range(len(tgt_depth)): # 多尺度
        tgt_disp_reshaped = 1- F.interpolate(tgt_disp_midas,scale_factor=0.5**i)
        # mean1,mean2 = tgt_depth[i].view((B,-1)).mean(axis=1),tgt_disp_reshaped.view((B,-1)).mean(axis=1)
        # ratio = mean1/mean2
        # ratio = ratio.view((B,1,1,1))
        # loss = loss + torch.norm(tgt_disp_reshaped*ratio - tgt_depth[i])
        tgt_disp_reshaped_aligned = align_depth_mean(tgt_depth[i], tgt_disp_reshaped) # NOTE 对齐平均值来计算损失
        loss = loss + torch.norm(tgt_disp_reshaped_aligned-tgt_depth[i])
    
    return loss

# 监督一部分区域
def compute_midas_loss_aux(tgt_depth,tgt_disp_midas, tgt_img=None):
    loss = 0.0
    B,_,h,w = tgt_depth[0].shape
    
    if tgt_img is not None: brightness_mask = (tgt_img[::,0:1,...]>0.85) | (tgt_img[::,0:1,...]<0.1) # NOTE 灰度
    else: brightness_mask = torch.ones_like(tgt_depth[0])
    
    # for i in range(1): # 只监督最大尺度的
    for i in range(len(tgt_depth)):
        tgt_disp_reshaped = 1- F.interpolate(tgt_disp_midas,scale_factor=0.5**i)
        if tgt_img is not None:
            tgt_img_reshaped = F.interpolate(tgt_img,scale_factor=0.5**i)
            brightness_mask = (tgt_img_reshaped[::,0:1,...]>0.85) | (tgt_img_reshaped[::,0:1,...]<0.1) # NOTE 灰度
        else:
            brightness_mask = torch.ones_like(tgt_disp_reshaped)
                    
        mean1,mean2 = tgt_depth[i].view((B,-1)).mean(axis=1),tgt_disp_reshaped.view((B,-1)).mean(axis=1)
        ratio = mean1/mean2
        ratio = ratio.view((B,1,1,1))
        if i!=0: loss = loss + torch.norm(tgt_disp_reshaped*ratio - tgt_depth[i])
        else: loss = loss + torch.norm( brightness_mask * (tgt_disp_reshaped*ratio - tgt_depth[i]) )
    
    return loss

# 监督一部分区域
def compute_midas_loss_aux2(tgt_depth,tgt_disp_midas, tgt_img=None):
    """不同之处在于1/(output+88.8)转化

    Args:
        tgt_depth (_type_): _description_
        tgt_disp_midas (_type_): _description_
        tgt_img (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    loss = 0.0
    B,_,h,w = tgt_depth[0].shape
    
    if tgt_img is not None: brightness_mask = (tgt_img[::,0:1,...]>0.85) | (tgt_img[::,0:1,...]<0.1) # NOTE 灰度
    else: brightness_mask = torch.ones_like(tgt_depth[0])
    
    # for i in range(1): # 只监督最大尺度的
    for i in range(len(tgt_depth)):
        tgt_disp_reshaped = 1/(88.8+F.interpolate(tgt_disp_midas,scale_factor=0.5**i)) # ! 不同之处在这里
        if tgt_img is not None:
            tgt_img_reshaped = F.interpolate(tgt_img,scale_factor=0.5**i)
            brightness_mask = (tgt_img_reshaped[::,0:1,...]>0.85) | (tgt_img_reshaped[::,0:1,...]<0.1) # NOTE 灰度
        else:
            brightness_mask = torch.ones_like(tgt_disp_reshaped)
                    
        mean1,mean2 = tgt_depth[i].view((B,-1)).mean(axis=1),tgt_disp_reshaped.view((B,-1)).mean(axis=1)
        ratio = mean1/mean2
        ratio = ratio.view((B,1,1,1))
        if i!=0: loss = loss + torch.norm(tgt_disp_reshaped*ratio - tgt_depth[i])
        else: loss = loss + torch.norm( brightness_mask * (tgt_disp_reshaped*ratio - tgt_depth[i]) )
    
    return loss


# 负皮尔逊相关损失
def compute_midas_loss_pearson(tgt_depth,tgt_disp_midas):
    loss = 0.0
    B,_,h,w = tgt_depth[0].shape
    for i in range(len(tgt_depth)):
        tmp_tgt = tgt_depth[i].view((B,-1))
        tgt_disp_reshaped = (1- F.interpolate(tgt_disp_midas,scale_factor=0.5**i)).view((B,-1))
        with torch.no_grad(): mean1,mean2 = tmp_tgt.mean(axis=-1).view((B,1)), tgt_disp_reshaped.mean(axis=-1).view((B,1))
        tmp_tgt_center = tmp_tgt - mean1
        tgt_disp_reshaped_center = tgt_disp_reshaped - mean2
        cov = torch.sum(tmp_tgt_center*tgt_disp_reshaped_center,dim=-1)
        var1 = torch.sum(tmp_tgt_center**2,dim=-1)
        var2 = torch.sum(tgt_disp_reshaped_center**2,dim=-1)
        # cov = torch.matmul(tmp_tgt_center.view((B,1,-1)),tgt_disp_reshaped_center.view((B,-1,1))).view((B,-1))
        # cov = cov/tmp_tgt_center.shape[-1]
        # var1 = torch.var(tmp_tgt_center,dim=-1)
        # var2 = torch.var(tgt_disp_reshaped_center, dim=-1)
        loss += ((1 - cov/torch.sqrt(var1*var2)) /2).mean()
    
    return loss


# 深度图的 SSIM 损失
def compute_ssim_loss(tgt_depth, tgt_disp_midas):
    from loss_functions import compute_ssim_loss
    loss = 0.0
    B,_,h,w = tgt_depth[0].shape
    for i in range(len(tgt_depth)): # 多尺度
        tgt_disp_reshaped = 1- F.interpolate(tgt_disp_midas,scale_factor=0.5**i)
        mean1,mean2 = tgt_depth[i].view((B,-1)).mean(axis=1),tgt_disp_reshaped.view((B,-1)).mean(axis=1)
        ratio = mean1/mean2
        ratio = ratio.view((B,1,1,1))
        loss = loss + compute_ssim_loss(tgt_disp_reshaped*ratio, tgt_depth[i]).mean()
    
    return loss


from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap


def high_res_colormap(low_res_cmap, resolution=1000, max_value=1):
    # Construct the list colormap, with interpolated values for higer resolution
    # For a linear segmented colormap, you can just specify the number of point in
    # cm.get_cmap(name, lutsize) with the parameter lutsize
    x = np.linspace(0, 1, low_res_cmap.N)
    low_res = low_res_cmap(x)
    new_x = np.linspace(0, max_value, resolution)
    high_res = np.stack([np.interp(new_x, x, low_res[:, i])
                         for i in range(low_res.shape[1])], axis=1)
    return ListedColormap(high_res)


def opencv_rainbow(resolution=1000):
    # Construct the opencv equivalent of Rainbow
    opencv_rainbow_data = (
        (0.000, (1.00, 0.00, 0.00)),
        (0.400, (1.00, 1.00, 0.00)),
        (0.600, (0.00, 1.00, 0.00)),
        (0.800, (0.00, 0.00, 1.00)),
        (1.000, (0.60, 0.00, 1.00))
    )

    return LinearSegmentedColormap.from_list('opencv_rainbow', opencv_rainbow_data, resolution)



COLORMAPS = {'rainbow': opencv_rainbow(),
             'magma': high_res_colormap(cm.get_cmap('magma')),
             'bone': cm.get_cmap('bone', 10000),
             'gray': cm.get_cmap('gray', 10000)}


def tensor2array(tensor, max_value=None, colormap='rainbow'):
    tensor = tensor.detach().cpu()
    if max_value is None:
        max_value = tensor.max().item()
    if tensor.ndimension() == 2 or tensor.size(0) == 1:
        norm_array = tensor.squeeze().numpy()/max_value
        array = COLORMAPS[colormap](norm_array).astype(np.float32)
        array = array.transpose(2, 0, 1)

    elif tensor.ndimension() == 3:
        assert(tensor.size(0) == 3)
        # array = 0.45 + tensor.numpy()*0.225
        # array = 0.58 + tensor.numpy()*0.2
        array = 0 + tensor.numpy()*1
    return array




if __name__ == '__main__':
    main()
