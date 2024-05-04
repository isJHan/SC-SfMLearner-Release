import torch.utils.data as data
import numpy as np
from imageio import imread
from path import Path
import random
import os
import torch
import re

def load_as_float(path):
    return imread(path).astype(np.float32)


def read_pfm(path):
    """Read pfm file.

    Args:
        path (str): path to file

    Returns:
        tuple: (data, scale)
    """
    with open(path, "rb") as file:

        color = None
        width = None
        height = None
        scale = None
        endian = None

        header = file.readline().rstrip()
        if header.decode("ascii") == "PF":
            color = True
        elif header.decode("ascii") == "Pf":
            color = False
        else:
            raise Exception("Not a PFM file: " + path)

        dim_match = re.match(r"^(\d+)\s(\d+)\s$", file.readline().decode("ascii"))
        if dim_match:
            width, height = list(map(int, dim_match.groups()))
        else:
            raise Exception("Malformed PFM header.")

        scale = float(file.readline().decode("ascii").rstrip())
        if scale < 0:
            # little-endian
            endian = "<"
            scale = -scale
        else:
            # big-endian
            endian = ">"

        data = np.fromfile(file, endian + "f")
        shape = (height, width, 3) if color else (height, width)

        data = np.reshape(data, shape)
        data = np.flipud(data)

        return data


class SequenceFolder(data.Dataset):
    """A sequence data loader where the files are arranged in this way:
        root/scene_1/0000000.jpg
        root/scene_1/0000001.jpg
        ..
        root/scene_1/cam.txt
        root/scene_2/0000000.jpg
        .
        transform functions must take in a list a images and a numpy array (usually intrinsics matrix)
    """

    def __init__(self, root, seed=None, train=True, sequence_length=3, transform=None, skip_frames=1, dataset='kitti'):
        np.random.seed(seed)
        random.seed(seed)
        self.root = Path(root)
        scene_list_path = self.root/'train.txt' if train else self.root/'val.txt'
        self.scenes = [self.root/folder[:-1] for folder in open(scene_list_path)]
        self.transform = transform
        self.dataset = dataset
        self.k = skip_frames
        self.use_pfm = False # ! for depth-anything
        self.crawl_folders(sequence_length)

    def crawl_folders(self, sequence_length):
        # k skip frames
        sequence_set = []
        demi_length = (sequence_length-1)//2
        shifts = list(range(-demi_length * self.k, demi_length * self.k + 1, self.k))
        shifts.pop(demi_length)
        for scene in self.scenes:
            intrinsics = np.genfromtxt(scene/'cam.txt').astype(np.float32).reshape((3, 3))
            imgs = sorted(scene.files('*.jpg'))
            # imgs = sorted(scene.files('*.png'))
            # depth
            if (scene/'output_monodepth').exists():
                if not self.use_pfm: depths_gt = sorted((scene/'output_monodepth').listdir('*.png'))
                else: depths_gt = sorted((scene/'output_monodepth').listdir('*.pfm')) # ! 不同之处在这里
            else:
                depths_gt = None
            # pose
            poses = np.load(scene/'Transforms.npy').astype(np.float32) if os.path.exists(scene/'Transforms.npy') else None  # load pose
            
            # oflow
            if (scene/'depth_gt').exists():
                depths_gtt = sorted((scene/'depth_gt').listdir('*.npy'))
            else:
                depths_gtt = None
            

            if len(imgs) < sequence_length:
                continue
            for i in range(demi_length * self.k, len(imgs)-demi_length * self.k):
                sample = {'intrinsics': intrinsics, 'tgt': imgs[i], 'ref_imgs': [], 'ref_poses': [], 'others':{
                    'tgt_depth':depths_gt[i] if depths_gt is not None else None, 'ref_depths':[],
                    'tgt_depth_gt':depths_gtt[i] if depths_gtt is not None else None, 'ref_depths_gt':[]
                }}
                
                if poses is not None: pose_tgt,pose_tgt_inv = poses[i], np.linalg.inv(poses[i])
                for j in shifts:
                    sample['ref_imgs'].append(imgs[i+j])
                    if depths_gt is not None: sample['others']['ref_depths'].append(depths_gt[i+j])
                    if depths_gtt is not None: sample['others']['ref_depths_gt'].append(depths_gtt[i+j])
                    if poses is not None: sample['ref_poses'].append(np.linalg.inv(poses[i+j]) @ pose_tgt) # project points from tgt to refs
                sequence_set.append(sample)
        random.shuffle(sequence_set)
        self.samples = sequence_set

    def __getitem__(self, index):
        others = {}
        others['idx'] = index
        sample = self.samples[index]
        tgt_img = load_as_float(sample['tgt'])
        ref_imgs = [load_as_float(ref_img) for ref_img in sample['ref_imgs']]

        if sample['others']['tgt_depth'] is not None:
            if not self.use_pfm: others['tgt_depth'] = load_as_float(sample['others']['tgt_depth'])[None,...]/65535.0
            else: others['tgt_depth'] = 1/(read_pfm(sample['others']['tgt_depth'])[None,...]+22.2) # ! 不同之处在这里
            if not self.use_pfm: others['ref_depths'] = [load_as_float(t)[None,...]/65535.0 for t in sample['others']['ref_depths']]
            else: others['ref_depths'] = [1/(read_pfm(sample['others']['tgt_depth'])[None,...]+22.2) for t in sample['others']['ref_depths']]
        if sample['others']['tgt_depth_gt'] is not None:            
            # compute oflow
            tgt_depth_gt = np.load(sample['others']['tgt_depth_gt'])[None,...,0].astype(np.float32)
            ref_depths_gt = [np.load(t)[None,...:,0].astype(np.float32) for t in sample['others']['ref_depths_gt']]

            # 深度图是 [0,100]mm 因此归一化
            others['tgt_depth_gt'] = tgt_depth_gt/100
            others['ref_depths_gt'] = [t/100 for t in ref_depths_gt]
            
            tgt_depth_gt = tgt_depth_gt[:,:,:,0]
            _,h,w = tgt_depth_gt.shape
            oflows = []
            K, K_inv = sample['intrinsics'], np.linalg.inv(sample['intrinsics'])
            for tgt2ref_pose,ref_depth in zip(sample['ref_poses'],ref_depths_gt):
                u,v = np.meshgrid(np.linspace(0,w-1,w),np.linspace(0,h-1,h))
                oflow = np.array([u,v])
                p3d = np.array([u,v,np.ones_like(u)])
                p3d = K_inv @ p3d.reshape((3,-1))
                p3d = p3d * tgt_depth_gt.reshape((-1))
                p3d = tgt2ref_pose[:3,:3]@p3d + tgt2ref_pose[:3,3:] # projection
                p3d = p3d / p3d[-1]
                p3d = K @ p3d
                p3d = p3d.reshape((3,h,w))
                p2d = p3d[:2]
                oflow = p2d-oflow
                oflows.append(torch.from_numpy(oflow.astype(np.float32)).permute((1,2,0)))
            others['ref_oflows'] = oflows
        if self.transform is not None:
            imgs, intrinsics = self.transform([tgt_img] + ref_imgs, np.copy(sample['intrinsics']))
            tgt_img = imgs[0]
            ref_imgs = imgs[1:]
        else:
            intrinsics = np.copy(sample['intrinsics'])
            
        return tgt_img, ref_imgs, intrinsics, np.linalg.inv(intrinsics), sample['ref_poses'], others

    def __len__(self):
        return len(self.samples)


class SimCol3D(data.Dataset):
    """A sequence data loader where the files are arranged in this way:
        root/scene_1/0000000.jpg
        root/scene_1/0000001.jpg
        ..
        root/scene_1/cam.txt
        root/scene_2/0000000.jpg
        .
        transform functions must take in a list a images and a numpy array (usually intrinsics matrix)
    """

    def __init__(self, root, seed=None, train=True, sequence_length=3, transform=None, skip_frames=1, dataset='kitti'):
        self.use_pfm = True
        np.random.seed(seed)
        random.seed(seed)
        self.root = Path(root)
        scene_list_path = self.root/'train.txt' if train else self.root/'val.txt'
        self.scenes = [self.root/folder[:-1] for folder in open(scene_list_path)]
        self.transform = transform
        self.dataset = dataset
        self.k = skip_frames
        self.crawl_folders(sequence_length)

    def crawl_folders(self, sequence_length):
        # k skip frames
        sequence_set = []
        demi_length = (sequence_length-1)//2
        shifts = list(range(-demi_length * self.k, demi_length * self.k + 1, self.k))
        shifts.pop(demi_length)
        for scene in self.scenes:
            intrinsics = np.genfromtxt(scene/'cam.txt').astype(np.float32).reshape((3, 3))
            # imgs = sorted(scene.files('*.jpg'))
            imgs = sorted(scene.files('F*.png')) # ! SimCol3D
            # depth
            if (scene/'output_monodepth').exists():
                if self.use_pfm: depths_gt = sorted((scene/'output_monodepth').listdir('*.pfm'))
                else: depths_gt = sorted((scene/'output_monodepth').listdir('*.png'))
            else:
                depths_gt = None
            # pose
            poses = np.load(scene/'Transforms.npy').astype(np.float32) if os.path.exists(scene/'Transforms.npy') else None  # load pose
            
            # oflow
            if (scene/'depth_gt').exists():
                depths_gtt = sorted((scene/'depth_gt').listdir('*.npy'))
            else:
                depths_gtt = None
            

            if len(imgs) < sequence_length:
                continue
            for i in range(demi_length * self.k, len(imgs)-demi_length * self.k):
                sample = {'intrinsics': intrinsics, 'tgt': imgs[i], 'ref_imgs': [], 'ref_poses': [], 'others':{
                    'tgt_depth':depths_gt[i] if depths_gt is not None else None, 'ref_depths':[],
                    'tgt_depth_gt':depths_gtt[i] if depths_gtt is not None else None, 'ref_depths_gt':[]
                }}
                
                if poses is not None: pose_tgt,pose_tgt_inv = poses[i], np.linalg.inv(poses[i])
                for j in shifts:
                    sample['ref_imgs'].append(imgs[i+j])
                    if depths_gt is not None: sample['others']['ref_depths'].append(depths_gt[i+j])
                    if depths_gtt is not None: sample['others']['ref_depths_gt'].append(depths_gtt[i+j])
                    if poses is not None: sample['ref_poses'].append(np.linalg.inv(poses[i+j]) @ pose_tgt) # project points from tgt to refs
                sequence_set.append(sample)
        random.shuffle(sequence_set)
        self.samples = sequence_set

    def __getitem__(self, index):
        others = {}
        others['idx'] = index
        sample = self.samples[index]
        tgt_img = load_as_float(sample['tgt'])[...,:-1][:448,:448] # ! NOTE [H,W,4] -> [H,W,3] and crop to 448x448 裁切图片
        ref_imgs = [load_as_float(ref_img)[...,:-1][:448,:448] for ref_img in sample['ref_imgs']] # ! 裁切图片

        if sample['others']['tgt_depth'] is not None:
            if self.use_pfm: 
                others['tgt_depth'] = (1/(read_pfm(sample['others']['tgt_depth'])[None,...]+22.2))[:,:448,:448] # ! 不同之处在这里
                others['ref_depths'] = [(1/(read_pfm(sample['others']['tgt_depth'])[None,...]+22.2))[:,:448,:448] for t in sample['others']['ref_depths']]
            else: 
                others['tgt_depth'] = (load_as_float(sample['others']['tgt_depth'])[None,...]/65535.0)[:,:448,:448] # ! 裁切 Midas值
                others['ref_depths'] = [(load_as_float(t)[None,...]/65535.0)[:,:448,:448] for t in sample['others']['ref_depths']] # ! 裁切 Midas值
            
        if sample['others']['tgt_depth_gt'] is not None:            
            # compute oflow
            tgt_depth_gt = np.load(sample['others']['tgt_depth_gt'])[None,...].astype(np.float32)[:,:448,:448] # ! 裁切 真值
            ref_depths_gt = [np.load(t)[None,...].astype(np.float32)[:,:448,:448] for t in sample['others']['ref_depths_gt']] # ! 裁切 真值

            # 深度图是 [0,200]mm 因此归一化
            # 外面*100来展示，因此这里/200
            others['tgt_depth_gt'] = tgt_depth_gt/200
            others['ref_depths_gt'] = [t/200 for t in ref_depths_gt]
            
        #     _,h,w = tgt_depth_gt.shape
        #     oflows = []
        #     K, K_inv = sample['intrinsics'], np.linalg.inv(sample['intrinsics'])
        #     for tgt2ref_pose,ref_depth in zip(sample['ref_poses'],ref_depths_gt):
        #         u,v = np.meshgrid(np.linspace(0,w-1,w),np.linspace(0,h-1,h))
        #         oflow = np.array([u,v])
        #         p3d = np.array([u,v,np.ones_like(u)])
        #         p3d = K_inv @ p3d.reshape((3,-1))
        #         p3d = p3d * tgt_depth_gt.reshape((-1))
        #         p3d = tgt2ref_pose[:3,:3]@p3d + tgt2ref_pose[:3,3:] # projection
        #         p3d = p3d / p3d[-1]
        #         p3d = K @ p3d
        #         p3d = p3d.reshape((3,h,w))
        #         p2d = p3d[:2]
        #         oflow = p2d-oflow
        #         oflows.append(torch.from_numpy(oflow.astype(np.float32)).permute((1,2,0)))
        #     others['ref_oflows'] = oflows
        
        if self.transform is not None:
            imgs, intrinsics = self.transform([tgt_img] + ref_imgs, np.copy(sample['intrinsics']))
            tgt_img = imgs[0]
            ref_imgs = imgs[1:]
        else:
            intrinsics = np.copy(sample['intrinsics'])
            
        return tgt_img, ref_imgs, intrinsics, np.linalg.inv(intrinsics), sample['ref_poses'], others

    def __len__(self):
        return len(self.samples)
