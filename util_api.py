# load model

device = 'cuda'
resnet_layers = 18
with_pretrain = True

import models
import torch
import cv2


def load_model(poseNetPath,dispNetPath):
    # create model
    print("=> creating model")
    disp_net = models.DispResNet(resnet_layers, with_pretrain).to(device)
    pose_net = models.PoseResNet(18, with_pretrain).to(device)

    # load weight
    print("=> load weight")
    weights = torch.load(poseNetPath)
    pose_net.load_state_dict(weights['state_dict'], strict=False)
    weights = torch.load(dispNetPath)
    disp_net.load_state_dict(weights['state_dict'], strict=False)

    # disp_net = torch.nn.DataParallel(disp_net)
    # pose_net = torch.nn.DataParallel(pose_net)
    
    return disp_net,pose_net

def load_dispNet(dispNetPath):
    try:
        disp_net = models.DispResNet(18, with_pretrain).to(device)
        weights = torch.load(dispNetPath)
        disp_net.load_state_dict(weights['state_dict'], strict=False)
        # disp_net = torch.nn.DataParallel(disp_net)
    except:
        disp_net = models.DispResNet(50, with_pretrain).to(device)
        weights = torch.load(dispNetPath)
        disp_net.load_state_dict(weights['state_dict'], strict=False)
    
    return disp_net

def load_denseNet(denseNetPath):
    disp_net = models.PTModel().to(device)
    weights = torch.load(denseNetPath)
    disp_net.load_state_dict(weights['state_dict'], strict=False)
    # disp_net = torch.nn.DataParallel(disp_net)
    
    return disp_net

# load data

from datasets.sequence_folders import load_as_float
import custom_transforms
import numpy as np

normalize = custom_transforms.Normalize(mean=[0.45, 0.45, 0.45],
                                        std=[0.225, 0.225, 0.225])
train_transform = custom_transforms.Compose([
    # custom_transforms.RandomHorizontalFlip(),
    # custom_transforms.RandomScaleCrop(),
    custom_transforms.ArrayToTensor(),
    # normalize
])

def load_data(tgt_img_path,ref_imgs_paths, train_transform=train_transform):
    # tgt_img,_ = train_transform([load_as_float(tgt_img_path)],np.identity(3))
    # ref_imgs = [ train_transform(load_as_float(ref_img_path),np.identity(3))[0] for ref_img_path in ref_imgs_paths ]
    tgt_img,ref_imgs = load_as_float(tgt_img_path), [load_as_float(ref_img_path) for ref_img_path in ref_imgs_paths]
    imgs, intrinsics = train_transform([tgt_img] + ref_imgs, np.identity(3))
    tgt_img = imgs[0]
    ref_imgs = imgs[1:]

    tgt_img = tgt_img.unsqueeze(0).to(device)
    ref_imgs = [ref_img.unsqueeze(0).to(device) for ref_img in ref_imgs]
    
    return tgt_img,ref_imgs


# infer

from train import compute_depth,compute_pose_with_inv

def run_model(pose_net,disp_net,tgt_img,ref_imgs):
    pose_,pose_inv_ = compute_pose_with_inv(pose_net,tgt_img,ref_imgs)
    tgt_depth_,ref_depths_ = compute_depth(disp_net,tgt_img,ref_imgs)

    pose = [t.detach().cpu().numpy() for t in pose_]
    tgt_depth = tgt_depth_[0].detach().cpu().numpy()[0]
    ref_depths = [ref_depth[0].detach().cpu().numpy()[0] for ref_depth in ref_depths_]
    
    return pose,tgt_depth,ref_depths


def infer2(poseNetPath,dispNetPath, tgt_img_path,ref_imgs_paths):
    disp_net,pose_net = load_model(poseNetPath,dispNetPath)
    tgt_img,ref_imgs = load_data(tgt_img_path,ref_imgs_paths)
    
    pose,tgt_depth,ref_depths = run_model(pose_net,disp_net,tgt_img,ref_imgs)
    
    return pose,tgt_depth,ref_depths

@torch.no_grad()
def infer(model_path, filename):
    
    disp_net = load_dispNet(model_path)
    
    tgt_img_path = filename
    tgt_img_tmp = cv2.imread(tgt_img_path).astype(np.float32)
    h,w,_ = tgt_img_tmp.shape
    
    tgt_img_tmp = cv2.resize(tgt_img_tmp, (512,512))
    print(tgt_img_tmp.shape)

    tgt_img,_ = train_transform([tgt_img_tmp],np.identity(3))

    tgt_img = tgt_img[0].unsqueeze(0).to(device)
    with torch.no_grad():
        disp = disp_net(tgt_img)
    # disp2 = min_max_norm(disp[0][0][0].detach().cpu().numpy())
    depth = 1/(disp[0][0][0].detach().cpu().numpy())
    
    depth = cv2.resize(depth, (w,h))
    return depth

@torch.no_grad()
def infer_model(model, filename, resize_shape=None):
    disp_net = model
    
    tgt_img_path = filename
    tgt_img_tmp = cv2.imread(tgt_img_path).astype(np.float32)
    h,w,_ = tgt_img_tmp.shape
    
    if resize_shape is None: tgt_img_tmp = tgt_img_tmp
    else: tgt_img_tmp = cv2.resize(tgt_img_tmp, resize_shape)
    # print(tgt_img_tmp.shape)

    tgt_img,_ = train_transform([tgt_img_tmp],np.identity(3))

    tgt_img = tgt_img[0].unsqueeze(0).to(device)
    with torch.no_grad():
        disp = disp_net(tgt_img)
    # disp2 = min_max_norm(disp[0][0][0].detach().cpu().numpy())
    depth = 1/(disp[0][0][0].detach().cpu().numpy())
    
    depth = cv2.resize(depth, (w,h))
    return depth


def save_pointcloud(pointcloud, save_file):
    pointclouds = [pointcloud]
    with open(save_file,'w') as f:
        for pointcloud in pointclouds:
            for t in pointcloud.reshape((3,-1)).T:
                line = "{},{},{}\n".format(t[0],t[1],t[2])
                f.write(line)
    return 

def unproj(K,depth,img):
    # h,w,_ = img.shape
    h,w = depth.shape
    x,y = np.meshgrid(np.linspace(0,w-1,w), np.linspace(0,h-1,h))
    x,y = x[None,...], y[None,...]
    z = np.ones((1,h,w))
    
    coor = np.concatenate((x,y,z),axis=0).reshape((3,-1))
    
    K_inv = np.linalg.inv(K)
    coor_camera = K_inv @ coor # 相机坐标
    
    coor_camera = coor_camera.reshape((3,h,w))
    coor_camera[0,...] *= depth
    coor_camera[1,...] *= depth
    coor_camera[2,...] *= depth
    
    return coor_camera
