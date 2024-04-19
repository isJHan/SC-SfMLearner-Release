import sys
sys.path.append("..")
sys.path.append(".")
sys.path.append("/home/jiahan/jiahan/codes/segment-anything")
from segment_anything import sam_model_registry, SamPredictor
import cv2
import numpy as np


def get_model(path=None):
    sam_checkpoint = "/home/jiahan/jiahan/codes/segment-anything/checkpoints/sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    predictor = SamPredictor(sam)
    return predictor

class Seg_Points:
    def __init__(self, model_path=None) -> None:
        self.predictor = get_model(model_path)
        self.image = None
        self.points = []
        self.label = []
    
    def set_image(self, filename):
        image = cv2.imread(filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.image = image
        self.predictor.set_image(image)
        
    def add_point(self, point):
        """_summary_

        Args:
            point (np.array): [2,]
        """
        self.points.append(point)
        self.label.append(1)
        
    def seg(self):
        input_point = np.array(self.points)
        input_label = np.array(self.label)

        masks, _, _ = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False,
        )
        return masks[0]

    def clear(self):
        self.points = []
        self.label = []
        self.image = None
        
    def clear_points(self):
        self.points = []
        self.label = []
        

import numpy as np
import cv2
from path import Path

import models
import torch
from util_api import infer, infer_model, unproj, save_pointcloud, load_dispNet

class SegMeasureAux:
    '''
    0. call load_model to load the model first
    1. give a img
    2. give the selected points
    3. call measure() function to get the size
    '''
    def __init__(self, sam_path, sc_depth_path, K):
        self.seg_aux = Seg_Points(sam_path)
        self.sc_depth_path = sc_depth_path
        self.K = K # intrics
    
    def __unproj(self):
        K = self.K
        depth = self.depth
        
        # h,w,_ = img.shape
        h,w = self.depth.shape
        x,y = np.meshgrid(np.linspace(0,w-1,w), np.linspace(0,h-1,h))
        x,y = x[None,...], y[None,...]
        z = np.ones((1,h,w))

        # print(np.concatenate((x,y,z),axis=0).shape)
        coor = np.concatenate((x,y,z),axis=0).reshape((3,-1)) # [3,h,w] -> [3,-1]

        K_inv = np.linalg.inv(K)
        coor_camera = K_inv @ coor # 相机坐标

        coor_camera = coor_camera.reshape((3,h,w))
        coor_camera[0,...] *= self.depth
        coor_camera[1,...] *= self.depth
        coor_camera[2,...] *= self.depth
    
        return coor_camera

    def __read_img(filename):
        return cv2.imread(filename)

    def __load_sam(self):
        "load the sam model to [self.sam_model]"
        
        pass

    def __load_sc(self):
        "load the sc model to [self.sc_model]"
        self.sc_model = load_dispNet(self.sc_depth_path)
        

    def load_model(self):
        print("==> loading sam")
        self.__load_sam()
        print("==> loading sc depth")
        self.__load_sc()

    def __infer_depth(self, filename):
        "infer the depth map of input_img"
        self.depth = infer_model(self.sc_model, filename, resize_shape=(1024,1024)).copy()
        return self.depth
        pass
    
    def set_image(self, img_path=None):
        "1. give a img"
        self.seg_aux.clear()
        
        if img_path is not None: self.seg_aux.set_image(img_path)
        self.depth = self.__infer_depth(img_path)

        self.points_c = self.__unproj()
        pass

    def add_point(self, point):
        self.seg_aux.add_point(point)
        
    def measure(self):
        """
        measure main. 
        @return: radius of the masked area
        """
        depth = self.depth
        mask = self.seg_aux.seg()
        """测量过程实现"""
        mask_3d = np.stack((mask,mask,mask))
        masked_points_c = self.points_c[np.where(mask_3d==1)].reshape((3,-1))
        center = np.mean(masked_points_c, axis=1)[...,None]
        diff_points = masked_points_c - center
        radius = np.mean( np.linalg.norm(diff_points, axis=0)  )


        return radius

    def get_depth(self): return self.depth

# ------------- aux func --------------


def mask_to_rgb(mask):
    """_summary_

    Args:
        mask (np.array): [h,w]
    """
    ret = np.array([mask,mask,mask]).astype(np.uint8).transpose((1,2,0))
    tmp = mask==1
    ret[tmp] = np.array([30, 144, 255])
    
    return ret
    
    
def mask_on_image(image, mask):
    colored_mask = mask_to_rgb(mask)
    
    return cv2.addWeighted(image,1,colored_mask,0.6,1)

# 创建回调函数来处理鼠标事件
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # 当鼠标左键按下时
        print("click: ", x, y)
        seg_measure.add_point(np.array([x,y]))
        mask = seg_measure.seg_aux.seg()
        cv2.circle(image,(x,y),5,(0,255,0),-1)
        cv2.imshow(name, mask_on_image(image,mask))
        
        param['clicked_point'].append([x, y])
        param['mask'] = mask

# -----------------------------------------------------------------
def save_value(values, scene, save_file):
    import time
    weighted_value = sum(values)/len(values)
    with open(save_file, 'a') as f:
        f.write("==========================\n")
        f.write(f"=========== {time.ctime()} =============\n")
        f.write("scene is "+scene+"\n")
        f.write("=> values: "+str(values)+"\n")
        f.write("=> radius is "+str(weighted_value)+"\n")
        f.write("===========================")
    
        

if __name__=="__main__":
    scenes = [
        "undist_20240313_175728_测大小-30mm-带蒂",
        "undist_20240313_175415_测大小-18mm-带蒂",
        "undist_20240313_174853_测大小-15mm-带蒂",
        "undist_20240313_181002-测大小-12mm-原装2号-带蒂",
        "undist_20240313_180538-测大小-11mm-原装1号",
        "undist_20240313_181514-测大小-12mm-原装4号-扁平",
        "undist_20240313_181228-测大小-8mm-原装3号"
    ]
    i = 1
    save_file = "measure_seg.txt"
    img_root_path = Path(f"/home/jiahan/jiahan/datasets/Ours/datasets/{scenes[i]}")
    # img_root_path = Path("/home/jiahan/jiahan/codes/segment-anything/notebooks/images_tmp")
    ext = "jpg"
    
    sam_path = None
    sc_depth_path = "/home/jiahan/jiahan/checkpoints/SC_Depth_on_C3VD/scenes_all_lossMidas_align/24-03-02-20:04/dispnet_5_checkpoint.pth.tar"
    fx,fy, cx,cy = 383.6930755563, 383.7529328059, 339.5271329985, 271.8234458423
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])

    print("=> loading model")
    seg_measure = SegMeasureAux(sam_path, sc_depth_path, K)
    seg_measure.load_model()

    uvs = []
    files = sorted(img_root_path.listdir(f"*.{ext}"))
    for f in files: print(f)
    
    values_cur = []
    for i in range(0, len(files), 20):
        file = files[i]
        print("\n--------------------------------")
        
        name = file.split("/")[-1][:-4]
        
        clicked_points = []
        seg_measure.set_image(file)
        
        # 创建一个窗口并加载图像
        image = cv2.imread(file)
        cv2.imshow(name, image)
        depth_map = seg_measure.get_depth()
        cv2.imshow(name+"depth", depth_map)

        # 将鼠标事件回调函数与窗口关联
        click_info = {'clicked_point': [], 'mask': None}
        cv2.setMouseCallback(name, mouse_callback, click_info)

        # 等待用户关闭窗口
        cv2.imshow(name, image)
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):  # 如果按下 ' ' 键，退出循环
                break
            elif key == ord('r'): # 如果按下r, 重新标记点
                seg_measure.seg_aux.clear_points()
                image = cv2.imread(file)
                cv2.imshow(name, image)
                print("------------ delete above---------------")
            elif key == ord('m'): # 'm' 开始测量，输出结果
                value = seg_measure.measure()
                values_cur.append(value)
                print(name, "=> radius is", value)

        # 获取鼠标点击坐标并显示
        # clicked_point = click_info['clicked_point']
        # if clicked_point is not None:
        #     uvs.append(clicked_point)
        #     print(f"坐标：{clicked_point}")
        
        
        cv2.destroyAllWindows()
        seg_measure.seg_aux.clear()
        
    save_value(values_cur, img_root_path.split('/')[-1], save_file)
    print(values_cur)
    print("!!! weighted radius is ", sum(values_cur)/len(values_cur))
    print("\n\n\n")


