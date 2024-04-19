import cv2
import numpy as np
from path import Path
import time

from util_api import infer, infer_model, unproj, save_pointcloud, load_dispNet


fx,fy = 719.7949119, 719.8346096
cx,cy = 575.49009715, 507.60611057
intrics = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0,0,1]
])


# 鼠标点击事件回调函数
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("鼠标左键点击位置坐标：({}, {})".format(x, y))
        param['clicked_point'].append([x, y])
        cv2.circle(image,(x,y),5,(0,255,0),-1)
        cv2.imshow(name, image)
        
        if len(param['clicked_point']) >= 2:
            depth = infer_model(model, file, resize_shape=(1024,1024))

            point_cloud = unproj(intrics, depth, None)
            # print(point_cloud.shape)
            # save_pointcloud(point_cloud, save_file)
            p1x,p1y = param['clicked_point'][-2][0], param['clicked_point'][-2][1]
            c1,c2 = point_cloud[:,p1y,p1x], point_cloud[:,y,x]
            # print(c1,"\n", c2)

            distance = np.linalg.norm(c1-c2)
            print("=> distance is ", distance)
            
            param['distances'].append(distance)
            
        
    elif event == cv2.EVENT_RBUTTONDOWN:
        print("鼠标右键点击位置坐标：({}, {})".format(x, y))
    elif event == cv2.EVENT_MBUTTONDOWN:
        print("鼠标中键点击位置坐标：({}, {})".format(x, y))





if __name__ == "__main__":
    scenes = [
        "undist_20240313_175728_测大小-30mm-带蒂",
        "undist_20240313_175415_测大小-18mm-带蒂",
        "undist_20240313_174853_测大小-15mm-带蒂",
        "undist_20240313_181002-测大小-12mm-原装2号-带蒂",
        "undist_20240313_180538-测大小-11mm-原装1号",
        "undist_20240313_181514-测大小-12mm-原装4号-扁平",
        "undist_20240313_181228-测大小-8mm-原装3号"
    ]
    scene_name = scenes[6]
    save_file = f"./measure.txt"
    
    print("=> measuring scene is ", scene_name)
    
    # C3VD训练过的
    # model_path = "/home/jiahan/jiahan/checkpoints/SC_Depth_on_C3VD/scenes_all_lossMidas_align/24-03-02-20:04/dispnet_5_checkpoint.pth.tar"
    model_path = "/home/jiahan/jiahan/checkpoints/SC_Depth_on_ours2403/24-04-01-20:17/dispnet_10_checkpoint.pth.tar" # 训练过的
    print("=> loading model", model_path)
    model = load_dispNet(model_path)
    
    with open(save_file, 'a+') as f:
        f.write(f"--------------- {time.ctime()} ---------------\n")
    for scene_name in scenes:
        img_root_path = Path(f"/home/jiahan/jiahan/datasets/Ours/datasets/{scene_name}")
        files = sorted(img_root_path.listdir("*.jpg"))

        numbers = 0
        distances = []
        for i in range(0,len(files),20):
            numbers += 1

            file = files[i]
            print("\n\n--------------------------------")

            name = file.split("/")[-1][:-4]

            clicked_points = []

            # 创建一个窗口并加载图像
            image = cv2.imread(file)
            cv2.imshow(name, image)

            # 将鼠标事件回调函数与窗口关联
            click_info = {'clicked_point': [], 'distances': distances}
            cv2.setMouseCallback(name, mouse_callback, click_info)

            # 等待用户关闭窗口
            cv2.imshow(name, image)
            while True:
                key = cv2.waitKey(1) & 0xFF
                if key == ord(' '):  # 如果按下 ' ' 键，退出循环
                    break

            # 获取鼠标点击坐标并显示
            # clicked_point = click_info['clicked_point']
            # if clicked_point is not None:
            #     uvs.append(clicked_point)
            #     print(f"坐标：{clicked_point}")

            cv2.destroyAllWindows()

        print("=> all distance is ", distances)
        print("=> avg distance is ", sum(distances)/len(distances))
        print("=> saving to ", save_file)
        with open(save_file, 'a+') as f:
            
            f.write(scene_name+"\n")
            f.write(str(distances)+"\n")
        with open(save_file, 'a+') as f:
            f.write("距离是 "+str(sum(distances)/len(distances))+"\n\n")
            f.write("\n")
            
        print("Done")




"""
30mm        0.08395
18mm        0.07599
15mm        0.12315
12mm        0.07944
11mm        0.05507
12mm        0.09387
8mm         0.09485
"""
