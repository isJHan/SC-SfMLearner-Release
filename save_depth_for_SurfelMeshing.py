import numpy as np
import cv2
from util_api import load_dispNet, infer_model, infer, load_as_float, train_transform
from path import Path
from tqdm import tqdm
import torch


model_path = "/home/jiahan/jiahan/checkpoints/SC_Depth_on_SimCol3D/depth-anyting_scenes_all_SimCol3D/24-05-01-23:51/dispnet_4_checkpoint.pth.tar"
model = load_dispNet(model_path).to('cuda')
max_value = 1.0



           
def infer_simcol():
    scenes = [
        "SyntheticColon_I/Frames_S15",
        "SyntheticColon_I/Frames_S7",
        "SyntheticColon_I/Frames_S12",
        "SyntheticColon_I/Frames_S13",
        "SyntheticColon_I/Frames_S1",
        "SyntheticColon_I/Frames_S4",
        "SyntheticColon_I/Frames_S3",
        "SyntheticColon_I/Frames_S2",
        "SyntheticColon_I/Frames_S11",
        "SyntheticColon_I/Frames_S14",
        "SyntheticColon_I/Frames_S9",
        "SyntheticColon_I/Frames_S6",
        "SyntheticColon_I/Frames_S8",
        "SyntheticColon_I/Frames_S5",
        "SyntheticColon_I/Frames_S10",
        "SyntheticColon_II/Frames_B12",
        "SyntheticColon_II/Frames_B2",
        "SyntheticColon_II/Frames_B1",
        "SyntheticColon_II/Frames_B7",
        "SyntheticColon_II/Frames_B13",
        "SyntheticColon_II/Frames_B10",
        "SyntheticColon_II/Frames_B4",
        "SyntheticColon_II/Frames_B8",
        "SyntheticColon_II/Frames_B14",
        "SyntheticColon_II/Frames_B5",
        "SyntheticColon_II/Frames_B9",
        "SyntheticColon_II/Frames_B6",
        "SyntheticColon_II/Frames_B3",
        "SyntheticColon_II/Frames_B15",
        "SyntheticColon_II/Frames_B11",
        "SyntheticColon_III/Frames_O3",
        "SyntheticColon_III/Frames_O2",
        "SyntheticColon_III/Frames_O1",
    ]
    save_root_path = Path("/home/jiahan/jiahan/datasets/SimCol/SC/depth-anything/inv_22-2_5") # 1/(y+22.2) train 5 epochs
    save_root_path.makedirs_p()
    img_root_path = Path("/home/jiahan/jiahan/datasets/SimCol/SimCol3D4SC_Depth")
    for scene in scenes:
        print("=> processing scene", scene)
        save_path = save_root_path/scene/'depth'
        save_path.makedirs_p()
        print(save_path)
        print(img_root_path/scene)
        
        lines_to_write = ""
        time = 1.0/60 
        files = sorted((img_root_path/scene).listdir("F*.png"))
        for i,file in enumerate(tqdm(files)):
            name = file.split('/')[-1][:-4]
            input = load_as_float(file)[:448,:448,:3]
            input,_ = train_transform([input],np.identity(3))
            input = input[0].unsqueeze(0).to('cuda')
            with torch.no_grad():
                output = model(input)
            output = output[0][0][0].detach().cpu().numpy()
            output = 1/output
            # output = infer_model(model, file)
            output = (15000*output/max_value).astype(np.uint16)
            cv2.imwrite(save_path/f"{i:05d}.depth.png", output)
            lines_to_write = lines_to_write + f"{time*i} image/FrameBuffer_{i:04d}.png {time*i} depth/{i:05d}.depth.png\n"
       # 生成scene的txt文件
        tmp_path = (save_root_path/'txt_files')
        tmp_path.makedirs_p()
        tmp = scene.split('/')
        print(save_root_path/'txt'/f"{tmp[0]}-{tmp[1]}.txt")
        with open(tmp_path/f"{tmp[0]}-{tmp[1]}.txt", 'w+') as f:
            f.write(lines_to_write)

if __name__=="__main__":
    infer_simcol()