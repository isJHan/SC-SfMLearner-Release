import numpy as np
import cv2
from util_api import load_dispNet, infer_model
from path import Path
from tqdm import tqdm


model_path = ""
model = load_dispNet(model_path)
max_value = 10.0



           
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
    save_root_path = Path("/home/jiahan/jiahan/datasets/SimCol/Depth-Anything/inv_22-2_5") # 1/(y+22.2) train 5 epochs
    save_root_path.makedirs_p()
    img_root_path = Path("/home/jiahan/jiahan/datasets/SimCol/SimCol3D4SC_Depth")
    for scene in scenes:
        print("=> processing scene", scene)
        save_path = save_root_path/scene
        save_path.makedirs_p()
        print(save_path)
        print(img_root_path/scene)
        
        files = sorted((img_root_path/scene).listdir("F*.png"))
        for file in tqdm(files):
            name = file.split('/')[-1][:-4]
            output = infer_model(model, file)
            output = (65535*output/max_value).astype(np.uint16)
            # write_pfm(save_path/name+'_depth.pfm', output)
            # cv2.imwrite(save_path/name+'_depth.png', (65535*(output-output.min())/(output.max()-output.min())).astype(np.uint16))
            cv2.imwrite(save_path/name+'_depth.png', (65535*(output-output.min())/(output.max()-output.min())).astype(np.uint16))
        
