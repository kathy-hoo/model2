import torch 
import torch.nn as nn
from torch.nn import functional as F  
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np 
from einops import rearrange 
import nibabel as nib 
import sys 
import pandas as pd 
import json 
sys.path.append('/home/bai_gairui/seg_3d/model2/segmentation_3d')
from config import config as cfg 

def nib_load(file_name):
    if not os.path.exists(file_name):
        print('Invalid file name, can not find the file!')

    proxy = nib.load(file_name)
    data = proxy.get_data()
    proxy.uncache()
    return data

class Config():
    def __init__(self) -> None:
        self.data_h, self.data_w, self.data_d = 240, 240, 155
        self.resolution = 128
class BratsMatrixTrainDataset(Dataset):
    def __init__(self, mode = "train"):
        with open("/home/bai_gairui/seg_3d/model2/udc/info.json", 'r', encoding="utf8") as f:
            res = json.loads(f.read())
        self.folder_list = ["/home/bai_gairui/seg_3d/model2/udc/vertices/" + folder for folder in res[mode]]
        self.mode = mode
    def __len__(self):
        return len(self.folder_list)
    
    def __getitem__(self, index):
        step_w , step_h, step_d = 16, 16, 16      
        feature_file = self.folder_list[index].replace("vertices", "images").replace("csv", "npy")
        label_file = self.folder_list[index].replace("vertices", "labels_small").replace("csv", "npy")
        # print(label_file)
        vertice_file = self.folder_list[index]
        vertice = pd.read_csv(vertice_file)
        vertice = vertice[vertice["count"] > 0]
        vertice.index = range(len(vertice))
        if self.mode == "train":
            num = np.random.randint(0, len(vertice))
        else:
            num = vertice["count"].argmax()
        i, j , k = vertice.loc[num, "i"], vertice.loc[num, "j"], vertice.loc[num, "k"]
        imgs = np.load(feature_file)
        imgs = imgs.astype("float32")
        imgs = imgs[step_h * i : step_h * i + cfg.resolution, step_w * j : step_w * j + cfg.resolution, step_d * k : step_d * k + cfg.resolution]
        label = torch.from_numpy(np.load(label_file))
        label = label[step_h * i : step_h * i + cfg.resolution, step_w * j : step_w * j + cfg.resolution, step_d * k : step_d * k + cfg.resolution]
        label[label == 3] = 1
        label[label == 4] = 2
        label[label == 5] = 3
        imgs = (imgs - imgs.mean()) / (imgs.std() + 1e-5)
        feature = torch.from_numpy(imgs)
        return feature.unsqueeze(0), label



    
    
def get_data_loader(batch_size = 8, mode = "train"):
    bmtd = BratsMatrixTrainDataset(mode) 
    
    return DataLoader(bmtd, batch_size=batch_size, shuffle=True if mode == "train" else False)





if __name__ == "__main__":
    import time 
    
    data = DataLoader(BratsMatrixTrainDataset(), batch_size=1, shuffle=True)
    print(len(data))
    loader = iter(data)
    X, y = next(loader)
    print(X.shape, y.shape)