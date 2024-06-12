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
sys.path.append('/home/bai_gairui/seg_3d/model2/segmentation_3d')
from config import config as cfg 

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
        self.train_root = f'/home/bai_gairui/seg_3d/model2/processing_data_1/{mode}/'
        self.folder_list = [self.train_root + folder for folder in os.listdir(self.train_root)]
        
    def __len__(self):
        return len(self.folder_list)
    
    def __getitem__(self, index):
        step_w , step_h, step_d = 16, 16, 16      
        feature_file = self.folder_list[index] + "/feature.npy"
        label_file = self.folder_list[index] + "/label.npy"
        vertice_file = self.folder_list[index] + "/vertices.csv" if cfg.resolution == 64 else self.folder_list[index] + "/vertices_128.csv"
        vertice = pd.read_csv(vertice_file)
        vertice = vertice[vertice["count"] > 0]
        vertice.index = range(len(vertice))
        num = np.random.randint(0, len(vertice))
        i, j , k = vertice.loc[num, "i"], vertice.loc[num, "j"], vertice.loc[num, "k"]
        imgs = np.load(feature_file)
        imgs = imgs.astype("float32")
        imgs = imgs[step_h * i : step_h * i + cfg.resolution, step_w * j : step_w * j + cfg.resolution, step_d * k : step_d * k + cfg.resolution, :]
        label = torch.from_numpy(np.load(label_file))
        label = label[step_h * i : step_h * i + cfg.resolution, step_w * j : step_w * j + cfg.resolution, step_d * k : step_d * k + cfg.resolution]
        
        label[label == 4] = 3 
        mask = imgs.sum(-1) > 0
        for k in range(4):
            x = imgs[..., k]  
            y = x[mask]
            x[mask] -= y.mean()
            x[mask] /= (y.std() + 1e-5)
            imgs[..., k] = x
        feature = torch.from_numpy(imgs)
        feature = rearrange(feature, "h w d c -> c h w d")
        return feature, label


# class BratsMatrixTestDataset(Dataset):
#     def __init__(self, mode = "test"):
#         self.train_root = f'/home/kathy/model2/processing_data_1/{mode}/'
#         self.folder_list = [self.train_root + folder for folder in os.listdir(self.train_root)]
#         self.cfg = Config()
#     def __len__(self):
#         return len(self.folder_list)
    
#     def __getitem__(self, index):
#         feature_file = self.folder_list[index] + "/feature.npy"
#         label_file = self.folder_list[index] + "/label.npy"
        
#         imgs = np.load(feature_file)
#         imgs = imgs.astype("float32")
#         mask = imgs.sum(-1) > 0
#         for k in range(4):
#             x = imgs[..., k]  
#             y = x[mask]
#             x[mask] -= y.mean()
#             x[mask] /= y.std()
#             imgs[..., k] = x
#         h_start = (self.cfg.data_h - self.cfg.resolution) // 2
#         w_start = (self.cfg.data_w - self.cfg.resolution) // 2
#         d_start = (self.cfg.data_d - self.cfg.resolution) // 2
#         h_end = h_start + self.cfg.resolution
#         w_end = w_start + self.cfg.resolution
#         d_end = d_start + self.cfg.resolution
#         # imgs = (imgs - imgs.min()) / (imgs.max() - imgs.min())
#         imgs = torch.from_numpy(imgs)
#         imgs = rearrange(imgs, "h w d c -> c h w d")
#         imgs = imgs[:, h_start : h_end, w_start : w_end, d_start : d_end]
        
#         label = torch.from_numpy(np.load(label_file))
#         label = label[h_start : h_end, w_start : w_end, d_start : d_end]
        
#         label[label == 4] = 3 
#         label = label.to(torch.int64)
#         return self.folder_list[index], imgs, label

class BratsMatrixTestDataset(Dataset):
    def __init__(self, mode = "test"):
        self.train_root = f'/home/bai_gairui/seg_3d/model2/processing_data_2/{mode}/'
        self.folder_list = [self.train_root + folder for folder in os.listdir(self.train_root)]
        
    def __len__(self):
        return len(self.folder_list)
    
    def __getitem__(self, index):
        step_w , step_h, step_d = 16, 16, 16      
        feature_file = self.folder_list[index] + "/feature.npy"
        label_file = self.folder_list[index] + "/label.npy"
        vertice_file = self.folder_list[index] + "/vertices_128.csv"
        vertice = pd.read_csv(vertice_file)
        vertice = vertice[vertice["count"] > 0]
        vertice.index = range(len(vertice))
        num = vertice["count"].argmax()
        i, j , k = vertice.loc[num, "i"], vertice.loc[num, "j"], vertice.loc[num, "k"]
        imgs = np.load(feature_file)
        imgs = imgs.astype("float32")
        imgs = imgs[step_h * i : step_h * i + cfg.resolution, step_w * j : step_w * j + cfg.resolution, step_d * k : step_d * k + cfg.resolution, :]
        label = torch.from_numpy(np.load(label_file))
        label = label[step_h * i : step_h * i + cfg.resolution, step_w * j : step_w * j + cfg.resolution, step_d * k : step_d * k + cfg.resolution]
        
        label[label == 4] = 3 
        mask = imgs.sum(-1) > 0
        for k in range(4):
            x = imgs[..., k]  
            y = x[mask]
            x[mask] -= y.mean()
            x[mask] /= (y.std() + 1e-5)
            imgs[..., k] = x
        feature = torch.from_numpy(imgs)
        feature = rearrange(feature, "h w d c -> c h w d")
        return self.folder_list[index], feature, label

    
    
def get_data_loader(batch_size = 8, mode = "train"):
    bmtd = BratsMatrixTrainDataset() if mode == "train" else BratsMatrixTestDataset() 
    
    return DataLoader(bmtd, batch_size=batch_size, shuffle=True if mode == "train" else False)





if __name__ == "__main__":
    import time 
    
    data = BratsMatrixTrainDataset()
    X, y = next(iter(
        data
    ))
    print(X.shape)
    print(y.shape)