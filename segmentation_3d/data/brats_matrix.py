import torch 
import torch.nn as nn
from torch.nn import functional as F  
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np 
from einops import rearrange 
import nibabel as nib 
import sys 
sys.path.append('..')
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
        self.data_h, self.data_w, self.data_d = 240, 240, 150
        self.resolution = 128
class BratsMatrixTrainDataset(Dataset):
    def __init__(self):
        self.train_root = '/home/kathy/model2/processing_data/train/'
        self.folder_list = [self.train_root + folder for folder in os.listdir(self.train_root)]
        
    def __len__(self):
        return len(self.folder_list)
    
    def __getitem__(self, index):
        feature_file = self.folder_list[index] + "/feature.npy"
        label_file = self.folder_list[index] + "/label.npy"
        
        imgs = np.load(feature_file)
        imgs = imgs.astype("float32")
        mask = imgs.sum(-1) > 0
        for k in range(4):
            x = imgs[..., k]  
            y = x[mask]
            x[mask] -= y.mean()
            x[mask] /= y.std()
            imgs[..., k] = x
        feature = torch.from_numpy(imgs)
        feature = rearrange(feature, "h w d c -> c h w d")
        label = torch.from_numpy(np.load(label_file))
        label[label == 4] = 3 
        return feature, label

class BratsMatrixTestDataset(Dataset):
    def __init__(self) :
        self.root = '/home/kathy/dataset/BraTS2020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/'
        self.flist = '/home/kathy/model2/test/test.txt'
        self.folder_list = []
        with open(self.flist, 'r', encoding="utf8") as f:
            self.folder_list = f.read().split("\n")
        
        self.cfg = Config()
        
    def __len__(self):
        return len(self.folder_list)
    
    def __getitem__(self, index):
        modalities = ["flair", "t1", "t1ce", "t2"]
                
        folder = self.folder_list[index]
        imgs = []
        for modality in modalities:
            path = self.root + folder + "/" + folder + "_" + modality + ".nii" 
            img = np.asanyarray(nib_load(path), dtype='float32', order='C')
            img = img[..., None]
            imgs.append(img)
        imgs = np.concatenate(imgs, axis=-1)
        imgs = imgs.astype("float32")
        mask = imgs.sum(-1) > 0
        for k in range(4):
            x = imgs[..., k]  
            y = x[mask]
            x[mask] -= y.mean()
            x[mask] /= y.std()
            imgs[..., k] = x
        h_start = (self.cfg.data_h - self.cfg.resolution) // 2
        w_start = (self.cfg.data_w - self.cfg.resolution) // 2
        d_start = (self.cfg.data_d - self.cfg.resolution) // 2
        h_end = h_start + self.cfg.resolution
        w_end = w_start + self.cfg.resolution
        d_end = d_start + self.cfg.resolution
        # imgs = (imgs - imgs.min()) / (imgs.max() - imgs.min())
        imgs = torch.from_numpy(imgs)
        imgs = rearrange(imgs, "h w d c -> c h w d")
        imgs = imgs[:, h_start : h_end, w_start : w_end, d_start : d_end]
        
        path = self.root + folder + "/" + folder + "_seg" + ".nii" 
        label = torch.from_numpy(np.asanyarray(nib_load(path), dtype='int64', order='C'))
        label[label == 4] = 3
        label = label[ h_start : h_end, w_start : w_end, d_start : d_end]
        label = label.to(torch.int64)
        
        return imgs, label
        

# class BratsMatrixTestDataset(Dataset):
#     def __init__(self):
#         self.train_root = '/home/kathy/model2/processing_data_1/test/'
#         self.folder_list = [self.train_root + folder for folder in os.listdir(self.train_root)]
        
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
#         feature = torch.from_numpy(imgs)

#         h_start = (cfg.data_h - cfg.resolution) // 2
#         w_start = (cfg.data_w - cfg.resolution) // 2
#         d_start = (cfg.data_d - cfg.resolution) // 2
#         h_end = h_start + cfg.resolution
#         w_end = w_start + cfg.resolution
#         d_end = d_start + cfg.resolution
#         # imgs = (imgs - imgs.min()) / (imgs.max() - imgs.min())
#         imgs = torch.from_numpy(imgs)
#         feature = rearrange(feature, "h w d c -> c h w d")
#         feature = feature[:, h_start : h_end, w_start : w_end, d_start : d_end]
        
#         label = torch.from_numpy(np.load(label_file))
#         label = label[ h_start : h_end, w_start : w_end, d_start : d_end]
        
#         label[label == 4] = 3 
#         return feature, label

    
    
def get_data_loader(batch_size = 8, mode = "train"):
    bmtd = BratsMatrixTrainDataset() if mode == "train" else BratsMatrixTestDataset()
    
    return DataLoader(bmtd, batch_size=batch_size, shuffle=True if mode == "train" else False)





if __name__ == "__main__":
    import time
    bmtd = get_data_loader()
    start = time.time()
    X, y = next(iter(bmtd))
    end = time.time()
    print(end - start)
    print(X.shape)
    print(torch.unique(y))