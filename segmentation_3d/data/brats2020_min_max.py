import torch 
from torch.utils.data import Dataset, DataLoader
import os 
import nibabel as nib 
import numpy as np 
from einops import rearrange
import sys 
sys.path.append('..')
from config import config as cfg 
train_set = {
    # 'root': 'path to training set',
    'root': '/home/kathy/dataset/BraTS2020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/',
    'flist': '/home/kathy/model2/test/train.txt',
    'has_label': True,
    'mode' : "train"
}
test_set = {
    # 'root': 'path to training set',
    'root': '/home/kathy/dataset/BraTS2020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/',
    'flist': '/home/kathy/model2/test/test.txt',
    'has_label': True, 
    'mode' : 'test'
}

modalities = ["flair", "t1", "t1ce", "t2"]

def nib_load(file_name):
    if not os.path.exists(file_name):
        print('Invalid file name, can not find the file!')

    proxy = nib.load(file_name)
    data = proxy.get_data()
    proxy.uncache()
    return data
class Brats2020(Dataset):
    def __init__(self, data_info) -> None:
        super().__init__()
        self.root = data_info["root"]
        flist = data_info["flist"]
        self.has_label = data_info["has_label"]
        self.folder_list = []
        self.mode = data_info['mode']
        with open(flist, 'r', encoding="utf8") as f:
            self.folder_list = f.read().split("\n")
    def __len__(self):
        return len(self.folder_list)
    def __getitem__(self, index):
        folder = self.folder_list[index]
        imgs = []
        for modality in modalities:
            path = self.root + folder + "/" + folder + "_" + modality + ".nii" 
            img = np.asanyarray(nib_load(path), dtype='float32', order='C')
            img = img[..., None]
            imgs.append(img)
        imgs = np.concatenate(imgs, axis=-1)
        imgs = imgs / 255
        imgs = torch.from_numpy(imgs)
        imgs = rearrange(imgs, 'h w d c -> c h w d')
        if self.mode == 'train':
            # random crop 
            h_start = torch.randint(low = 0, high = cfg.data_h - cfg.resolution, size = (1, ))
            h_end = h_start + cfg.resolution
            w_start = torch.randint(low = 0, high = cfg.data_w - cfg.resolution, size = (1, ))
            w_end =w_start + cfg.resolution
            d_start = torch.randint(low = 0, high = cfg.data_d - cfg.resolution, size = (1, ))
            d_end =d_start + cfg.resolution
        else:
            # center crop
            h_start = (cfg.data_h - cfg.resolution) // 2
            w_start = (cfg.data_w - cfg.resolution) // 2
            d_start = (cfg.data_d - cfg.resolution) // 2
            h_end = h_start + cfg.resolution
            w_end = w_start + cfg.resolution
            d_end = d_start + cfg.resolution
        imgs = imgs[:, h_start : h_end, w_start : w_end, d_start : d_end]
        imgs = imgs.to(torch.float64)
        if not self.has_label:
            return imgs 
        
        path = self.root + folder + "/" + folder + "_seg" + ".nii" 
        label = torch.from_numpy(np.asanyarray(nib_load(path), dtype='int64', order='C'))
        label[label == 4] = 3
        label = label[ h_start : h_end, w_start : w_end, d_start : d_end]
        label = label.to(torch.int64)
        return folder, imgs, label
    
def get_data_loader_bra_2020(batch_size = 4):
    bs_train = Brats2020(data_info=train_set)
    bs_test = Brats2020(data_info=test_set)
    train_loader = DataLoader(bs_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(bs_test, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader    

if __name__ == "__main__":
    bs = Brats2020(data_info=test_set)
    folder, imgs, label = next(iter(bs))
    print(imgs.shape, label.shape)
            
        
        
        