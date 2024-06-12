import sys 
sys.path.append('/home/kathy/model2/segmentation_3d')
import torch 
import torch.nn as nn 
import os 
os.chdir('/home/kathy/model2/segmentation_3d/')
from predict.predict64 import  before_predict, predict
from data.brats_matrix import  get_data_loader
import numpy as np 
from config import  config as cfg 
from einops import  rearrange
import pytorch_lightning as pl 
from models.Unet3D_mlp import UNet

class Wapper(pl.LightningModule):
    def __init__(self, info ):
        super().__init__()
        self.save_hyperparameters()
        self.os = UNet(4, 4)
        for param in self.os.parameters():
            param.requires_grad = True
        self.ce_loss = nn.CrossEntropyLoss()
    def forward(self, X):
        return self.os(X)
    
    
model = Wapper.load_from_checkpoint('/home/kathy/model2/segmentation_3d/logs/logger_test_mlp_easy/version_6/checkpoints/epoch=98-step=3465.ckpt')
X = np.load('/home/kathy/model2/processing_data_1/test/BraTS20_Training_007/feature.npy')
y = np.load('/home/kathy/model2/processing_data_1/test/BraTS20_Training_007/label.npy')

X = torch.from_numpy(X)[None, ...]
y = torch.from_numpy(y)[None, ...]

mask = X.sum(-1) > 0
X = X.to(torch.float32)
X = (X - X[mask].mean()) / (X[mask].std())
X = rearrange(X, "b h w d c -> b c h w d")

output, label, ijk, indices = before_predict(y, X)

print('...predicting')
output = predict(model, output, ijk, indices)
print(output.shape)
np.save('./demo', output.detach().numpy())