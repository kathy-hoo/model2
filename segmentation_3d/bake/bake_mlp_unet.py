from data.brats2020 import get_data_loader_bra_2020
import torch 
import pytorch_lightning as pl 
import torch.nn as nn 
import torch.nn.functional as F 
from loss.dice_loss import DiceLoss
from pytorch_lightning.loggers import CSVLogger
from models.Unet3D_mlp import UNet
from tqdm import tqdm


class Wapper(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.os = UNet(4, 4)
        for param in self.os.parameters():
            param.requires_grad = True
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()
    def forward(self, X):
        return self.os(X)
    
    def training_step(self, batch, batch_idx):
        file_name, X, y = batch 
        X = X.to(torch.float32)
        y = y.to(torch.int64)
        output = self(X)
        ce_loss = self.ce_loss(output, y)
        dice_loss = self.dice_loss(torch.nn.functional.softmax(output, dim = 1), y)
        
        self.log("ce_loss: ", ce_loss, prog_bar=True, on_epoch=True, on_step=True)
        self.log("dice_loss: ", dice_loss, prog_bar=True, on_epoch=True, on_step=True)
        
        return ce_loss + 0.5 * dice_loss
    
    # def validation_step(self, batch, batch_idx):
    #     file_name, X, y = batch 
    #     X = X.to(torch.float32)
    #     y = y.to(torch.int64)
    #     output = self(X)
    #     ce_loss = self.ce_loss(output, y)
    #     dice_loss = self.dice_loss(output, y)
        
    #     self.log("ce_loss_val", ce_loss, prog_bar=True, on_epoch=True, on_step=True)
    #     self.log("dice_loss_val", dice_loss, prog_bar=True, on_epoch=True, on_step=True)
        
    #     return ce_loss + 0.5 * dice_loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        schedular = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        return [optimizer], [schedular]  
    
