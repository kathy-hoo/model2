from data.brats2020 import get_data_loader_bra_2020
import torch 
import pytorch_lightning as pl 
import torch.nn as nn 
import torch.nn.functional as F 
from loss.dice_loss import DiceLoss
from pytorch_lightning.loggers import CSVLogger
from models.Unet3D_mlp import UNet
from tqdm import tqdm
from loss.dice_loss import softmax_dice2

class Wapper(pl.LightningModule):
    def __init__(self, info ):
        super().__init__()
        self.save_hyperparameters()
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
        # dice_loss = self.dice_loss(torch.nn.functional.softmax(output, dim = 1), y)
        dice_loss, dice0,  dice1, dice2, dice3 = softmax_dice2(torch.nn.functional.softmax(output, dim = 1), y)
        
        self.log("ce_loss: ", ce_loss, prog_bar=True, on_epoch=True, on_step=True)
        self.log("dice_loss: ", dice_loss, prog_bar=True, on_epoch=True, on_step=True)
        
        return ce_loss + 0.5 * dice_loss
    
    def validation_step(self, batch, batch_idx):
        file_name, X, y = batch 
        X = X.to(torch.float32)
        y = y.to(torch.int64)
        output = self(X)
        ce_loss = self.ce_loss(output, y)
        # dice_loss = self.dice_loss(output, y)
        dice_loss, dice0,  dice1, dice2, dice3 = softmax_dice2(torch.nn.functional.softmax(output, dim = 1), y)
        
        
        self.log("ce_loss_val", ce_loss, prog_bar=True, on_epoch=True, on_step=True)
        self.log("dice_loss_val", dice_loss, prog_bar=True, on_epoch=True, on_step=True)
        self.log("dice0", dice0)
        self.log("dice1", dice1)
        self.log("dice2", dice2)
        self.log("dice3", dice3)
        
        
        return ce_loss + 0.5 * dice_loss
    def test_step(self, batch, batch_idx):
        folder, X, y = batch 
        X = X.to(torch.float32)
        y = y.to(torch.int64)
        # mask_true = torch.nn.functional.one_hot(y).permute(0, 4, 1, 2, 3)
        pred = self(X)
        
        dice_loss, dice1, dice2, dice3 = softmax_dice2(torch.nn.functional.softmax(pred, dim = 1), y, with_zeros = False)
        self.log("dice1", dice1)
        # self.log("dice1", dice1)
        self.log("dice2", dice2)
        self.log("dice3", dice3)
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        schedular = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        return [optimizer], [schedular]  
    

if __name__ == "__main__":
    train_loader, test_loader = get_data_loader_bra_2020(batch_size=8)
    logger = CSVLogger("logs", name="logger_train2_mlp")
    trainer = pl.Trainer(accelerator="gpu",  devices=[6],  precision=32, max_epochs=100, logger=logger, log_every_n_steps=1)
    model = Wapper(info="permute_mlp中修改s的位置, 6卡执行")
    # model = Wapper.load_from_checkpoint('/home/kathy/model2/segmentation_3d/logs/logger_train2_mlp/version_1/checkpoints/epoch=48-step=1715.ckpt')
    # model = Wapper.load_from_checkpoint('/home/kathy/model2/segmentation_3d/logs/logger_train2/version_1/checkpoints/epoch=99-step=1800.ckpt')
    trainer.fit(model, train_loader, test_loader)
    
            
            
            
            
    
