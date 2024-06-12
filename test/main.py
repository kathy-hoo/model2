from structure import OverallStructure
from data.brats2020 import get_data_loader_bra_2020
import torch 
import pytorch_lightning as pl 
import torch.nn as nn 
import torch.nn.functional as F 
from config import config as cfg
from loss.dice_loss import DiceLoss
from pytorch_lightning.loggers import CSVLogger
from model.Unet3D import Unet3D
from tqdm import tqdm


class Wapper(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.os = OverallStructure(resolution=64)
        for param in self.os.parameters():
            param.requires_grad = True
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()
        self.batch_dice_loss = []
        self.batch_ce_loss = []
        self.pre_dice_loss = 3
        self.current_dice_loss = 0
        self.current_ce_loss = 0
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
    def on_train_batch_end(self):
        pass  
    
    def on_validation_batch_end(self):
        self.current_dice_loss = sum(self.batch_dice_loss) / len(self.batch_dice_loss)
        self.current_ = sum(self.batch_dice_loss) / len(self.batch_dice_loss)
        self.batch_dice_loss.clear()
        
    
    def validation_step(self, batch, batch_idx):
        file_name, X, y = batch 
        X = X.to(torch.float32)
        y = y.to(torch.int64)
        output = self(X)
        ce_loss = self.ce_loss(output, y)
        dice_loss = self.dice_loss(output, y)
        self.batch_ce_loss.append(ce_loss.item())
        self.batch_dice_loss.append(dice_loss.item())
        
        self.log("ce_loss_val", ce_loss, prog_bar=True, on_epoch=True, on_step=True)
        self.log("dice_loss_val", dice_loss, prog_bar=True, on_epoch=True, on_step=True)
        
        return ce_loss + 0.5 * dice_loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        schedular = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        return [optimizer], [schedular]  
    

if __name__ == "__main__":
    train_loader, test_loader = get_data_loader_bra_2020(batch_size=8)
    logger = CSVLogger("logs", name="logger_train2")
    trainer = pl.Trainer(accelerator="gpu",  devices=[2],  precision=32, max_epochs=100, logger=logger, log_every_n_steps=1)
    
    model = Wapper()
    trainer.fit(model, train_loader)
    # device = torch.device("cuda:1")
    # train_loader, test_loader = get_data_loader_bra_2020(batch_size=2)
    # model = OverallStructure(resolution=cfg.resolution, num_classes=cfg.num_classes + 1).to(torch.float32).to(device)
    # ce_loss = nn.CrossEntropyLoss()
    # dice_loss = DiceLoss()
    # optim = torch.optim.Adam(model.parameters() ,lr=1e-3)
    
    # for i in range(100):
    #     for step, (folder, X, y) in enumerate(train_loader):
    #         X = X.to(torch.float32).to(device)
    #         y = y.to(device)
    #         output = model(X)
            
    #         ce_loss_output = ce_loss(output, y)
    #         dice_loss_output = dice_loss(output, y)
    #         loss = ce_loss_output + 0.5 * dice_loss_output
            
    #         optim.zero_grad()
    #         loss.backward()
    #         optim.step()
            
    #         print(ce_loss_output, dice_loss_output)
            
            
            
            
    
