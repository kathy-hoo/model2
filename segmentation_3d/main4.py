from data.brats_matrix_sample import get_data_loader
import torch 
import pytorch_lightning as pl 
import torch.nn as nn 
import torch.nn.functional as F 
from loss.dice_loss import DiceLoss
from pytorch_lightning.loggers import CSVLogger
from models.Unet3D_mlp_abl_circle import UNet
from tqdm import tqdm
from loss.dice_loss import softmax_dice2, dice_score, mask_dice_loss,  mask_dice_score, dice_score_per_class_list
from predict.predict64 import before_predict

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
        X, y = batch 
        X = X.to(torch.float32)
        y = y.to(torch.int64)
        output = self(X)
        ce_loss = self.ce_loss(output, y)
        # dice_loss = self.dice_loss(torch.nn.functional.softmax(output, dim = 1), y)
        dice1, dice2, dice3 = mask_dice_loss(torch.nn.functional.softmax(output, dim = 1), y, num_classes=4)
        
        avgdice = 1 - (dice1 + dice2 + dice3)/3
        
        self.log("ce_loss: ", ce_loss, prog_bar=True, on_epoch=True, on_step=True)
        self.log("dice_loss: ", avgdice, prog_bar=True, on_epoch=True, on_step=True)
        # self.log("dice_loss1: ", dice1, prog_bar=True, on_epoch=True, on_step=True)
        # self.log("dice_loss2: ", dice2, prog_bar=True, on_epoch=True, on_step=True)
        # self.log("dice_loss3: ", dice3, prog_bar=True, on_epoch=True, on_step=True)
        
        # return ce_loss +  dice1 + dice2 + dice3
        return ce_loss + avgdice
    
    def validation_step(self, batch, batch_idx):
        X, y = batch 
        X = X.to(torch.float32)
        y = y.to(torch.int64)
        output = self(X)
        ce_loss = self.ce_loss(output, y)
        # dice_loss = self.dice_loss(output, y)
        dice_loss,   dice1, dice2, dice3 = softmax_dice2(torch.nn.functional.softmax(output, dim = 1), y, with_zeros=False)
        
        
        self.log("ce_loss_val", ce_loss, prog_bar=True, on_epoch=True, on_step=True)
        self.log("dice_loss_val", dice_loss, prog_bar=True, on_epoch=True, on_step=True)
        self.log("dice1", dice1)
        self.log("dice2", dice2)
        self.log("dice3", dice3)
        
        
        return ce_loss + 0.5 * dice_loss
    def test_step(self, batch, batch_idx):
        folder, X, y = batch 
        
        X = X.to(torch.float32)
        y = y.to(torch.int64)
        # y_one_hot = torch.nn.functional.one_hot(y).permute(0, 4, 1, 2, 3)
        # X, y = before_predict(y, X)
        # dice1, dice2, dice3 = 0, 0, 0
        # for i in range(X.shape[0]):
        # # mask_true = torch.nn.functional.one_hot(y).permute(0, 4, 1, 2, 3)
        pred = self(X)
            
        dice1, dice2, dice3 = mask_dice_score(torch.nn.functional.softmax(pred, dim = 1), y, num_classes=4)
        # dice1, dice2, dice3 = dice_score(torch.nn.functional.softmax(pred, dim = 1), y, num_classes=4)
        # self.log("folder", int(folder[0][-3:]), on_step=True, on_epoch=False)
        self.log("dice1", dice1, on_step=True)
        self.log("dice2", dice2, on_step=True)
        self.log("dice3", dice3, on_step=True)
    def _calculate_dice_scores_list(self, model_output, target, weighted ):
        lis = [[1], [1, 3], [1, 2, 3]]
        result = []
        for item in lis:
            result.append(dice_score_per_class_list(model_output, target, item, weighted))
        return result
    
    def _cls_dices_reet(self, model_output, target, *args):
        ret = softmax_output_dice(model_output, target)
        return ret 
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        schedular = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)
        return [optimizer], [schedular]  
    

if __name__ == "__main__":
    mode = "test"
    info =  "消融circle matrix"
    checkpoint_path = '/home/bai_gairui/seg_3d/model2/segmentation_3d/logs/resoulution-128-train/version_32/checkpoints/epoch=84-step=7820.ckpt'
    if 1 == 1:
        train_loader = get_data_loader(batch_size=1 if mode== "test" else 4, mode=mode)
        logger = CSVLogger("logs", name=f"resoulution-128-{mode}-{info}")
        trainer = pl.Trainer(accelerator="gpu",  devices=[2],  precision=32, max_epochs=1000, logger=logger)
        # model = Wapper.load_from_checkpoint('/home/bai_gairui/seg_3d/model2/segmentation_3d/logs/resoulution-128-train/version_12/checkpoints/epoch=99-step=27400.ckpt')
        model = Wapper.load_from_checkpoint(checkpoint_path=checkpoint_path)
        if mode == "train":
            trainer.fit(model, train_loader)
        else:
            trainer.test(model, train_loader)
            
    else:
        pass 
            
            
    
