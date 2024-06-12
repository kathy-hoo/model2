from data.udc_matrix_sample import get_data_loader
import torch 
import pytorch_lightning as pl 
import torch.nn as nn 
import torch.nn.functional as F 
from loss.dice_loss import DiceLoss
from pytorch_lightning.loggers import CSVLogger
from models.Unet3D_mlp_abl_circle import UNet
from tqdm import tqdm
from loss.dice_loss import softmax_dice2, dice_score, mask_dice_loss,  mask_dice_score, dice_score_per_class, dice_score_per_class_list
from predict.predict64 import before_predict
from loss.cls_dices import softmax_output_dice
import numpy as np 
import seg_metrics.seg_metrics as sg
from einops import rearrange
import time
from models.TransBTS.TransBTS_downsample8x_skipconnection import TransBTS

class Wapper(pl.LightningModule):
    def __init__(self, info ):
        super().__init__()
        self.save_hyperparameters()
        _, self.os = TransBTS(dataset='brats', _conv_repr=True, _pe_type="learned")
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
        dice1, dice2, dice3  = mask_dice_loss(torch.nn.functional.softmax(output, dim = 1), y, num_classes=4)
        
        avgdice = 1 - (dice1 + dice2 + dice3 )/3
        
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
        X, y = batch 
        
        X = X.to(torch.float32)
        y = y.to(torch.int64)
        # y_one_hot = torch.nn.functional.one_hot(y).permute(0, 4, 1, 2, 3)
        # X, y = before_predict(y, X)
        # dice1, dice2, dice3 = 0, 0, 0
        # for i in range(X.shape[0]):
        start = time.time()
        pred = self(X)
        end = time.time()
        
        # for i in range(1, 4):
        #     dice_score = dice_score_per_class(torch.nn.functional.softmax(pred, dim = 1), y, class_index=i)
        #     self.log(f"dice{i}", dice_score, on_step=True)
            
        # model_output = torch.nn.functional.softmax(pred, dim = 1)
        model_output = pred.argmax(dim = 1) 
        cls124, cls14 , cls4 = self._cls_dices_reet(model_output, y)
        hd1, hd2, hd3 = self._cls_hd(model_output[0], y[0])
        self.log("cls124", cls124, on_step=True, on_epoch=True, prog_bar=True)
        self.log("cls14", cls14, on_step=True, on_epoch=True, prog_bar=True)
        self.log("cls4", cls4, on_step=True, on_epoch=True, prog_bar=True)
        self.log("hd1", hd1, on_step=True, on_epoch=True, prog_bar=True)
        self.log("hd2", hd2, on_step=True, on_epoch=True, prog_bar=True)
        self.log("hd3", hd3, on_step=True, on_epoch=True, prog_bar=True)
        self.log("consume", end - start, on_step=True, on_epoch=True, prog_bar=True)
        

    def _calculate_dice_scores_list(self, model_output, target, weighted ):
        lis = [[1], [1, 3], [1, 2, 3]]
        result = []
        for item in lis:
            result.append(dice_score_per_class_list(model_output, target, item, weighted))
        return result
    
    def _cls_dices_reet(self, model_output, target, *args):
        ret = softmax_output_dice(model_output, target)
        return ret 
    
    def _cls_hd(self, model_output, target):
        '''
        model_output : [num_cls, h, w, d]
        target : [num_cls, h, w, d]
        '''
        # model_output = rearrange(model_output, "c h w d -> h w d c")
        # target = rearrange(target, "c h w d -> h w d c")
        pred_img = model_output.cpu().detach().numpy()
        pred_img[0, 0, 0] = 1
        pred_img[0, 0, 1] = 2
        pred_img[0, 0, 2] = 3
        gdth_img = target.cpu().detach().numpy()
        gdth_img[0, 0, 0] = 1
        gdth_img[0, 0, 1] = 2
        gdth_img[0, 0, 2] = 3
        return sg.write_metrics(labels=[1, 2, 3], gdth_img=gdth_img, pred_img=pred_img, metrics=["hd95"])[0]["hd95"]
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
        schedular = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5000)
        return [optimizer], [schedular]  
    


if __name__ == "__main__":
    mode = "train"
    info =  "cp_transUnet_udc"
    
    checkpoint_path = '/home/bai_gairui/seg_3d/model2/segmentation_3d/logs/resoulution-128-train-cp_transUnet/version_4/checkpoints/epoch=905-step=166704.ckpt'
    if 1 == 1:
        train_loader = get_data_loader(batch_size=1 if mode== "test" else 2, mode=mode)
        logger = CSVLogger("logs", name=f"resoulution-128-{mode}-{info}")
        trainer = pl.Trainer(accelerator="gpu",  devices=[3],  precision=32, max_epochs=3000, logger=logger)
        # model = Wapper.load_from_checkpoint('/home/bai_gairui/seg_3d/model2/segmentation_3d/logs/resoulution-128-train/version_12/checkpoints/epoch=99-step=27400.ckpt')
        model = Wapper("")
        if mode == "train":
            trainer.fit(model, train_loader)
        else:
            trainer.test(model, train_loader)
            
    else:
        model = Wapper("") 
        x = torch.rand((1, 4, 128, 128, 128), device=model.device)
        y = model(x)
        print(y.shape)
        
            
    
