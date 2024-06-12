import torch 
import torch.nn as nn 
import sys 
sys.path.append('..')
from config import config as cfg 

# def dice_metrcis_brats2020(output, target):
#     '''
#     X : b num_classes h w d 
#     y : b h w d 
#     '''
    
#     # dice1
#     output[:, 1, ...].contiguous().view(-1), taregt 


def test(output, target):
    output = output[:, 1, ...]
    target = (target == 1).float()
    
    output = output.contiguous().view(-1)
    target = target.contiguous().view(-1)
    
    intercection =  output * target
    
    
    
    