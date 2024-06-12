import torch 
import torch.nn as nn 
from torch.nn import  functional as F 
import time 

import sys 
sys.path.append('..')
from utils.utils import to_3_tuple

class Conv3D_block(nn.Module):
    def __init__(self, in_channels = 1, out_channles = 64, up_sample = True):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channles, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm3d(out_channles), 
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Conv3d(in_channels=out_channles, out_channels=out_channles, kernel_size=3, padding=1, stride=1)
        if up_sample:
            self.up_sampler = nn.Identity() if not up_sample else nn.Conv3d(in_channels=in_channels, out_channels=out_channles, kernel_size=1, stride=1)
        self.up_sample = up_sample
    def forward(self, X):
        return self.conv2(self.conv1(X) + self.up_sampler(X)) if self.up_sample else self.conv2(self.conv1(X))
        
class DeConv3D_block(nn.Module):
    def __init__(self, in_channels = 3, out_channles = 64,  stride = 2, padding = 1, kernel_size = 3, out_padding = 1):
        super().__init__()
        kernel_size  = to_3_tuple(kernel_size)
        stride = to_3_tuple(stride)
        padding = to_3_tuple(padding)
        self.deconv = nn.ConvTranspose3d(in_channels=in_channels, out_channels=out_channles, kernel_size=kernel_size, padding=padding, stride=stride, output_padding=out_padding)
        
    def forward(self, X):
        return self.deconv(X)
    
    

        
        
        
class Unet3D(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Encoder 
        self.conv1 = Conv3D_block(in_channels=4, out_channles=64, up_sample=False)
        self.conv2 = Conv3D_block(in_channels=64, out_channles=128, up_sample=False)
        self.conv3 = Conv3D_block(in_channels=128, out_channles=256, up_sample=False)
        self.conv4 = Conv3D_block(in_channels=256, out_channles=512, up_sample=False)
        self.conv5 = Conv3D_block(in_channels=512, out_channles=1024, up_sample=False)
        
        self.pool1 = nn.MaxPool3d((2, 2, 2))
        self.pool2 = nn.MaxPool3d((2, 2, 2))
        self.pool3 = nn.MaxPool3d((2, 2, 2))
        self.pool4 = nn.MaxPool3d((2, 2, 2))
        
        # Decoder
        self.deconv1 = DeConv3D_block(in_channels=1024, out_channles=512)
        self.deconv2 = DeConv3D_block(in_channels=512, out_channles=256)
        self.deconv3 = DeConv3D_block(in_channels=256, out_channles=128)
        self.deconv4 = DeConv3D_block(in_channels=128, out_channles=64)
        
        self.decoder_conv1 = Conv3D_block(in_channels=512 * 2, out_channles=512)
        self.decoder_conv2 = Conv3D_block(in_channels=256 * 2, out_channles=256)
        self.decoder_conv3 = Conv3D_block(in_channels=128 * 2, out_channles=128)
        self.decoder_conv4 = Conv3D_block(in_channels=64 * 2, out_channles=64)
        
        self.one_classes = nn.Conv3d(in_channels=64, out_channels=4, kernel_size=1)
        
    def forward(self, X):
        low_1 = self.conv1(X)
        low_x1 = self.pool1(low_1) # b 64, 32 ^ 3
        
        low_2 = self.conv2(low_x1)
        low_x2 = self.pool2(low_2) # b 128, 16 ^ 3
        
        low_3 = self.conv3(low_x2) # b 256 16^3
        low_x3 = self.pool3(low_3) # # b 256, 8 ^ 3
        
        low_4 = self.conv4(low_x3) # b 256 8 ^ 3
        low_x4 = self.pool4(low_4) # b 512, 4 ^ 3
        
        base = self.conv5(low_x4) # # b 1024, 4 ^ 3
        
        # Decoding 
        
        up_1 = torch.cat([self.deconv1(base), low_4], dim = 1)
        up_x1 = self.decoder_conv1(up_1) # b 512 8^3
        
        up_2 = torch.cat([self.deconv2(up_x1), low_3], dim = 1) # b 256*2 16^3
        up_x2 = self.decoder_conv2(up_2) # b 256 8^3
        
        up_3 = torch.cat([self.deconv3(up_x2), low_2], dim = 1) # b 128*2 32^3
        up_x3 = self.decoder_conv3(up_3) # b 128 32^3
        
        
        up_4 = torch.cat([self.deconv4(up_x3), low_1], dim=1)
        up_x4 = self.decoder_conv4(up_4)
        
        output = torch.sigmoid(self.one_classes(up_x4))
        
        return output
    
# (4 - 1) * s - 2p + k + op = 8
# 3s - 2p + 3 + op = 8
    



if __name__ == '__main__':
    if 1 == 1:
        start = time.time()
        device = torch.device("cpu")
        ud = Unet3D().to(device)
        X = torch.randn(1, 1, 64, 64, 64).to(device)
        oup = ud(X)
        print(oup.shape)
        end = time.time()
        print(end - start )
    
    if 1  != 1:
        db = DeConv3D_block(in_channels=1024, out_channles=512)
        x = torch.rand(1, 1024, 4, 4, 4)
        print(db(x).shape)