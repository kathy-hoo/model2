import torch 
import torch.nn as nn 
from einops import rearrange
from model.Unet3D import Conv3D_block, DeConv3D_block
from components.permute_mlp import Pc_mlp


class OverallStructure(nn.Module):
    def __init__(self, resolution, num_classes = 4):
        
        # Conv Stage
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels=4, out_channels=64, kernel_size=(3, 3, 1), padding=(1, 1, 0)), 
            nn.BatchNorm3d(64), 
            nn.ReLU(inplace=True)
        )
        self.conv2 = Conv3D_block(in_channels=64, out_channles=128, up_sample=False)
        self.conv3 = Conv3D_block(in_channels=128, out_channles=256, up_sample=False)
        
        input_hwd = resolution // (2 ** 3)
        self.conv_pc_mlp_1 = Pc_mlp(hwd=input_hwd, dim = 256, out_dim=512)
        input_hwd = input_hwd // 2
        self.conv_pc_mlp_2 = Pc_mlp(hwd=input_hwd, dim = 512, out_dim=1024)
        
        self.down_sample1 = nn.MaxPool3d((2, 2, 2))
        self.down_sample2 = nn.MaxPool3d((2, 2, 2))
        self.down_sample3 = nn.MaxPool3d((2, 2, 2))
        self.down_sample4 = nn.MaxPool3d((2, 2, 2))
        
        # Decoder
        self.deconv1 = DeConv3D_block(in_channels=1024, out_channles=512) # 4 -> 8
        self.deconv2 = DeConv3D_block(in_channels=512, out_channles=256) # 4 -> 8
        self.deconv3 = DeConv3D_block(in_channels=256, out_channles=128) # 4 -> 8
        self.deconv4 = DeConv3D_block(in_channels=128, out_channles=64) # 4 -> 8
        
        self.decoder_conv1 = Pc_mlp(hwd = input_hwd * 2, dim = 512 * 2, out_dim=512)
        self.decoder_conv2 = Conv3D_block(in_channels=256 * 2, out_channles=256)
        self.decoder_conv3 = Conv3D_block(in_channels=128 * 2, out_channles=128)
        self.decoder_conv4 = Conv3D_block(in_channels=64 * 2, out_channles=64)
        
        self.conv_one = nn.Conv3d(in_channels=64, out_channels=num_classes, kernel_size=1)
        
        
    def forward(self, X):
        '''
        X : B C H W D
        '''
        x1 = self.conv1(X) # B 64 64 64 64
        x1_low = self.down_sample1(x1) # B 64 32 32 32
        
        x2 = self.conv2(x1_low) # B 128 32 32 32
        x2_low = self.down_sample2(x2) # B 128 16 16 16
        
        x3 = self.conv3(x2_low) # B 256 16 16 16
        x3_low = self.down_sample3(x3) # B 256 8 8 8 
        
        x4 = self.conv_pc_mlp_1(x3_low) # B 512 8 8 8 
        x4_low = self.down_sample4(x4) # B 512 4 4 4 
        
        base = self.conv_pc_mlp_2(x4_low) # B 1024 4 4 4
        
        up_1 = torch.cat([self.deconv1(base), x4], dim = 1) # b 512*2 8 8 8 
        up_x1 = self.decoder_conv1(up_1) # b 512 8 8 8 
        
        up_2 = torch.cat([self.deconv2(up_x1), x3], dim = 1) # B 256*2 16 16 16
        up_x2 = self.decoder_conv2(up_2) # B 256 16 16 16
        
        up_3 = torch.cat([self.deconv3(up_x2), x2], dim = 1) 
        up_x3 = self.decoder_conv3(up_3) # B 128 32 32 32
        
        up_4 = torch.cat([self.deconv4(up_x3), x1], dim = 1)
        up_x4 = self.decoder_conv4(up_4) # B 64 64 64 64 
        
        oup = self.conv_one(up_x4)
        
        return torch.sigmoid(oup)
    
    
if __name__ == "__main__":
    device = torch.device("cuda:3")
    x = torch.rand(8, 3, 64, 64, 64).to(device)
    ost = OverallStructure(resolution=64).to(device)
    oup = ost(x) # 4 4 64 64 64
    print(oup.shape)        
        
        
        
        
        
        