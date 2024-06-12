"""Adapted from https://github.com/milesial/Pytorch-UNet/tree/master/unet"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import os 
import sys 
import thop 
sys.path.append('..')
from config import config as cfg
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, width_multiplier=1, trilinear=True, use_ds_conv=False):
        """A simple 3D Unet, adapted from a 2D Unet from https://github.com/milesial/Pytorch-UNet/tree/master/unet
        Arguments:
          n_channels = number of input channels; 3 for RGB, 1 for grayscale input
          n_classes = number of output channels/classes
          width_multiplier = how much 'wider' your UNet should be compared with a standard UNet
                  default is 1;, meaning 32 -> 64 -> 128 -> 256 -> 512 -> 256 -> 128 -> 64 -> 32
                  higher values increase the number of kernels pay layer, by that factor
          trilinear = use trilinear interpolation to upsample; if false, 3D convtranspose layers will be used instead
          use_ds_conv = if True, we use depthwise-separable convolutional layers. in my experience, this is of little help. This
                  appears to be because with 3D data, the vast vast majority of GPU RAM is the input data/labels, not the params, so little
                  VRAM is saved by using ds_conv, and yet performance suffers."""
        super(UNet, self).__init__()
        _channels = (32, 64, 128, 256, 512)
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.channels = [int(c*width_multiplier) for c in _channels]
        self.trilinear = trilinear
        self.convtype = DepthwiseSeparableConv3d if use_ds_conv else nn.Conv3d

        self.inc = DoubleConv(n_channels, self.channels[0], conv_type=self.convtype)
        self.down1 = Down(self.channels[0], self.channels[1], conv_type=self.convtype)
        self.down2 = Down(self.channels[1], self.channels[2], conv_type=self.convtype)
        # self.down3 = Down(self.channels[2], self.channels[3], conv_type=self.convtype)
        self.down3 = Pc_mlp_down(cfg.resolution // 4, self.channels[2], self.channels[3])
        factor = 2 if trilinear else 1
        self.down4 = Pc_mlp_down(cfg.resolution // 8, self.channels[3], self.channels[4] // factor)
        # self.down4 = Down(self.channels[3], self.channels[4] // factor, conv_type=self.convtype)
        # self.up1 = Up(self.channels[4], self.channels[3] // factor, trilinear)
        self.up1 = Pc_mlp_up(cfg.resolution // 8, self.channels[4], self.channels[3] // factor)
        
        self.up2 = Up(self.channels[3], self.channels[2] // factor, trilinear)
        self.up3 = Up(self.channels[2], self.channels[1] // factor, trilinear)
        self.up4 = Up(self.channels[1], self.channels[0], trilinear)
        self.outc = OutConv(self.channels[0], n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, conv_type=nn.Conv3d, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            conv_type(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            conv_type(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, conv_type=nn.Conv3d):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels, conv_type=conv_type)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, trilinear=True):
        super().__init__()

        # if trilinear, use the normal convolutions to reduce the number of channels
        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, mid_channels=in_channels // 2) 
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class DepthwiseSeparableConv3d(nn.Module):
    def __init__(self, nin, nout, kernel_size, padding, kernels_per_layer=1):
        super(DepthwiseSeparableConv3d, self).__init__()
        self.depthwise = nn.Conv3d(nin, nin * kernels_per_layer, kernel_size=kernel_size, padding=padding, groups=nin)
        self.pointwise = nn.Conv3d(nin * kernels_per_layer, nout, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out
    

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

   
class WeightedPermuteMLP_3d(nn.Module):
    def __init__(self, hwd, dim , out_dim = None, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.segment_dim = dim // hwd

        # self.mlp_h = nn.Linear(hwd, hwd, bias=qkv_bias)
        # self.mlp_w = nn.Linear(hwd, hwd, bias=qkv_bias)
        # self.mlp_d = nn.Linear(hwd, hwd, bias=qkv_bias)
        # self.mlp_h = nn.Parameter(torch.rand(hwd), requires_grad=True)
        # self.mlp_w = nn.Parameter(torch.rand(hwd), requires_grad=True)
        # self.mlp_d = nn.Parameter(torch.rand(hwd), requires_grad=True)

        self.mlp_h = CCS(hwd)
        self.mlp_w = CCS(hwd)
        self.mlp_d = CCS(hwd)
        
        self.hwd = hwd 
        self.reweight = Mlp(dim , dim // 4, dim *3)
        
        self.proj = nn.Linear(dim, out_dim)
        self.proj_drop = nn.Dropout(proj_drop)



    def forward(self, x):
        B, H, W, D, C = x.shape

        # mlp_h = torch.cat([torch.roll(self.mlp_h, shifts=i).unsqueeze(0) for i in range(self.hwd)], dim = 0)  
        # mlp_w = torch.cat([torch.roll(self.mlp_w, shifts=i).unsqueeze(0) for i in range(self.hwd)], dim = 0)  
        # mlp_d = torch.cat([torch.roll(self.mlp_d, shifts=i).unsqueeze(0) for i in range(self.hwd)], dim = 0)  
        N = C // self.segment_dim
        h = rearrange(x, "b w h d (s n) -> b s w d n h", n = N)
        # h = h @ mlp_h
        h = self.mlp_h(h)
        h = rearrange(h, "b s w d n h -> b w h d (s n)")
        
        w = rearrange(x, "b w h d (s n) -> b s h d n w",n = N)
        # w = w @ mlp_w
        w = self.mlp_w(w)
        w = rearrange(w, "b s h d n w -> b w h d (s n)")
        
        
        d = rearrange(x, "b w h d (s n) -> b w h s n d", n = N)
        # d = d @ mlp_d 
        d = self.mlp_d(d)
        
        d = rearrange(d, "b w h s n d -> b w h d (s n)")
        

        a = h + w + d # 4 32 32 32 512
        a = a.permute(0, 4, 1, 2, 3).flatten(2).mean(2) # b 512 
        a = self.reweight(a).reshape(B, C, 3).permute(2, 0, 1).softmax(dim = 0).unsqueeze(2).unsqueeze(2).unsqueeze(2)
        
        
        x = h * a[0] + w * a[1] + d * a[2]

        x = self.proj(x)
        
        x = self.proj_drop(x)
        

        return x
    
class Pc_mlp(nn.Module):
    def __init__(self ,hwd, dim, out_dim, proj_drop = 0.0):
        super().__init__()
        self.up_sample_mlp = nn.Linear(in_features=dim, out_features=out_dim)
        self.efficient_mlp_block = WeightedPermuteMLP_3d(hwd, dim, out_dim, proj_drop=proj_drop)
        self.channle_mlp = Mlp(in_features=out_dim, hidden_features=dim // 4, out_features=out_dim)
        self.layer_norm_before = nn.LayerNorm(dim)
        self.layer_norm_after = nn.LayerNorm(out_dim)
        # self.pool = nn.MaxPool3d((2, 2, 2)) if action == "down" else nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
    def forward(self, X): 
        '''
        X : b c h w d
        '''
        X = rearrange(X, "b c h w d -> b h w d c")
        X = self.layer_norm_before(X)
        # X = self.up_sample_mlp(X)
        ef_x = self.efficient_mlp_block(X)
        ef_x = self.up_sample_mlp(X) + ef_x 
        c_x = self.channle_mlp(ef_x)
        oup =  ef_x + c_x 
        oup = self.layer_norm_after(oup)
        oup = rearrange(oup, "b h w d c -> b c h w d")
        return oup 
    
class Pc_mlp_down(nn.Module):
    def __init__(self, hwd, in_channles, out_channles):
        super().__init__()
        self.conv = nn.MaxPool3d((2, 2, 2))
        self.pc_mlp = Pc_mlp(hwd, in_channles, out_channles)
    def forward(self, X):
        return self.conv(self.pc_mlp(X))
    
class Pc_mlp_up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, hwd, in_channels, out_channels, trilinear=True):
        super().__init__()

        # if trilinear, use the normal convolutions to reduce the number of channels
        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = Pc_mlp(hwd, in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = Pc_mlp(hwd, in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
class CCS(nn.Module):
    def __init__(self, hwd):
        super().__init__()
        self.groups = hwd
        # self.w = nn.Parameter(nn.Linear(hwd,self.groups).weight, requires_grad=True)
        self.w = nn.Linear(in_features=hwd, out_features=hwd)
    def forward(self, x : torch.Tensor):
        '''
        x : B W D S N H
        '''
        # B,  W , D , S, N, H = x.shape
        # x = torch.fft.ifft(x, dim = -1)
        # w = self.w.type_as(x)
        # w = torch.fft.fft(w, dim=1).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(B,  W , D , S, N, H)
        # x = x.matmul(w)
        # x = torch.fft.fft(x,dim=1).real
        
        return self.w(x)



if __name__ == "__main__":
    X = torch.rand(1, 4, 128, 128, 128)
    model = UNet(4, 4)
    flops, params = thop.profile(model, inputs = (X, ))

    print(model(X).shape)
    print(flops / (10 ** 9))
    print(params / 1000 / 2 ** 10)