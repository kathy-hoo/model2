import torch 
import torch.nn as nn 
from torch.nn import functional as F  
from einops import rearrange

class DoubleConv2d(nn.Module):
    def __init__(self, in_channles, out_channles):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels=in_channles, out_channels=out_channles, kernel_size=(3, 3, 1), padding=(1, 1, 0)), 
            nn.BatchNorm3d(out_channles),
            nn.ReLU(inplace=True), 
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels=out_channles, out_channels=out_channles, kernel_size=(3, 3, 1), padding=(1, 1, 0)), 
            nn.BatchNorm3d(out_channles),
            nn.ReLU(inplace=True), 
        )
        self.up = nn.Conv3d(in_channels=in_channles, out_channels=out_channles,  kernel_size=(3, 3, 1), padding=(1, 1, 0))
        
    def forward(self, x):
        return self.up(x) + self.conv2(self.conv1(x))
        
        
    
class DoubleConv3d(nn.Module):
    def __init__(self, in_channles, out_channles):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels=in_channles, out_channels=out_channles, kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=(2, 2, 2)), 
            nn.BatchNorm3d(out_channles),
            nn.ReLU(inplace=True), 
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels=out_channles, out_channels=out_channles, kernel_size=(3, 3, 3), padding=(1, 1, 1)), 
            nn.BatchNorm3d(out_channles),
            nn.ReLU(inplace=True), 
        )
        self.up = nn.Conv3d(in_channels=in_channles, out_channels=out_channles,  kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=(2, 2, 2))
        
    def forward(self, x):
        x1 =  self.up(x) 
        x2 = self.conv2(self.conv1(x))
        return x1 + x2

class MLPP(nn.Module):
    def __init__(self, whdc = 256, L = 16):
        '''
        W = H = C = D
        ''' 
        super().__init__()
        g = whdc // L 
        self.proj_h = nn.Linear(whdc, whdc)
        self.proj_w = nn.Linear(whdc, whdc)
        self.proj_c = nn.Linear(whdc, whdc)
        self.proj_d = nn.Linear(whdc, whdc)
        self.attn = nn.Linear(L ** 2, L ** 2)
        self.proj_1, self.proj_2, self.proj_3 =  nn.Linear(whdc, whdc), nn.Linear(whdc, whdc), nn.Linear(whdc, whdc)
        self.norm_1, self.norm_2, self.norm_3 =  nn.LayerNorm(whdc), nn.LayerNorm(whdc), nn.LayerNorm(whdc)
        self.L = L 
        self.g  = g 
        self.whdc = whdc 
    def IP_MLP(self, x):
        '''
        X : [b, h, w, d, c]
        '''
        b, h, w, d, c = x.shape
        x_att = rearrange(x, "b (h1 lh) (w1 lw) d c -> b h1 w1 d c (lh lw)", lh = self.L, lw = self.L)
        x_att = self.attn(x_att)
        x_att = rearrange(x_att, "b h1 w1 d c (lh lw) -> b (h1 lh) (w1 lw) d c", lh = self.L, lw = self.L)
        
        x_h = rearrange(x, "b (h1 lh) w d (lc g) -> b (h1 w) lc d (lh g)", lh = self.L, g = self.g)
        x_h = self.proj_h(x_h)
        x_h = rearrange(x_h ,"b (h1 w) lc d (lh g) -> b (h1 lh) w d (lc g)", w = w, g = self.g)
        
        x_w = rearrange(x, "b h (w1 lw) d (lc g) -> b (h w1) lc d (lw g)", lw = self.L, g = self.g)
        x_w = self.proj_w(x_w)
        x_w = rearrange(x_w, "b (h w1) lc d (lw g) -> b h (w1 lw) d (lc g)", h = h, g = self.g)
        
        x_c = self.proj_c(x)
        x = x_h + x_w + x_c
        x = (1 + x_att) * x
        x = self.proj_1(x)  
        
        return x 
    def TP_MLP(self, x):
        x_d = rearrange(x, "b h w (d1 ld) (l g) -> b h w d1 l (ld g)", ld = self.L, g = self.g)
        x_d = self.proj_d(x_d)
        x_d = rearrange(x_d, "b h w d1 l (ld g) -> b h w (d1 ld) (l g)", ld = self.L, g = self.g)
        x_d = self.proj_2(x_d)
        return x_d
        
    def forward(self, x):
       x = x + self.IP_MLP(x)
       x = x + self.TP_MLP(x)
       x = x + self.proj_3(x)
       
       return x 
   

class Decoder(nn.Module):
    def __init__(self):
        # self.deconv = nn.ConvTranspose3d(in_channels=64, out_channels=4, kernel_size=(3, 3, 3))
        super().__init__()
        self.cls = nn.Conv3d(in_channels=224, out_channels=4, kernel_size=(1, 1, 1))
    def forward(self, X):
        return self.cls(X)
    
class Overrall(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2 = nn.Sequential(
            DoubleConv2d(4, 16), 
            DoubleConv2d(16, 32)
        )
        self.conv3 = nn.Sequential(
            DoubleConv3d(32, 64)
        )
        mlpp_layers_1 = []
        for _ in range(15):
            mlpp_layers_1.append(MLPP(64, 8))
            mlpp_layers_1.append(nn.BatchNorm3d(64))
        
        self.mlpps_1 = nn.Sequential(*mlpp_layers_1)
        
        mlpp_layers_2 = []
        for _ in range(4):
            mlpp_layers_2.append(MLPP(64, 8))
            mlpp_layers_2.append(nn.BatchNorm3d(64))
        
        self.mlpps_2 = nn.Sequential(*mlpp_layers_2)
        self.up_sample = nn.Upsample(scale_factor=2)
        self.decoder = Decoder()
    def forward(self, x):
        x_conv2 = self.conv2(x) # b 32 128 ^ 3
        x_conv3 = self.conv3(x_conv2) # b 64 64 ^3
        x_mlpp1 = self.mlpps_1(x_conv3) #b 64 64 ^3
        x_mlpp2 = self.mlpps_2(x_mlpp1) #b 64 64 ^3

        x_deep = torch.cat([x_conv3, x_mlpp1, x_mlpp2], dim = 1) # b 192 64^3 
        x_deep = self.up_sample(x_deep) # b 192 128 ^ 3
        features = torch.cat((x_conv2, x_deep), dim = 1) # 192 + 32
        
        return self.decoder(features)
    
if __name__ == "__main__":
    
    
    x = torch.rand(4, 4, 128, 128, 128)
    ov = Overrall()
    y = ov(x)
    print(y.shape)
    print(y)
    