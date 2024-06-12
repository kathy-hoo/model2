import torch 
import torch.nn as nn 
from einops import rearrange
import sys 
sys.path.append('..')
from components.ccs import CCS



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


class WeightedPermuteMLP(nn.Module):
    def __init__(self, dim, segment_dim=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.segment_dim = segment_dim

        self.mlp_c = nn.Linear(dim, dim, bias=qkv_bias)
        self.mlp_h = nn.Linear(dim, dim, bias=qkv_bias)
        self.mlp_w = nn.Linear(dim, dim, bias=qkv_bias)
        
        self.reweight = Mlp(dim, dim // 4, dim *3)
        
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)



    def forward(self, x):
        B, H, W, C = x.shape

        S = C // self.segment_dim
        h = x.reshape(B, H, W, self.segment_dim, S).permute(0, 3, 2, 1, 4).reshape(B, self.segment_dim, W, H*S) # H * S = C， S * 8 = C  # B H W N S -> B N W H S -> B N W (HS)

        h = self.mlp_h(h).reshape(B, self.segment_dim, W, H, S).permute(0, 3, 2, 1, 4).reshape(B, H, W, C)

        w = x.reshape(B, H, W, self.segment_dim, S).permute(0, 1, 3, 2, 4).reshape(B, H, self.segment_dim, W*S) # B H W N S -> B H N W S -> B H N (WS)
        w = self.mlp_w(w).reshape(B, H, self.segment_dim, W, S).permute(0, 1, 3, 2, 4).reshape(B, H, W, C)

        c = self.mlp_c(x)
        
        a = (h + w + c).permute(0, 3, 1, 2).flatten(2).mean(2)
        a = self.reweight(a).reshape(B, C, 3).permute(2, 0, 1).softmax(dim=0).unsqueeze(2).unsqueeze(2)

        x = h * a[0] + w * a[1] + c * a[2]

        x = self.proj(x)
        x = self.proj_drop(x)

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
        h = rearrange(x, "b w h d (s n) -> b w d s n h", n = N)
        # h = h @ mlp_h
        h = self.mlp_h(h)
        h = rearrange(h, "b w d s n h -> b w h d (s n)")
        
        w = rearrange(x, "b w h d (s n) -> b h d s n w",n = N)
        # w = w @ mlp_w
        w = self.mlp_w(w)
        w = rearrange(w, "b w d s n h -> b w h d (s n)")
        
        
        d = rearrange(x, "b w h d (s n) -> b w h s n d", n = N)
        # d = d @ mlp_d 
        d = self.mlp_d(d)
        
        d = rearrange(d, "b w d s n h -> b w h d (s n)")
        

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
if __name__ == "__main__":
    device = torch.device("cuda:3")
    x = torch.rand(4, 512, 32, 32, 32).to(device)
    wp = Pc_mlp(hwd=32, dim = 512, out_dim=256).to(device)
    oup = wp(x)
    print(oup.shape)
