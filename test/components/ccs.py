import torch 
import torch.nn as nn 
from einops import rearrange


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
    x = torch.rand(4, 32, 32, 16, 32, 32)
    ccs = CCS(hwd=32)
    print(ccs(x).shape )