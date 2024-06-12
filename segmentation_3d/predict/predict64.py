import torch 
import torch.nn as nn 
from torch.nn import functional as F 
import sys 
sys.path.append('..')
from config import config as cfg


class Demo_model(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, X):
        '''
        X : [b, 4, 64, 64, 64]
        return : [b, 4, 64, 64, 64]
        '''
        return torch.rand(X.shape)





def before_predict(label, X):
    '''
    X : [b, 4, 240, 240, 155]
    label : [b, 240, 240, 155]
    '''
    lis = []
    all_labels = []
    ijk = []
    indices = []
    index = 0
    for i in range(4):
        for j in range(4):
            for k in range(3):
                lis.append(
                    X[...,i * cfg.resolution if i < 3 else cfg.data_h - cfg.resolution - 1: (i + 1) * cfg.resolution if i < 3 else -1, 
                          j * cfg.resolution if j < 3 else cfg.data_w - cfg.resolution - 1: (j + 1) * cfg.resolution if j < 3 else -1, 
                          k * cfg.resolution if k < 2 else cfg.data_d - cfg.resolution - 1: (k + 1) * cfg.resolution if k < 2 else -1]
                )
                all_labels.append(
                    label[:,i * cfg.resolution if i < 3 else cfg.data_h - cfg.resolution - 1: (i + 1) * cfg.resolution if i < 3 else -1, 
                          j * cfg.resolution if j < 3 else cfg.data_w - cfg.resolution - 1: (j + 1) * cfg.resolution if j < 3 else -1, 
                          k * cfg.resolution if k < 2 else cfg.data_d - cfg.resolution - 1: (k + 1) * cfg.resolution if k < 2 else -1]
                )
                ijk.append(( i, j, k))
                indices.append(index)
                index = index + 1
    return torch.cat(lis, dim = 0), torch.cat(all_labels, dim = 0), ijk, indices 

def predict(model, X, ijk, indices):
    '''
    X : [4*4*3, 4, 240, 240, 150]
    '''
    model.eval()
    output = torch.zeros(1, 4, 240, 240, 155)
    for (index, (i, j, k)) in enumerate(ijk):
        temp =X[[indices[index]]].to(model.device)
        temp_output = model(temp)
        temp_output = temp_output.detach().to(torch.device("cpu"))
        temp = temp.detach().to(torch.device("cpu"))
        
        output[0, :, i * cfg.resolution if i < 3 else cfg.data_h - cfg.resolution - 1: (i + 1) * cfg.resolution if i < 3 else -1, 
                          j * cfg.resolution if j < 3 else cfg.data_w - cfg.resolution - 1: (j + 1) * cfg.resolution if j < 3 else -1, 
                          k * cfg.resolution if k < 2 else cfg.data_d - cfg.resolution - 1: (k + 1) * cfg.resolution if k < 2 else -1] = temp_output
        del temp
        del temp_output

    return output
    
    
    
    
if __name__ == "__main__":
    model = Demo_model()
    X = torch.rand(1, 4, 240, 240, 155)
    # model(X)
    label = torch.rand(1, 240, 240, 155)
    output, label, ijk, indices = before_predict(label, X)
    print(output.shape)
    print(label.shape)
    output = predict(model, output, ijk, indices)
    print(output.shape)
    