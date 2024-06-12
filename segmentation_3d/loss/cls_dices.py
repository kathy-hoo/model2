
import torch 
import torch.nn as nn 
from torch.nn import functional as F 

def dice_score(o, t, eps=1e-8):
    num = 2*(o*t).sum() + eps
    den = o.sum() + t.sum() + eps
    return num/den


def softmax_output_dice(output, target):
    ret = [] # 124, 14, 4

    # whole
    o = output > 0; t = target > 0 # ce
    ret += dice_score(o, t),
    # core
    o = (output == 1) | (output == 3)
    t = (target == 1) | (target == 3)
    ret += dice_score(o, t),
    # active
    o = (output == 3);t = (target == 3)
    ret += dice_score(o, t),

    return ret


if __name__ == "__main__":
    X = torch.rand(1, 4, 128, 128, 128)
    y = torch.randint(1, 4 , size = (128, 128, 128))
    X = X.argmax(dim = 1)
    ret = softmax_output_dice(X, y)
    print(ret)