import torch 
import torch.nn as nn 
def Dice(output, target, eps=nn.Parameter(torch.tensor(1e-5), requires_grad=False)):
    target = target.float()
    num = 2 * (output * target).sum()
    eps = eps.type_as(output)
    den = output.sum() + target.sum() + eps
    return 1.0 - num/den


def softmax_dice2(output, target):
    '''
    The dice loss for using softmax activation function
    :param output: (b, num_class, d, h, w)
    :param target: (b, d, h, w)
    :return: softmax dice loss
    '''
    loss1 = Dice(output[:, 1, ...], (target == 1).float())
    loss2 = Dice(output[:, 2, ...], (target == 2).float())
    loss3 = Dice(output[:, 3, ...], (target == 3).float())

    return loss1 + loss2 + loss3 , 1-loss1.data, 1-loss2.data, 1-loss3.data

class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, X, y):
        '''
        X : b num_classe + 1 , h, w , d
        y : b , h, w , d
        '''
        return softmax_dice2(X, y)[0]
    


if __name__ == "__main__":
    model_output = torch.rand(4, 5, 64, 64, 64)
    target = torch.randint(size=(4, 64, 64, 64), low=0, high=5)
    
    loss = torch.nn.CrossEntropyLoss()
    print(loss(model_output, target))