import torch 
import torch.nn as nn 
def Dice(output, target, eps=nn.Parameter(torch.tensor(1e-5), requires_grad=False)):
    target = target.float()
    num = 2 * (output * target).sum()
    eps = eps.type_as(output)
    den = output.sum() + target.sum() + eps
    return 1.0 - num/den

def dice_score(prediction, target, num_classes, smooth=1e-6):
    dice_scores = []
    for class_index in range(1, num_classes):
        class_prediction = prediction[:, class_index, ...]
        class_target = (target == class_index).float()
        
        # Flatten tensors
        class_prediction = class_prediction.contiguous().view(-1)
        class_target = class_target.contiguous().view(-1)
        
        # Calculate intersection and union
        intersection = torch.sum(class_prediction * class_target)
        union = torch.sum(class_prediction) + torch.sum(class_target)
        
        # Calculate Dice score
        dice = (2. * intersection + smooth) / (union + smooth)
        dice_scores.append(dice)
    
    return dice_scores

def softmax_dice2(output, target, with_zeros = True):
    '''
    The dice loss for using softmax activation function
    :param output: (b, num_class, d, h, w)
    :param target: (b, d, h, w)
    :return: softmax dice loss
    '''
    loss0 = Dice(output[:, 0, ...], (target == 0).float()) if with_zeros else 0
    loss1 = Dice(output[:, 1, ...], (target == 1).float())
    loss2 = Dice(output[:, 2, ...], (target == 2).float())
    loss3 = Dice(output[:, 3, ...], (target == 3).float())
    if with_zeros:
        return loss0 + loss1 + loss2 + loss3 , 1 - loss0.data,  1-loss1.data, 1-loss2.data, 1-loss3.data
    return loss0 + loss1 + loss2 + loss3 ,   1-loss1.data, 1-loss2.data, 1-loss3.data
class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, X, y):
        '''
        X : b num_classe + 1 , h, w , d
        y : b , h, w , d
        '''
        return softmax_dice2(X, y)[0]
    
def mask_dice_loss(prediction, target, num_classes, smooth=1e-6):
    total_loss = 0.0
    dice_losses = [0 for _ in range(num_classes - 1)]
    for class_index in range(1, num_classes):
        class_prediction = prediction[:, class_index, ...]
        class_target = (target == class_index).float()
        
        # Compute intersection and union
        intersection = torch.sum(class_prediction * class_target)
        class_pred_sum = torch.sum(class_prediction)
        class_target_sum = torch.sum(class_target)
        
        # Calculate Dice loss only if the class exists in the ground truth
        if class_target_sum > 0:
            dice = (2. * intersection + smooth) / (class_pred_sum + class_target_sum + smooth)
            # class_loss = 1.0 - dice
            # dice_losses[class_index - 1] = class_loss
            dice_losses[class_index - 1] = dice
    
    return dice_losses
def mask_dice_score(prediction, target, num_classes, smooth=1e-6):
    total_loss = 0.0
    dice_scores = [-1, -1, -1]
    for class_index in range(1, num_classes):
        class_prediction = prediction[:, class_index, ...]
        class_target = (target == class_index).float()
        
        # Compute intersection and union
        intersection = torch.sum(class_prediction * class_target)
        class_pred_sum = torch.sum(class_prediction)
        class_target_sum = torch.sum(class_target)
        
        # Calculate Dice loss only if the class exists in the ground truth
        if class_target_sum > 0:
            dice = (2. * intersection + smooth) / (class_pred_sum + class_target_sum + smooth)
            
            dice_scores[class_index - 1] = dice
    
    return dice_scores


def dice_score_per_class(output, target, class_index):
    # Reshape output and target
    output = output[:, class_index, ...].contiguous().view(-1)
    
    target = (target == class_index).float().view(-1)

    # Compute intersection and union
    intersection = torch.sum(output * target)
    union = torch.sum(output) + torch.sum(target)

    # Compute Dice score
    dice_score = (2.0 * intersection) / (union + 1e-6)  # Add epsilon to avoid division by zero

    return dice_score

def dice_score_per_class_list(output, target, class_indices, weighted = False):
    weight = [1 for i in class_indices]
    if weighted:
        for (i, class_index) in enumerate(class_indices):
            weight[i] = (target == class_index).sum()
    weight = torch.tensor(weight)
    weight = weight /( weight.sum() +1e-6)
    dice_scores = torch.zeros((len(class_indices), ))
    for (i, class_index) in enumerate(class_indices):
        dice_scores[i] = dice_score_per_class(output, target, class_index)
    return weight.dot(dice_scores)
        
    
def dice_score_all(output, target, num_classes):
    # Reshape output and target
    output = output.view(-1, num_classes)
    target = target.view(-1)

    # Compute intersection and union for each class
    intersection = torch.sum(output * target.view(-1, 1), dim=0)
    union = torch.sum(output, dim=0) + torch.sum(target.view(-1, 1), dim=0)

    # Compute Dice score for each class
    dice_scores = (2.0 * intersection) / (union + 1e-6)  # Add epsilon to avoid division by zero

    return dice_scores  # Return average Dice score over classes

if __name__ == "__main__":
    model_output = torch.rand(4, 4, 64, 64, 64)
    model_output = torch.nn.functional.softmax(model_output, dim = 1)
    target = torch.randint(size=(4, 64, 64, 64), low=0, high=4)
    dice_score = dice_score_per_class(model_output, target, 1)
    # dice_all = dice_score_all(model_output, target, 4)
    dice_all = dice_score_per_class_list(model_output, target, [1, 2], weighted=True)
    dice_all_2 = dice_score_per_class_list(model_output, target, [1, 2], weighted=False)
    print(dice_score)
    print(dice_all)
    print(dice_all_2)