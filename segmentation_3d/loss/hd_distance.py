
import torch 
def hausdorff_distance(prediction, target):
    # Flatten prediction and target tensors
    prediction = prediction.contiguous().view(-1)
    target = target.contiguous().view(-1)

    # Find non-zero indices
    prediction_indices = torch.nonzero(prediction).float()
    target_indices = torch.nonzero(target).float()

    # Compute pairwise distances between non-zero indices
    distances_pred_to_target = torch.cdist(prediction_indices, target_indices)
    distances_target_to_pred = torch.cdist(target_indices, prediction_indices)

    # Compute Hausdorff distance
    hd_distance = torch.max(torch.max(torch.min(distances_pred_to_target, dim=1)[0]), 
                            torch.max(torch.min(distances_target_to_pred, dim=1)[0]))

    
    return hd_distance.item()

if __name__ == "__main__":
    output = torch.rand(4, 4, 64, 64, 64)
    target = torch.randint(0, 4, (4, 4, 64, 64, 64))
    hausdorff_distance(output)