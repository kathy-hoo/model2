import torch
import torch.nn.functional as F
import numpy as np 
import seg_metrics.seg_metrics as sg

def hausdorff_distance(seg1, seg2):
    # Get nonzero points from segmentations
    seg1_points = torch.nonzero(seg1, as_tuple=True)
    seg2_points = torch.nonzero(seg2, as_tuple=True)

    # Compute pairwise Euclidean distances
    distances1 = torch.cdist(seg1_points[1:], seg2_points[1:], p=2).min(dim=1).values
    distances2 = torch.cdist(seg2_points[1:], seg1_points[1:], p=2).min(dim=1).values

    # Compute Hausdorff distance
    hausdorff_dist = torch.max(torch.max(distances1), torch.max(distances2))

    return hausdorff_dist.item()


if __name__ == "__main__":
    # Example usage
    labels = [0, 1, 2]
    gdth_img = np.random.randint(0, 3, size = (128, 128, 128))
    pred_img = np.random.randint(0, 3, size = (128, 128, 128))
    metrics = sg.write_metrics(labels=labels[1:], gdth_img=gdth_img, pred_img=pred_img, metrics=["hd"])
    print(metrics[0]["hd"])