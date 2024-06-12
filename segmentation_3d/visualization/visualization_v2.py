import sys 
sys.path.append('..')
import os 
# os.chdir('/home/bai_gairui/seg_3d/model2')
from main2 import Wapper
import torch 
import pandas as pd 
import numpy as np 
from config import config as cfg 
from einops import rearrange
from matplotlib import pyplot as plt 
import cv2 

def tailor_and_concat(x, model, k = 0):
    temp = []

    temp.append(x[..., :128, :128, k : k+ 128])
    temp.append(x[..., :128, 112:240, k: k+ 128])
    temp.append(x[..., 112:240, :128, k : k+128])
    temp.append(x[..., 112:240, 112:240, k: k + 128])
    # temp.append(x[..., :128, :128, 27:155])
    # temp.append(x[..., :128, 112:240, 27:155])
    # temp.append(x[..., 112:240, :128, 27:155])
    # temp.append(x[..., 112:240, 112:240, 27:155])

    y = x[..., :128].clone()
    y = y.cpu().detach().numpy()

    for i in range(len(temp)):
        temp[i] = model(temp[i])
        temp[i] = temp[i].cpu().detach().numpy()

    y[..., :128, :128, k: k+128] = temp[0]
    y[..., :128, 128:240, k : k+128] = temp[1][..., :, 16:128, :]
    y[..., 128:240, :128, k : k+128] = temp[2][..., 16:128, :, :]
    y[..., 128:240, 128:240, k: k+ 128] = temp[3][..., 16:128, 16:128, :]
    # y[..., :128, :128, 128:155] = temp[4][..., 96:123]
    # y[..., :128, 128:240, 128:155] = temp[5][..., :, 16:128, 96:123]
    # y[..., 128:240, :128, 128:155] = temp[6][..., 16:128, :, 96:123]
    # y[..., 128:240, 128:240, 128:155] = temp[7][..., 16:128, 16:128, 96:123]

    return y

checkpoint_path = '/home/bai_gairui/seg_3d/model2/segmentation_3d/logs/resoulution-128-train/version_16/checkpoints/epoch=999-step=92000.ckpt'
device = torch.device("cuda:6")
is_plot = False
model = Wapper.load_from_checkpoint(checkpoint_path, map_location = device)
model.eval()
step_w , step_h, step_d = 16, 16, 16    
base_root = '/home/bai_gairui/seg_3d/model2/processing_data_2/test/' 


folder_name_list = [
    "BraTS19_TCIA01_131_1",
    # "BraTS19_TCIA01_390_1",
    # "BraTS19_TCIA01_499_1",
    # "BraTS19_TCIA02_117_1"
]
output1 = "output2"
folder_name_list_ind = [
    [i for i in range(-11 + 107 - 5 , -11 + 107 + 5)], 
    [i for i in range(-11 + 96 - 5,   -11 + 96 + 5)], 
    [i for i in range(-11 + 58 - 5,   -11 + 58 + 5)], 
    [i for i in range(-11 + 70 - 5,   -11 + 70 + 5)], 
]

for (folder_indx , folder_name) in zip(folder_name_list_ind, folder_name_list):
    # make dir if not exist 
    if not os.path.exists(f'./{output1}/{folder_name}'):
        os.mkdir(f'./{output1}/{folder_name}')
    if not os.path.exists(f'./{output1}/{folder_name}/pred'):
        os.mkdir(f'./{output1}/{folder_name}/pred')
    if not os.path.exists(f'./{output1}/{folder_name}/true'):
        os.mkdir(f'./{output1}/{folder_name}/true')
    
    # load feature file
    feature_file = base_root + folder_name + "/feature.npy"
    # load label file
    label_file = base_root + folder_name + "/label.npy"
    # load vertice file 
    vertice_file = base_root + folder_name + "/vertices_128.csv"
    vertice = pd.read_csv(vertice_file)
    vertice = vertice[vertice["count"] > 0]
    vertice.index = range(len(vertice))
    max_index = vertice["count"].argmax()
    _, _ , k = vertice.loc[max_index, "i"], vertice.loc[max_index, "j"], vertice.loc[max_index, "k"]
    
    imgs = np.load(feature_file)
    labels = np.load(label_file)
    labels  = labels[..., k : k + cfg.resolution] # 240, 240, 128
    image = imgs.copy()
    origin_img = imgs.copy()
    imgs = imgs.astype("float32")

    for i in range(4):
        mask = imgs.sum(-1) > 0
        for modal_index in range(4):
            x = imgs[..., modal_index]  
            y = x[mask]
            x[mask] -= y.mean()
            x[mask] /= (y.std() + 1e-5)
            imgs[..., modal_index] = x
    imgs = torch.from_numpy(imgs).to(model.device)
    imgs = imgs.unsqueeze(0)
    imgs = rearrange(imgs, "b h w d c -> b c h w d")
    y = tailor_and_concat(imgs, model) 
    y = y[0]
    y = y.argmax(axis = 0)
    # plot true 
    

    for ind in folder_indx:
        colors = [
                    (0, 0, 0),    # 背景 - 黑色
                    (0, 0, 255),   # 器官1 - 红色
                    (0, 255, 0),  # 器官2 - 绿色
                    (255, 0, 0),  # 器官3 - 蓝色
                ]

        overlay_true = np.zeros(shape = (240, 240, 3))
        for i in range(4):
            overlay_true[labels[..., ind] == i] = colors[i]
            
        plt.imsave('./origin.png', origin_img[..., ind, 1])
        
        background = cv2.imread('./origin.png', cv2.IMREAD_GRAYSCALE)
        background  =cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)
        result = cv2.addWeighted(background, 0.5, overlay_true.astype('uint8'), 0.8, 0)
        plt.imshow(result)
        plt.xticks([])
        plt.yticks([])
        plt.title(str(k + ind), fontsize = 8, pad = -3)
        plt.savefig(f'./{output1}/{folder_name}/true/{ind}.png')
        print(result.shape)
    
    
        colors = [
                    (0, 0, 0),    # 背景 - 黑色
                    (0, 0, 255),   # 器官1 - 红色
                    (0, 255, 0),  # 器官2 - 绿色
                    (255, 0, 0),  # 器官3 - 蓝色
                ]

        overlay_pred = np.zeros(shape = (240, 240, 3))
        for i in range(4):
            overlay_pred[y[..., ind] == i] = colors[i]
            
        
        result = cv2.addWeighted(background, 0.5, overlay_pred.astype('uint8'), 0.8, 0)
        
        plt.imshow(result)
        plt.xticks([])
        plt.yticks([])
        plt.title(str(k + ind), fontsize = 8, pad = -3)
        plt.savefig(f'./{output1}/{folder_name}/pred/{ind}.png')
    
    


