import sys 
sys.path.append('../..')
import os 
# os.chdir('/home/bai_gairui/seg_3d/model2')
from main6 import Wapper
import torch 
import pandas as pd 
import numpy as np 
from config import config as cfg 
from einops import rearrange
from matplotlib import pyplot as plt 
import cv2 
import json


proj_info = "VT_Unet"
if not os.path.exists(f"./{proj_info}"):
    os.mkdir(proj_info)

def tailor_and_concat(x, model, k = 0):
    temp = []
    for i in range(4):
        for j in range(4):
            
            temp.append(x[..., i * 128:(i + 1) * 128, j * 128 : (j + 1) * 128, k : k+ 128])


    y = x[..., :128].clone()
    y = y.cpu().detach().numpy()

    for i in range(len(temp)):
        temp[i] = model(temp[i])
        temp[i] = temp[i].cpu().detach().numpy()
    for i in range(4):
        for j in range(4):
            y[...,  i * 128:(i + 1) * 128, j * 128 : (j + 1) * 128, k : k+ 128] = temp[i * 4 + j].argmax(axis = 1)



    return y

with open("/home/bai_gairui/seg_3d/model2/udc/udc_log.json", "r", encoding="utf8") as f:
    res = json.loads(f.read())
    
checkpoint_path = res[proj_info]["model"]
device = torch.device("cuda:3")
is_plot = False
model = Wapper.load_from_checkpoint(checkpoint_path, map_location = device)
model.eval()

mode = "test"
with open("/home/bai_gairui/seg_3d/model2/udc/info.json", 'r', encoding="utf8") as f:
    res = json.loads(f.read())
folder_list = ["/home/bai_gairui/seg_3d/model2/udc/vertices/" + folder for folder in res[mode]]
    
for index in range(len(folder_list)):
    if not os.path.exists(f"./{proj_info}/{index}"):
        os.mkdir(f"./{proj_info}/{index}")
    step_w , step_h, step_d = 16, 16, 16      
    feature_file = folder_list[index].replace("vertices", "images").replace("csv", "npy")
    label_file = folder_list[index].replace("vertices", "labels_small").replace("csv", "npy")
    vertice_file = folder_list[index]
    vertice = pd.read_csv(vertice_file)
    vertice = vertice[vertice["count"] > 0]
    vertice.index = range(len(vertice))
    num = vertice["count"].argmax()
    i, j , k = vertice.loc[num, "i"], vertice.loc[num, "j"], vertice.loc[num, "k"]
    
    imgs = np.load(feature_file)
    imgs = (imgs - imgs.mean()) / (imgs.std() + 1e-5)

    labels = np.load(label_file)
    labels[labels == 3] = 1
    labels[labels == 4] = 2
    labels[labels == 5] = 3
    
    labels  = labels[..., k : k + cfg.resolution]
    origin_img = imgs.copy()
    imgs = imgs.astype("float32")
    imgs = torch.from_numpy(imgs).to(model.device)
    imgs = imgs.unsqueeze(0)
    imgs = imgs.unsqueeze(0) 
    y = tailor_and_concat(imgs, model) 
    y = y[0][0]
    
    
    fig, axes = plt.subplots(11, 11) # h, w, d = 240, 240, 128
    fig.set_size_inches(11, 11)

    for num in range(11 ** 2):
        ind = num + 3
        colors = [
                            (0, 0, 0),    # 背景 - 黑色
                            (0, 0, 255),   # 器官1 - 红色
                            (0, 255, 0),  # 器官2 - 绿色
                            (255, 0, 0),  # 器官3 - 蓝色
                        ]

        overlay_true = np.zeros(shape = (512, 512, 3))
        for i in range(4):
            overlay_true[labels[..., ind] == i] = colors[i]
            
        plt.imsave('./origin.png', origin_img[..., ind])
        background = cv2.imread('./origin.png', cv2.IMREAD_GRAYSCALE)
        background  =cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)
        result = cv2.addWeighted(background, 0.5, overlay_true.astype('uint8'), 0.8, 0)
        ax = axes.flatten()[num]
        ax.imshow(result)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.savefig(f"./{proj_info}/{index}/origin.png")
    
    fig, axes = plt.subplots(11, 11) # h, w, d = 240, 240, 128
    fig.set_size_inches(11, 11)
    
    

    for num in range(11 ** 2):
        ind = num + 3
        colors = [
                            (0, 0, 0),    # 背景 - 黑色
                            (0, 0, 255),   # 器官1 - 红色
                            (0, 255, 0),  # 器官2 - 绿色
                            (255, 0, 0),  # 器官3 - 蓝色
                        ]

        overlay_true = np.zeros(shape = (512, 512, 3))
        for i in range(4):
            overlay_true[y[..., ind] == i] = colors[i]
            
        plt.imsave('./origin.png', origin_img[..., ind])
        background = cv2.imread('./origin.png', cv2.IMREAD_GRAYSCALE)
        background  =cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)
        result = cv2.addWeighted(background, 0.5, overlay_true.astype('uint8'), 0.8, 0)
        ax = axes.flatten()[num]
        ax.imshow(result)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.savefig(f"./{proj_info}/{index}/pred.png")
