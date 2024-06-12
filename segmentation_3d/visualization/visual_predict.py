import sys 
sys.path.append('..')
from main2 import Wapper
import torch 
import pandas as pd 
import numpy as np 
from config import config as cfg 
from einops import rearrange
from matplotlib import pyplot as plt 
import cv2 

checkpoint_path = '/home/bai_gairui/seg_3d/model2/segmentation_3d/logs/resoulution-128-train/version_16/checkpoints/epoch=999-step=92000.ckpt'
device = torch.device("cuda:6")
model = Wapper.load_from_checkpoint(checkpoint_path, map_location = device)

step_w , step_h, step_d = 16, 16, 16    
base_root = '/home/bai_gairui/seg_3d/model2/processing_data_2/test/' 
folder_name =   'BraTS19_2013_8_1'
feature_file = base_root + folder_name + "/feature.npy"
label_file = base_root + folder_name + "/label.npy"
vertice_file = base_root + folder_name + "/vertices_128.csv"
vertice = pd.read_csv(vertice_file)
vertice = vertice[vertice["count"] > 0]
vertice.index = range(len(vertice))
num = vertice["count"].argmax()
i, j , k = vertice.loc[num, "i"], vertice.loc[num, "j"], vertice.loc[num, "k"]

imgs = np.load(feature_file)
image = imgs.copy()
imgs = imgs.astype("float32")
imgs = imgs[step_h * i : step_h * i + cfg.resolution, step_w * j : step_w * j + cfg.resolution, step_d * k : step_d * k + cfg.resolution, :]
image = image[step_h * i : step_h * i + cfg.resolution, step_w * j : step_w * j + cfg.resolution, step_d * k : step_d * k + cfg.resolution, :]
label = torch.from_numpy(np.load(label_file))
print([(label == i).sum() for i in range(4)])
label = label[step_h * i : step_h * i + cfg.resolution, step_w * j : step_w * j + cfg.resolution, step_d * k : step_d * k + cfg.resolution]

label[label == 4] = 3 
mask = imgs.sum(-1) > 0
for k in range(4):
    x = imgs[..., k]  
    y = x[mask]
    x[mask] -= y.mean()
    x[mask] /= (y.std() + 1e-5)
    imgs[..., k] = x
feature = torch.from_numpy(imgs)

feature = feature[None, ...]
feature.shape 
feature = feature.to(model.device)
feature = rearrange(feature, "b h w d c -> b c h w d")
feature = feature.to(torch.float32)
output = model(feature)

output = output[0]
output = output.argmax(axis = 0)
res = output.detach().cpu().numpy()
label = label.detach().numpy()

colors = [
    (0, 0, 0),    # 背景 - 黑色
    (0, 0, 255),   # 器官1 - 红色
    (0, 255, 0),  # 器官2 - 绿色
    (255, 0, 0),  # 器官3 - 蓝色
]


ind = 45

fig, axes = plt.subplots(8, 8)
fig.set_size_inches(10, 10)
for j in range(64):
    ind = 40 + j
    demo = label[..., ind]
    demo_pred = res[..., ind] 
    ax = axes.flatten()[j]
    overlay = np.zeros(shape = (128, 128, 3))
    for i in range(4):
        overlay[demo == i] = colors[i]

    background = image[..., ind, 1] 
    plt.imsave('./back.png', background, cmap = "gray")
    new_img = cv2.imread('./back.png',  cv2.IMREAD_GRAYSCALE)
    background  =cv2.cvtColor(new_img, cv2.COLOR_GRAY2BGR)
    result = cv2.addWeighted(background, 0.5, overlay.astype("uint8"), 0.8, 0)
    ax.imshow(result)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(str(ind), fontsize = 8, pad = -3)
plt.savefig(f'./oup64_true_{folder_name}.png')

fig, axes = plt.subplots(8, 8)
fig.set_size_inches(10, 10)
for j in range(64):
    ind = 40 + j
    demo = label[..., ind]
    demo_pred = res[..., ind] 
    ax = axes.flatten()[j]
    overlay = np.zeros(shape = (128, 128, 3))
    for i in range(4):
        overlay[demo_pred == i] = colors[i]

    background = image[..., ind, 1] 
    plt.imsave('./back.png', background, cmap = "gray")
    new_img = cv2.imread('./back.png',  cv2.IMREAD_GRAYSCALE)
    background  =cv2.cvtColor(new_img, cv2.COLOR_GRAY2BGR)
    result = cv2.addWeighted(background, 0.5, overlay.astype("uint8"), 0.8, 0)
    ax.imshow(result)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(str(ind), fontsize = 8, pad = -3)
plt.savefig(f'./oup64_pred_{folder_name}.png')

