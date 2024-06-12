import os 
import nibabel as nib 
import numpy as np 
import pandas as pd 
import json 
def nib_load(file_name):
    if not os.path.exists(file_name):
        print('Invalid file name, can not find the file!')

    proxy = nib.load(file_name)
    data = proxy.get_data()
    proxy.uncache()
    return data

train_set = {
    # 'root': 'path to training set',
    'root': '/home/kathy/dataset/BraTS2020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/',
    'flist': '/home/kathy/model2/test/test.txt',
    'has_label': True,
    'mode' : "test"
}
modalities = ["flair", "t1", "t1ce", "t2"]
step_w , step_h, step_d = 16, 16, 16
step_iter_w, step_iter_h, step_iter_d = (240 - 128) // step_w, (240 - 128) // step_h, (155 - 128) // step_d
with open(train_set["flist"], 'r', encoding="utf8") as f:
    files = f.read().split("\n")

for f in files:
    label_file = f"./test/{f}/label.npy"
    label = np.load(label_file)
    label[label == 4] = 3
    img_file = f"./test/{f}/feature.npy"
    json_file = f"./test/{f}/category.csv"
    imgs = np.load(img_file)
    
    # step_lis = []
    # img_non_zeros = []
    # for i in range(step_iter_h):
    #     for j in range(step_iter_w):
    #         for k in range(step_iter_d):
    #             demo = label[step_h * i : step_h * i + 128, step_w * j : step_w * j + 128, step_d * k : step_d * k + 128]
    #             img_non_zeros.append((demo.sum(axis = -1) > 0).sum())
    #             step_lis.append((i, j, k))
    # df = pd.concat((pd.DataFrame([list(i) for i in step_lis], columns = list("ijk")), pd.DataFrame(img_non_zeros, columns = ["count"])), axis = 1)
    pd.DataFrame(
        {
            "count": [(label == i ).sum() for i in range(4)], 
            "label" : [i for i in range(4)]
        }
    ).to_csv(json_file, index = False)
    # with open(json_file, 'w', encoding="utf8") as f:
    #     f.write(json.dumps(np.unique(label).tolist()))
    print(label_file)
    


    