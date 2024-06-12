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

image_set = "/home/bai_gairui/seg_3d/model2/udc/submit2/images/"
modalities = ["flair", "t1", "t1ce", "t2"]
step_w , step_h, step_d = 16, 16, 16

image_files = os.listdir(image_set)


for f in image_files:
    img_file = image_set + f 
    imgs = np.load(img_file)
    label_file = img_file.replace("images", "labels")
    label = np.load(label_file)
    
    label[label == 3] = 1
    label[label == 4] = 2
    label[label == 5] = 3
    json_file = label_file.replace("labels", "vertices").replace("npy", "csv")
    
    
    step_iter_w, step_iter_h, step_iter_d = (label.shape[0] - 128) // step_w, (label.shape[1] - 128) // step_h, (label.shape[2] - 128) // step_d
        
    step_lis = []
    img_non_zeros = []
    for i in range(step_iter_h):
        for j in range(step_iter_w):
            for k in range(step_iter_d):
                demo = label[step_h * i : step_h * i + 128, step_w * j : step_w * j + 128, step_d * k : step_d * k + 128]
                img_non_zeros.append((demo.sum(axis = -1) > 0).sum())
                step_lis.append((i, j, k))
    df = pd.concat((pd.DataFrame([list(i) for i in step_lis], columns = list("ijk")), pd.DataFrame(img_non_zeros, columns = ["count"])), axis = 1)
    df.to_csv(json_file)
    # pd.DataFrame(
    #     {
    #         "count": [(label == i ).sum() for i in range(4)], 
    #         "label" : [i for i in range(4)]
    #     }
    # ).to_csv(json_file, index = False)
    # with open(json_file, 'w', encoding="utf8") as f:
    #     f.write(json.dumps(np.unique(label).tolist()))
    print(label_file)
    


    