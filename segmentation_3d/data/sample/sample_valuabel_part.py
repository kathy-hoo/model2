import os 
import nibabel as nib 
import numpy as np 

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
    'mode' : "train"
}
modalities = ["flair", "t1", "t1ce", "t2"]
step_w , step_h, step_d = 16, 16, 16
step_iter_w, step_iter_h, step_iter_d = (240 - 64) // step_w, (240 - 64) // step_h, (155 - 64) // step_d
save_root = '/home/kathy/model2/processing_data/test/'
with open(train_set["flist"], 'r', encoding="utf8") as f:
    files = f.read().split("\n")

for f in files:
    label_file = train_set["root"] + f'{f}/' + f'{f}_seg.nii'
    label = np.asanyarray(nib_load(label_file), dtype='int64', order='C')

    all_files = [train_set["root"] + f'{f}/' + f'{f}_{modal}.nii' for modal in modalities]
    imgs = np.concatenate([np.asanyarray(nib_load(file)[..., None], dtype='int64', order='C') for file in all_files], axis = -1)
    
    step_lis = []
    img_non_zeros = []
    for i in range(step_iter_h):
        for j in range(step_iter_w):
            for k in range(step_iter_d):
                demo = label[step_h * i : step_h * i + 64, step_w * j : step_w * j + 64, step_d * k : step_d * k + 64]
                img_non_zeros.append((demo.sum(axis = -1) > 0).sum())
                step_lis.append((i, j, k))
    
    i, j, k = step_lis[np.argmax(img_non_zeros )]
    demo = imgs[step_h * i : step_h * i + 64, step_w * j : step_w * j + 64, step_d * k : step_d * k + 64, :]
    label = label[step_h * i : step_h * i + 64, step_w * j : step_w * j + 64, step_d * k : step_d * k + 64]
    save_dir = save_root + f + "/"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    np.save(save_dir + "feature", demo)
    np.save(save_dir + "label", label)
    
    