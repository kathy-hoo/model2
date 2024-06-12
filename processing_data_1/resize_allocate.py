import numpy as np 
import os 

def relloc(img, target_type):
    img = img.astype(target_type)
    return img 

def load_img(path):
    return np.load(path)

def main():
    base_root = '/home/kathy/model2/processing_data_1/train/'
    folders = [base_root + i + "/" for i in os.listdir(base_root)]
    for folder in folders:
        feature = load_img(folder + "feature.npy")
        label = load_img(folder + "label.npy")
        feature = relloc(feature, "int16")
        label = relloc(label, "int8")
        np.save(folder + "feature", feature)
        np.save(folder + "label", label)
    print(folder)
if __name__ == "__main__":
    main()
    