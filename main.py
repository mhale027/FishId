import os
import csv
import numpy as np
import pandas as pd
import datetime
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2


os.chdir("projects/kaggle/FishId")

folder = "input/train"
species = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']



files = pd.DataFrame(columns = ['label', 'image'])

def list_files():
    for i in species:
        paths = { 'label' : np.repeat(i, len(os.listdir(os.path.join("input", "train", i)))),
            'image' : os.listdir(os.path.join("input", "train", i)) }
        if i == "ALB":
            fil = pd.DataFrame(paths, columns = ['label', 'image'])
        else: 
            fil = fil.append(pd.DataFrame(paths), ignore_index = True)
    return(fil)   

fish_files = files[files['label'] != "NoF"]

markers = pd.DataFrame.from_csv("markers.csv")

img = cv2.imread("input/train/"+str(files['label'][1])+str(files['image'][1]))






def load_train():
    X_train = []
    X_train_id = []
    y_train = []
    start_time = time.time()

    print('Read train images')
    folders = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
    for fld in folders:
        index = folders.index(fld)
        print('Load folder {} (Index: {})'.format(fld, index))
        path = os.path.join('input', 'train', fld, '*.jpg')
        files = glob.glob(path)
        for fl in files:
            flbase = os.path.basename(fl)
            img = get_im_cv2(fl)
            X_train.append(img)
            X_train_id.append(flbase)
            y_train.append(index)
