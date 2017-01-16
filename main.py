import os
import csv
import numpy as np
import pandas as pd
import datetime
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2


os.chdir("/home/matt/projects/kaggle/FishId")

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

files = list_files()
fish_files = files[files['label'] != "NoF"]
markers = pd.DataFrame.from_csv("markers.csv")

for i in range(1):
    m = markers[i:i+1]
    file = files[files['image'] == m.index[0]].iloc[i]['label']
    num = markers.index[i]
    img = cv2.imread("input/train/"+str(file)+'/'+str(num),1)
    buffer = 20
    x1 = int(min(m.iloc[0][0], m.iloc[0][2])) - buffer
    y1 = int(min(m.iloc[0][1], m.iloc[0][3])) - buffer
    x2 = int(max(m.iloc[0][0], m.iloc[0][2])) + buffer
    y2 = int(max(m.iloc[0][1], m.iloc[0][3])) + buffer


    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)


print(img)

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


