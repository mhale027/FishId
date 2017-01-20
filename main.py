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
#
# for i in range(len(markers)):
#     m = markers[i:i+1]
#     file = files[files['image'] == m.index[0]].iloc[i]['label']
#     num = markers.index[i]
#     img = cv2.imread("input/train/"+str(file)+'/'+str(num),1)
#     buffer = 20
#     x1 = int(min(m.iloc[0][0], m.iloc[0][2])) - buffer
#     y1 = int(min(m.iloc[0][1], m.iloc[0][3])) - buffer
#     x2 = int(max(m.iloc[0][0], m.iloc[0][2])) + buffer
#     y2 = int(max(m.iloc[0][1], m.iloc[0][3])) + buffer
#
#
#     cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)


# print(img)

img = cv2.imread(os.path.join('input/train/YFT/',os.listdir('input/train/YFT')[8]),1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# img = cv2.resize(img, (500, 300))
# gray = cv2.resize(gray, (500, 300))

ALB_cascade = cv2.CascadeClassifier('outputs/cascades/ALB/cascade.xml')
BET_cascade = cv2.CascadeClassifier('outputs/cascades/BET/cascade.xml')
DOL_cascade = cv2.CascadeClassifier('outputs/cascades/DOL/cascade.xml')
LAG_cascade = cv2.CascadeClassifier('outputs/cascades/LAG/cascade.xml')
OTHER_cascade = cv2.CascadeClassifier('outputs/cascades/OTHER/cascade.xml')
SHARK_cascade = cv2.CascadeClassifier('outputs/cascades/SHARK/cascade.xml')
YFT_cascade = cv2.CascadeClassifier('outputs/cascades/YFT/cascade.xml')

# fish_cascade2 = cv2.CascadeClassifier('outputs/cascades/lbp2/lbp2.xml')
# fish_cascade3 = cv2.CascadeClassifier('outputs/cascades/lbp3/lbp75x50.xml')
# fish_cascade4 = cv2.CascadeClassifier('outputs/cascades/lbp4/lbp4.xml')
# fish_cascade5 = cv2.CascadeClassifier('outputs/cascades/lbp6/lbp6.xml')

scaleF = 1.001
minN = 100
minx = 100
miny = 100
maxx = 800
maxy = 800

ALB = ALB_cascade.detectMultiScale(gray, scaleFactor=scaleF, minNeighbors=minN, minSize=(minx, miny), maxSize=(maxx, maxy))
BET = BET_cascade.detectMultiScale(gray, scaleFactor=scaleF, minNeighbors=minN, minSize=(minx, miny), maxSize=(maxx, maxy))
DOL = DOL_cascade.detectMultiScale(gray, scaleFactor=scaleF, minNeighbors=minN, minSize=(minx, miny), maxSize=(maxx, maxy))
LAG = LAG_cascade.detectMultiScale(gray, scaleFactor=scaleF, minNeighbors=minN, minSize=(minx, miny), maxSize=(maxx, maxy))
OTHER = OTHER_cascade.detectMultiScale(gray, scaleFactor=scaleF, minNeighbors=minN, minSize=(minx, miny), maxSize=(maxx, maxy))
SHARK = SHARK_cascade.detectMultiScale(gray, scaleFactor=scaleF, minNeighbors=minN, minSize=(minx, miny), maxSize=(maxx, maxy))
YFT = YFT_cascade.detectMultiScale(gray, scaleFactor=scaleF, minNeighbors=minN, minSize=(minx, miny), maxSize=(maxx, maxy))


# # white
for (x, y, w, h) in ALB:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)

# # green
for (x, y, w, h) in BET:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# # blue
for (x, y, w, h) in DOL:
    cv2.rectangle(img, (x, y), (x + w, y+h), (255, 0, 0), 2)

# # red
for (x, y, w, h) in LAG:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

# # aqua
for (x, y, w, h) in OTHER:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)

# # purple
for (x, y, w, h) in SHARK:
    cv2.rectangle(img, (x, y), (x + w, y+h), (255, 0, 255), 2)

# # # yellow
for (x, y, w, h) in YFT:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)





#
# # white
# for (x, y, w, h) in fish4:
#     cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
#
# for (x, y, w, h) in fish5:
#     cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
#
# for (x, y, w, h) in fish6:
#     cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


