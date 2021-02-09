#!/usr/bin/env python
from os import rename, listdir

badprefix = "_PreviewData"
fnames = listdir('.')

for fname in fnames:
    rename(fname, fname.replace(badprefix, '', 1))

############


import os, random
import shutil

for i in range(1000):
    item=random.choice(os.listdir("../yolov3/coco/images/FLIR_Dataset/training/flir_gan"))
    os.replace("../yolov3/coco/images/FLIR_Dataset/training/flir_gan/"+item,
               "../yolov3/coco/images/FLIR_Dataset/training/Data_with_gan/"+item)
