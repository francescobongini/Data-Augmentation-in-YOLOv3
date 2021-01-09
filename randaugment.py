import numpy as np
from PIL import Image
from django.utils.translation import gettext as _
from torchvision.transforms import transforms
#from RandAugment import RandAugment, augmentations
#from RandAugment import *
#from augmentations import *
import matplotlib.pyplot as plt
import inspect
import glob
from random import randrange
import os
from torchvision import transforms

# code in this file is adpated from rpmcruz/autoaugment
# https://github.com/rpmcruz/autoaugment/blob/master/transformations.py
import random

import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw
import numpy as np
import torch
from PIL import Image


def ShearX(img, v):  # [-0.3, 0.3]
    v=v*0.3
    assert -0.3 <= v <= 0.3
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))


def ShearY(img, v):  # [-0.3, 0.3]
    v=v*0.3
    assert -0.3 <= v <= 0.3
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))


def TranslateX(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    v=v*0.45
    assert -0.45 <= v <= 0.45
    #if random.random() > 0.5:
     #   v = -v
    v = v * img.size[0]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))

def TranslateX_lab(box,v):
    v = v * 0.45
    for line in box: #transform to x1,y1,x2,y2 and then calculate the transformated box
        #line1=[float(line[1]-line[3]/2)*640,float(line[2]-line[4]/2)*512,float(line[1]+line[3]/2)*640,float(line[2]+line[4]/2)*512]
         line=[float(line[1])-float(line[1])*v,float(line[2]),float(line[3]),float(line[4])]       
    return box

def TranslateY(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    v = v * 0.45
    assert -0.45 <= v <= 0.45
    if random.random() > 0.5:
        v = -v
    v = v * img.size[1]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))

def TranslateY_lab(box,v):
    v = v * 0.45
    for line in box: #transform to x1,y1,x2,y2 and then calculate the transformated box
        #line1=[float(line[1]-line[3]/2)*640,float(line[2]-line[4]/2)*512,float(line[1]+line[3]/2)*640,float(line[2]+line[4]/2)*512]
         line=[float(line[1]),float(line[2])-float(line[2])*v,float(line[3]),float(line[4])]       
    return box

def TranslateXabs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))

def TranslateYabs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def Rotate(img, v):  # [-30, 30]
    v = v * 30
    assert -30 <= v <= 30
    #if random.random() > 0.5:
      #  v = -v
    return img.rotate(v)

def Rotate_lab(box, v):
    v = v * 30
    #for line in box: #transform to x1,y1,x2,y2 and then calculate the transformated box
    #line1=[float(line[1]-line[3]/2)*640,float(line[2]-line[4]/2)*512,float(line[1]+line[3]/2)*640,float(line[2]+line[4]/2)*512]
        #line=[float(1-line[1]),float(line[2]),float(line[3]),float(line[4])]
    return box

def AutoContrast(img, v):
    return PIL.ImageOps.autocontrast(img)

def AutoContrast_lab(box):
    return box

def Invert(img, _):
    return PIL.ImageOps.invert(img)

def Invert_lab(box):
    return box


def Equalize(img):
    return PIL.ImageOps.equalize(img)

def Equalize_lab(box,v):
    return box


def Flip(img, v):  # not from the paper
    return PIL.ImageOps.mirror(img)

def Flip_lab(box):  # not from the paper
    for line in box: #transform to x1,y1,x2,y2 and then calculate the transformated box
    #line1=[float(line[1]-line[3]/2)*640,float(line[2]-line[4]/2)*512,float(line[1]+line[3]/2)*640,float(line[2]+line[4]/2)*512]
        line=[float(1-line[1]),float(line[2]),float(line[3]),float(line[4])]
    return box


def Solarize(img, v):  # [0, 256]
    assert 0 <= v <= 256
    return PIL.ImageOps.solarize(img, v)
def Solarize_lab(box, v):  # [0, 256]
    assert 0 <= v <= 256
    return box


def SolarizeAdd(img, addition=0, threshold=128):
    img_np = np.array(img).astype(np.int)
    img_np = img_np + addition
    img_np = np.clip(img_np, 0, 255)
    img_np = img_np.astype(np.uint8)
    img = Image.fromarray(img_np)
    return PIL.ImageOps.solarize(img, threshold)


def Posterize(img, v):  # [4, 8]
    v = int(v)
    v = max(1, v)
    return PIL.ImageOps.posterize(img, v)

def Posterize_lab(box,v):
    return box


def Contrast(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Contrast(img).enhance(v)

def Contrast_lab(box,v):
    return box

def Color(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Color(img).enhance(v)

def Color_lab(box,v):
    return box

def Brightness(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Brightness(img).enhance(v)

def Brightness_lab(box, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return box


def Sharpness(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Sharpness(img).enhance(v)

def Sharpness_lab(box,v):
    return box


def Cutout(img, v):  # [0, 60] => percentage: [0, 0.2]
    assert 0.0 <= v <= 0.2
    if v <= 0.:
        return img

    v = v * img.size[0]
    return CutoutAbs(img, v)


def CutoutAbs(img, v):  # [0, 60] => percentage: [0, 0.2]
    # assert 0 <= v <= 20
    if v < 0:
        return img
    w, h = img.size
    x0 = np.random.uniform(w)
    y0 = np.random.uniform(h)

    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = min(w, x0 + v)
    y1 = min(h, y0 + v)

    xy = (x0, y0, x1, y1)
    color = (125, 123, 114)
    # color = (0, 0, 0)
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img


def SamplePairing(imgs):  # [0, 0.4]
    def f(img1, v):
        i = np.random.choice(len(imgs))
        img2 = PIL.Image.fromarray(imgs[i])
        return PIL.Image.blend(img1, img2, v)

    return f


def Identity(img, _):
    return img

def Identity_lab(box):
    return box





def augment_list():  # 16 oeprations and their ranges
    # https://github.com/google-research/uda/blob/master/image/randaugment/policies.py#L57
    # l = [
    #     (Identity, 0., 1.0),
    #     (ShearX, 0., 0.3),  # 0
    #     (ShearY, 0., 0.3),  # 1
    #     (TranslateX, 0., 0.33),  # 2
    #     (TranslateY, 0., 0.33),  # 3
    #     (Rotate, 0, 30),  # 4
    #     (AutoContrast, 0, 1),  # 5
    #     (Invert, 0, 1),  # 6
    #     (Equalize, 0, 1),  # 7
    #     (Solarize, 0, 110),  # 8
    #     (Posterize, 4, 8),  # 9
    #     # (Contrast, 0.1, 1.9),  # 10
    #     (Color, 0.1, 1.9),  # 11
    #     (Brightness, 0.1, 1.9),  # 12
    #     (Sharpness, 0.1, 1.9),  # 13
    #     # (Cutout, 0, 0.2),  # 14
    #     # (SamplePairing(imgs), 0, 0.4),  # 15
    # ]

    # https://github.com/tensorflow/tpu/blob/8462d083dd89489a79e3200bcc8d4063bf362186/models/official/efficientnet/autoaugment.py#L505
    l = [
        (AutoContrast, 0, 1),
        (Equalize, 0, 1),
        (Invert, 0, 1),
        (Rotate, 0, 30),
        (Posterize, 0, 4),
        (Solarize, 0, 256),
        (SolarizeAdd, 0, 110),
        (Color, 0.1, 1.9),
        (Contrast, 0.1, 1.9),
        (Brightness, 0.1, 1.9),
        (Sharpness, 0.1, 1.9),
        (ShearX, 0., 0.3),
        (ShearY, 0., 0.3),
        (CutoutAbs, 0, 40),
        (TranslateXabs, 0., 100),
        (TranslateYabs, 0., 100),
    ]

    return l


class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = torch.Tensor(eigval)
        self.eigvec = torch.Tensor(eigvec)

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone() \
            .mul(alpha.view(1, 3).expand(3, 3)) \
            .mul(self.eigval.view(1, 3).expand(3, 3)) \
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))


class CutoutDefault(object):
    """
    Reference : https://github.com/quark0/darts/blob/master/cnn/utils.py
    """
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


class RandAugment:
    def __init__(self, n, m):
        self.n = n
        self.m = m      # [0, 30]
        self.augment_list = augment_list()

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)
        for op, minval, maxval in ops:
            val = (float(self.m) / 30) * float(maxval - minval) + minval
            img = op(img, val)

        return img


###############################################################################################

tensor_to_image = transforms.ToPILImage()
transformations = [
"Identity", "AutoContrast", "Equalize",
"Rotate", "Solarize", "Color", "Posterize",
"Contrast", "Brightness", "Sharpness",
"ShearX", "ShearY", "TranslateX", "TranslateY"]
def randaugment(N, M):
#Generate a set of distortions.
#Args:
#N: Number of augmentation transformations to
#apply sequentially.
#M: Magnitude for all the transformations.

    sampled_ops = np.random.choice(transformations, N)
    return [(op, M) for op in sampled_ops]

randaugment(4, 2)


a=Image.open("../yolov3/coco/images/FLIR_Dataset/training/Data/FLIR_00001.jpeg")

#Identity(a,1) #ritorna l'immagine identica, il parametro v non serve a niente
#AutoContrast(a,_)
#Equalize(a,_)
#Rotate(a,3)
#Solarize(a,255)
#Color(a,1.8)
#Posterize(a,5)
#Contrast(a,1.8)
#Brightness(a,1.9)
#Sharpness(a,0.8)
#ShearX(a,0.3)
#ShearY(a,-0.1)
#TranslateX(a,0.4)
#TranslateY(a,-0.3)



imgs = list(sorted(glob.glob(f'../yolov3/coco/images/FLIR_Dataset/training/Data/*.jpeg')))
boxes = list(sorted(glob.glob(f'../yolov3/coco/images/FLIR_Dataset/training/labels/*.txt')))


def __getitem__(boxes,idx):
    img = np.array(Image.open(imgs[idx]))
    boxes_path = boxes[idx]
    to_tensor = transforms.ToTensor()

    boxes = []
    if 1>0:
    # For convenience I’ve hard coded the label and co-ordinates as label, x_min, y_min, x_max, y_max
    # for each bounding box in the image. For your own model you will need to load
    # in the coordinates and do the appropriate transformations.
        boxes = []
        boxes1=[]
        labels = []
        print(boxes_path)
        with open(boxes_path, 'r') as in_box:
            for line in in_box:
                if line:
                    line = line.split()
                    line1=list(map(float, line))
                    a=[float(line1[1]-line1[3]/2)*640,float(line1[2]-line1[4]/2)*512,float(line1[1]+line1[3]/2)*640,float(line1[2]+line1[4]/2)*512]
                    boxes.append(list(map(float, line[1:])))
                    boxes1.append(list(map(float, a)))
                    labels.append(int(line[0]))
        labels = np.array(labels)
        img = to_tensor(img) # Convert the image to a tensor
        print("labels",labels, boxes)
        if len(labels)>0:
            boxes = np.hstack((np.vstack(labels.astype("int")), np.array(boxes)))

    return img, boxes


img,box=__getitem__(boxes,3)

i=0
#tensor_to_image(img).save("C:/Users/Francesco/Desktop/prova/Data/prova_aug_0{:05d}.jpeg".format(i+1))


img=tensor_to_image(img)

def randaug(img,box):
    v=1 #value from 0 to 1
    transformations=randaugment(2, 2)
    img_aug=img
    print("type",type(img))
    box_aug=box
    print(transformations)
    for transformation in transformations:
        if transformation[0]=="Identity":
            #img_aug, box_aug=Identity(img,_)
            cd=0
        elif transformation[0]=="Autocontrast":
            img_aug, box_aug=AutoContrast(img_aug,1)
        elif transformation[0]=="Equalize":
            cd=0
            #img_aug, box_aug=Equalize(img)
        elif transformation[0]=="Rotate":
            img_aug=Rotate(img_aug,v)
            box_aug=Rotate_lab(box, v)
        elif transformation[0]=="Color":
            img_aug=Color(img_aug,v)
        elif transformation[0]=="Posterize":
            img_aug=Posterize(img_aug,v)
            box_aug = Posterize_lab(box_aug, v)
        elif transformation[0]=="Contrast":
            img_aug=Contrast(img_aug,v)
            box_aug = Contrast_lab(box_aug, v)
        elif transformation[0]=="Brightness":
            img_aug=Brightness(img_aug,v)
            box_aug = Brightness_lab(box_aug, v)
        elif transformation[0]=="Sharpness":
            img_aug=Sharpness(img_aug,v)
            box_aug = Sharpness_lab(box_aug, v)
        elif transformation[0]=="ShearX":
            img_aug=ShearX(img_aug,v)
            #box_aug = ShearX_lab(box_aug, v)
        elif transformation[0]=="ShearY":
            img_aug=ShearY(img_aug,v)
            # box_aug = ShearY_lab(box_aug, v)
        elif transformation[0]=="TranslateX":
            img_aug=TranslateX(img_aug,v)
            box_aug = TranslateX_lab(box_aug, v)
        elif transformation[0]=="TranslateY":
            img_aug=TranslateY(img_aug,v)
            box_aug = TranslateY_lab(box_aug, v)
    return img_aug,box_aug

img_aug,box_aug=randaug(img,box)
#i=0
#img_aug.save("../yolov3/coco/images/FLIR_Dataset/training/Data_randaug/randaug_0{:05d}.jpeg".format(i+1))
#box_aug.write("./prova_aug_0{:05d}.txt".format(i+1))
#with open("../yolov3/coco/images/FLIR_Dataset/training/Data_randaug/randaug_0{:05d}.txt".format(i + 1), 'w') as file:
#    for j in box_aug:
 #       d = 0
 #       for line in j:
  #          if d == 0:
  #              file.write(str(int(line)))
  #              file.write(' ')
  #          else:
  #              file.write(str(line))
  ##              file.write(' ')
  #          d = d + 1
  #      file.write('\n')
#i=0
#with open("../yolov3/coco/images/FLIR_Dataset/training/labels/FLIR_00000i.txt" %i,'w') as file:
#    for j in box_aug:
#        d=0
#        for line in j:
#            if d==0:
#                file.write(str(int(line)))
#                file.write(' ')
#            else:
#                file.write(str(line))
#                file.write(' ')
#            d=d+1
#        file.write('\n')
#img_aug.save("C:/Users/Francesco/Desktop/prova/Data/prova_aug_0{:05d}.jpeg".format(i+1))

#Identity(a,1) #ritorna l'immagine identica, il parametro v non serve a niente v
#AutoContrast(a,_) v
#Equalize(a,_) v
#Rotate(a,3) v
#Solarize(a,255) v
#Color(a,1.8) v
#Posterize(a,5) v
#Contrast(a,1.8) v
#Brightness(a,1.9) v
#Sharpness(a,0.8) v
#ShearX(a,0.3)
#ShearY(a,-0.1)
#TranslateX(a,0.4) v
#TranslateY(a,-0.3) v
#Flip(a,_) v


TranslateX(a,-0.3)
# For each epoch the function takes some random images and makes the augmentation, putting them in Data_randaug folder, calling them FLIR_000001, .. , FLIR_002000
def make_rand_augmentation(n):
    imgs = list(sorted(glob.glob(f'../yolov3/coco/images/FLIR_Dataset/training/Data/*.jpeg')))
    boxes = list(sorted(glob.glob(f'../yolov3/coco/images/FLIR_Dataset/training/labels/*.txt')))
    for i in range(n):
        rdm=randrange(8862)
        img,box=__getitem__(boxes,rdm)
        img = tensor_to_image(img)
        img_aug,box_aug=randaug(img,box)
        img_aug.save("../yolov3/coco/images/FLIR_Dataset/training/Data_randaug/randaug_0{:05d}.jpeg".format(i+1))
        #box_aug.write("./prova_aug_0{:05d}.txt".format(i+1))
        with open("../yolov3/coco/images/FLIR_Dataset/training/labels/randaug_0{:05d}.txt".format(i + 1), 'w') as file:
            for j in box_aug:
                d = 0
                for line in j:
                    if d == 0:
                        file.write(str(int(line)))
                        file.write(' ')
                    else:
                        file.write(str(line))
                        file.write(' ')
                    d = d + 1
                file.write('\n')




