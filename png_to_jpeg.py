'''
from PIL import Image
#image = Image.open('C:/Users/Francesco/Desktop/Scena03_Stefania_1799_fake_B.png')
#image=image.resize((640, 512), Image.ANTIALIAS)
#image.save('C:/Users/Francesco/Desktop/Scena03_Stefania_1799_fake_B.jpeg', 'JPEG')

from os import listdir
from os.path import isfile, join
mypath="../dataset_sequenze/"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

for i in onlyfiles:
    #print(mypath+i)
    image = Image.open(mypath+i)
    image = image.resize((640, 512), Image.ANTIALIAS)
    image1=mypath.replace("/dataset_sequenze/", "/dataset_sequenze1/")+i.replace(".png", ".jpeg")
    print(image1)
    image.save(image1, 'JPEG')

'''

#from PIL import Image
#from os import listdir
#from os.path import isfile, join
#with open("./dataset_sequenze.txt", 'a') as file:
 #   mypath = "../dataset_sequenze/"
  #  onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
   # for i in onlyfiles:
    #    file.write("../yolov3/coco/images/FLIR_Dataset/training/dataset_sequenze/"+i.replace(".png", ".jpeg"))
     #   file.write("\n")












