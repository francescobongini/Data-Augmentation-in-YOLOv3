from pathlib import Path
for i in range(8842):
    my_file = Path("../yolov3/coco/images/FLIR_Dataset/training/Data_aug/FLIR_aug_{:05d}.jpeg".format(i+1))
    #print(my_file)
    if my_file.is_file():
        #print(my_file)
        a=0
    else:
        print(i+1, " manca")