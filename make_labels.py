#1036,1552, 1778

with open("C:/Users/Francesco/Desktop/flir_train_scene2.txt", 'w') as file:
    for i in range(0,1036,2):
        file.write("../yolov3/coco/images/FLIR_Dataset/training/scene/Fabio_{:d}.jpeg".format(i))
        file.write("\n")      
    for i in range(0,1552,2):
        file.write("../yolov3/coco/images/FLIR_Dataset/training/scene/Leopold_{:d}.jpeg".format(i))
        file.write("\n")
    for i in range(0,1798,2):
        file.write("../yolov3/coco/images/FLIR_Dataset/training/scene/Stefania_{:d}.jpeg".format(i))
        file.write("\n")
    for i in range(0,1036,2):
        file.write("../yolov3/coco/images/FLIR_Dataset/training/scene/Alice01_{:d}.jpeg".format(i))
        file.write("\n")
    for i in range(0,1552,2):
        file.write("../yolov3/coco/images/FLIR_Dataset/training/scene/Alice02_{:d}.jpeg".format(i))
        file.write("\n")
    for i in range(0,1798,2):
        file.write("../yolov3/coco/images/FLIR_Dataset/training/scene/Alice03_{:d}.jpeg".format(i))
        file.write("\n")        
        
        
        
        
        
        
        
#################################################################################





from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
ffmpeg_extract_subclip("C:/Users/Francesco/Desktop/risultati tesi/predicted.mp4", 158, 170, targetname="C:/Users/Francesco/Desktop/risultati tesi/test.mp4")






import imageio
import os, sys

class TargetFormat(object):
    GIF = ".gif"
    MP4 = ".mp4"
    AVI = ".avi"

def convertFile(inputpath, targetFormat):
    """Reference: http://imageio.readthedocs.io/en/latest/examples.html#convert-a-movie"""
    outputpath = os.path.splitext(inputpath)[0] + targetFormat
    print("converting\r\n\t{0}\r\nto\r\n\t{1}".format(inputpath, outputpath))

    reader = imageio.get_reader(inputpath)
    fps = reader.get_meta_data()['fps']

    writer = imageio.get_writer(outputpath, fps=fps)
    for i,im in enumerate(reader):
        sys.stdout.write("\rframe {0}".format(i))
        sys.stdout.flush()
        writer.append_data(im)
    print("\r\nFinalizing...")
    writer.close()
    print("Done.")

convertFile("C:/Users/Francesco/Desktop/risultati tesi/test.mp4", TargetFormat.GIF)



from moviepy.editor import *
clip = (VideoFileClip("C:/Users/Francesco/Desktop/risultati tesi/test.mp4"))
clip.write_gif("C:/Users/Francesco/Desktop/risultati tesi/test.gif")









with open("C:/Users/Francesco/Desktop/flir_train_3cars.txt", 'w') as file:
    for i in range(0,240):
        file.write("../yolov3/coco/images/FLIR_Dataset/training/scene/3Cars3People_{:d}.jpeg".format(i))
        file.write("\n") 







with open("C:/Users/Francesco/Desktop/ds.txt", 'w') as file:
    for i in range(0,600):
        file.write("../yolov3/coco/images/FLIR_Dataset/training/scene/3Cars6People_{:d}.jpeg".format(i))
        file.write("\n")  









