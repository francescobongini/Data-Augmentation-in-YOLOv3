import json
import argparse

def convert(source_path, destination_path):
  
  with open(source_path) as json_file:
    data = json.load(json_file)
 
    print("Converting...\n")
    counter = 0

    for p in data['annotations']:
      if counter > 0 and counter % 1000 == 0:
        print("Converted: {0} images".format(counter))  
      for p in data['annotations']:
        if (p['image_id'] == counter) and p['category_id']-1<3:
          print(p['category_id']-1, end=" ", file=open(destination_path +'/FLIR_{:05d}.txt'.format(counter + 1), 'a'))
          print((p['segmentation'][0][0] + p['segmentation'][0][4])/ 2 / 640, end=" ", file=open(destination_path + '/FLIR_{:05d}.txt'.format(counter + 1), 'a'))
          print((p['segmentation'][0][1] + p['segmentation'][0][3]) / 2 / 512, end=" ", file=open(destination_path + '/FLIR_{:05d}.txt'.format(counter + 1), 'a'))
          print(p['bbox'][2] / 640, end=" ", file=open(destination_path + '/FLIR_{:05d}.txt'.format(counter + 1), 'a'))
          print(p['bbox'][3] / 512, file=open(destination_path + '/FLIR_{:05d}.txt'.format(counter + 1), 'a'))
      counter += 1
      if (counter >= int(p['image_id'])):
        print("Total: {0} images converted\n".format(counter))
        print("Conversion successfully!" )
        break

  return None    

parser = argparse.ArgumentParser()
#parser.add_argument('-src', default="../../../../data/datasets/FLIR_ADAS/FLIR_ADAS/training/Annotations/", type=str, required=True, help='path to source folder')
#parser.add_argument('-dst', default="./flir/training/", type=str, required=True, help='path to destination folder')
args = parser.parse_args()

source_path="./FLIR_JSON/thermal_annotations.json" #train valid o test
destination_path="./flir/train" #train valid o test


def main():
  convert(source_path, destination_path)

if __name__ == "__main__":
  main()
'''
###############TRAINING###########################
from pathlib import Path
import glob
for i in range(1,8862):
  my_file = "./flir/training/FLIR_{:05d}.txt".format(i)
  #print(my_file)
  #print(i)
  if Path(my_file).is_file()==False:
    print(my_file,"manca")
    open(my_file, 'w')

###############VALID###########################
from pathlib import Path
import glob
for i in range(1,1366):
  my_file = "./flir/valid/FLIR_0{:04d}.txt".format(i)
  #print(my_file)
  #print(i)
  if Path(my_file).is_file()==False:
    print(my_file,"manca")
    open(my_file, 'w')

###############TEST###########################
from pathlib import Path
import glob
for i in range(1,4224):
  my_file = "./flir/test/FLIR_0{:04d}.txt".format(i)
  #print(my_file)
  #print(i)
  if Path(my_file).is_file()==False:
    print(my_file,"manca")
    open(my_file, 'w')
    
'''
import shutil
from pathlib import Path
import glob
#make a copy of the invoice to work with
'''
for i in range(1,1366):
  src="./flir/valid/FLIR_0{:04d}.txt".format(i)
  j=i+8862
  dst = "./flir/valid/FLIR_0{:04d}.txt".format(j)
  shutil.copy(src, dst)
  print(src,dst)
'''