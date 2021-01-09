import torch
print(torch.cuda.is_available())
import os
#print(os.listdir("../../../../data/datasets/FLIR_ADAS/FLIR_ADAS/validation/RGB/"))
print(os.listdir("./weights"))

import json
filename="../../../../data/datasets/FLIR_ADAS/FLIR_ADAS/training/Annotations/FLIR_06647.json"

with open(filename) as f:
    d = json.load(f)


filename='./annotations1/set06/V000/I00019.txt'
my_file = open(filename, "w")