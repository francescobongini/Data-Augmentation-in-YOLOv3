# Data Augmentation strategies for Object Detection in the thermal spectrum

 
This repository is forked from great work pytorch-yolov3 of <a href="https://github.com/andy-yun/pytorch-0.4-yolov3">@github/andy-yun </a> 
. However, this repository is changed many files and functions for our research.

### How to run this repository
1. Download or clone this repository to your computer.
2. Install some basic requirements if needed.
3. Download <a href="https://drive.google.com/file/d/1xx4nhja95VeFsZydTycD8ArTYl1p-bnx/view?usp=sharing">flir_detector.weights </a> files and put in the directory 'weights'.
4. Open a terminal and run following commands according to functions:

Noted that all of these instructions for Linux environment (for Windows, you should visit original repository to read more)

### Some default parameters:
* weightfile = weights/flir_thermal_detector.weights 
* configurationfile = cfg/yolov3_flir.cfg 
* datafile = data/flir.data
* listname = data/flir.names
For all of following commands, if command with [...] will be an option,
you can use your parameter or leave there to use default paramaters above.

### Data set and labels
Download the FLIR data set from https://www.flir.com/oem/adas/adas-dataset-form/ .
All the labels used in the experiments are available on https://drive.google.com/drive/u/1/folders/1womuSXrb8uWXYHeGX5eOdTTIEozTnL8M .

### Train the model:
Train the model without augmentation
```
python3 train.py
```

### Train the model with Rand augmentation:
```
python3 train_randaug.py
```
### Generate data with Rand augmentation:
```
python3 randaugment.py
```

### Detection (detect bounding box):
Detect bounding box result on image(s) or video by parameter: 
image file or folder with all images or video file. 
The result will appear with the same name + 'predicted'
```
python detect.py image/video/folder
Example:
python detect.py thermal_kaist.png
```


```
python map.py weightfile
```

### Mean Average Precision (mAP):
```
python3 map.py [weightfile]
```

### Draw bounding box:
Given the folder of images with its annotation.
Drawing bounding box on every image with correct detection (blue boxes),
wrong detection (red boxes) and miss detection (green boxes)

```
python drawBBxs.py imagefolder
```



### Example results:
![Image detection](screenshot/FLIR_08938_predicted.png)

[![Video detection](screenshot/test.gif)]

### Initial Results:

FLIR dataset results (precision):
* person:    	75.6%
* bicycle:   	57.4%
* car:         	86.5%

mean Average Precision:  	73.2%

### Data augmentation strategies results:

Boundary Box Augmentation:
* person:    	79.4%
* bicycle:   	58.4%
* car:         	87.2%

mean Average Precision:  	75.2%

RandAugment:
* person:    	74.4%
* bicycle:   	60.2%
* car:         	85.3%

mean Average Precision:  	73.3%

RandAugment + BBox:
* person:    	74.6%
* bicycle:   	61.1%
* car:         	86.0%

mean Average Precision:  	73.9%

Scenes:
* person:    	79.3%
* bicycle:   	60.1%
* car:         	86.2%

mean Average Precision:  	75.2%

Scenes + BBox:
* person:    	77.1%
* bicycle:   	64.2%
* car:         	85.6%

mean Average Precision:  	75.6%


If you use our Layer-wise method, please cite also the paper
```
@inproceedings{kieu2020layerwise,
	Author = {Kieu, My and Bagdanov, Andrew D and Bertini, Marco},
	Booktitle = {ACM Transactions on Multimedia Computing Communications and Applications (ACM TOMM)},
	Title = {Bottom-up and Layer-wise Domain Adaptation for Pedestrian Detection in Thermal Images},
	Year = {2020}
	}
```

If you have any comment or question to contribute, please leave it in Issues.

Other question, please contact me by email: francesco.bongini@stud.unifi.it.

Thank you.
