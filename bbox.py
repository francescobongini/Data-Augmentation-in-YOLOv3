import inspect
import glob
import os

from imgaug import augmenters as iaa
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from bbaug.policies import policies
aug_policy = policies.policies_v0()

aug_policy_container = policies.PolicyContainer(aug_policy, return_yolo=True)


class ExampleDataset:
    
    def __init__(self, root, policy_container=None):
        self.root = root
        self.policy_container = policy_container
        self.imgs = list(sorted(glob.glob(f'{root}/Data_aug/*.jpeg')))
        self.boxes = list(sorted(glob.glob(f'{root}/labels/*.txt')))
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.imgs)
        
    def __getitem__(self, idx):
        #print(idx)
        img = np.array(Image.open(self.imgs[idx]))
        boxes_path = self.boxes[idx]
        
        # For convenience I’ve hard coded the label and co-ordinates as label, x_min, y_min, x_max, y_max
        # for each bounding box in the image. For your own model you will need to load
        # in the coordinates and do the appropriate transformations.
        boxes = []
        boxes1=[]
        labels = []
        with open(boxes_path, 'r') as in_box:
            for line in in_box:
                if line:
                    line = line.split()
                    line1=list(map(float, line))
                    a=[float(line1[1]-line1[3]/2)*640,float(line1[2]-line1[4]/2)*512,float(line1[1]+line1[3]/2)*640,float(line1[2]+line1[4]/2)*512]
                    boxes.append(list(map(float, line[1:])))
                    boxes1.append(list(map(float, a)))
                    labels.append(int(line[0]))
        
        if self.policy_container and len(labels)>0:

            # Select a random sub-policy from the policy list
            random_policy = self.policy_container.select_random_policy()
            print(random_policy)

            # Apply this augmentation to the image, returns the augmented image and bounding boxes
            # The boxes must be at a pixel level. e.g. x_min, y_min, x_max, y_max with pixel values
            img_aug, bbs_aug = self.policy_container.apply_augmentation(
                random_policy,
                img,
                boxes1,
                labels,
            )
            labels = np.array(labels)
            img = self.to_tensor(img) # Convert the image to a tensor
            boxes = np.hstack((np.vstack(labels.astype("int")), np.array(boxes))) # Add the labels to the boxes
            img_aug = self.to_tensor(img_aug) # Convert the augmented image to a tensor
            bbs_aug= np.array(bbs_aug)
            
            # Only return the augmented image and bounded boxes if there are
            # boxes present after the image augmentation
            if bbs_aug.size > 0:
                return img, boxes, img_aug, bbs_aug
            else:
                return img, boxes, [], np.array([])
        return img, boxes
        
    def collate_fn(self, batch):
        """
        Custom collate function to incorporate the augmentations into the 
        input tensor
        """
        if self.policy_container:
            imgs, targets, imgs_aug, targets_aug = list(zip(*batch))


            # Create the image and target list for the unaugmented data
            imgs = [i for i in imgs]
            targets = [i for i in targets]

            # Only add the augmented images and targets if there are targets
            for i, box_aug in enumerate(targets_aug):
                if box_aug.size > 0:
                    imgs.append(imgs_aug[i])
                    targets.append(box_aug)

            # Stack the unaugmented and augmented images together
            imgs = torch.stack(imgs)
            
            # Concatenate the unaugmented and augmented targets together
            # also add the sample index to the first column
            for i in range(len(imgs)):
                targets[i] = torch.Tensor(np.insert(targets[i], 0, i, axis=1))
            targets = torch.cat(targets, 0)
            
            return imgs, targets
        
        
        
dataset = ExampleDataset('../yolov3/coco/images/FLIR_Dataset/training/', policy_container=aug_policy_container)
#img, bbs, img_aug, bbs_aug = dataset[0]



jj=0
tensor_to_image = transforms.ToPILImage()
for i in range(0,8862,1): #8862
    print(i)
    if len(dataset[i])==4:
        img, bbs, img_aug, bbs_aug = dataset[i]
        if len(img_aug)>0:
        #tensor_to_image(img).save("C:/Users/Francesco/Desktop/prova/Data/prova_%i.jpeg" %i)
            tensor_to_image(img_aug).save("../yolov3/coco/images/FLIR_Dataset/training/Data_aug/FLIR_aug_{:05d}.jpeg".format(jj+1))
            with open("../yolov3/coco/images/FLIR_Dataset/training/labels/FLIR_aug_{:05d}.txt".format(jj+1),'w') as file:
                for j in bbs_aug:
                    d=0
                    for line in j:
                        if d==0:
                            file.write(str(int(line)))
                            file.write(' ')
                        else:
                            file.write(str(line))
                            file.write(' ')
                        d=d+1
                    file.write('\n')
    else:
        img, bbs = dataset[i]
        tensor_to_image(img).save("../yolov3/coco/images/FLIR_Dataset/training/Data_aug/FLIR_aug_{:05d}.jpeg".format(jj+1))
        with open("../yolov3/coco/images/FLIR_Dataset/training/labels/FLIR_aug_{:05d}.txt".format(jj+1),'w') as file:
            for j in bbs:
                d=0
                for line in j:
                    if d==0:
                        file.write(str(int(line)))
                        file.write(' ')
                    else:
                        file.write(str(line))
                        file.write(' ')
                    d=d+1
                file.write('\n')
    jj=jj+1






'''

dataloader = DataLoader(
    dataset,
    batch_size=5,
    num_workers=0,
    collate_fn=dataset.collate_fn
)

for i in range(1,10):
    for batch in dataloader:
        images, targets = batch
        print(images.shape)
        print(targets.shape)
        tensor_to_image = transforms.ToPILImage()
        #tensor_to_image(images[0])
        tensor_to_image(images[i]).save("C:/Users/Francesco/Desktop/data/prova_%i.jpeg" %i)
print(targets) 
with open("C:/Users/Francesco/Desktop/prova/labels/labels_%i.txt" %i,'w') as file:
    for j in bbs_aug:
        d=0
        for line in j:
            if d==0:
                file.write(str(int(line)))
                file.write(' ')
            else:
                file.write(str(line))
                file.write(' ')
            d=d+1
        file.write('\n')       
'''
