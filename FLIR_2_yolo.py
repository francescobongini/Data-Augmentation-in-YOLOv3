from __future__ import print_function
import argparse
import glob
import os
import json

if 1<2:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path", help="../yolov3/coco/images/FLIR_Dataset/video/Annotations/")
    parser.add_argument(
        "output_path", help='./flir/video/')
    args = parser.parse_args()
    json_files = sorted(glob.glob(os.path.join(args.path, '*.json')))

    for json_file in json_files:
        with open(json_file) as f:
            data = json.load(f)
            images = data['image']
            annotations = data['annotation']
            print(annotations)
            file_names = []
            for i in range(0, len(images)):
                #(images)
                file_names.append(images['file_name'])
            #print(file_names)
            width = 640.0
            height = 512.0

            for i in range(0, len(images)):
                converted_results = []
                for ann in annotations:
                    if int(ann['category_id']) <= 3:
                        cat_id = int(ann['category_id'])
                        # if cat_id <= 3:
                        left, top, bbox_width, bbox_height = map(float, ann['bbox'])
                        print(ann)
                        #print(i)
                        # Yolo classes are starting from zero index
                        cat_id -= 1
                        x_center, y_center = (
                            left + bbox_width / 2, top + bbox_height / 2)

                        # darknet expects relative values wrt image width&height
                        x_rel, y_rel = (x_center / width, y_center / height)
                        w_rel, h_rel = (bbox_width / width, bbox_height / height)
                        converted_results.append(
                            (cat_id, x_rel, y_rel, w_rel, h_rel))
                image_name = images['file_name']
                #print(converted_results)
                #image_name = image_name[14:-5]
                file = open(args.output_path + str(image_name) + '.txt', 'w+')
                file.write('\n'.join('%d %.6f %.6f %.6f %.6f' % res for res in converted_results))
                file.close()