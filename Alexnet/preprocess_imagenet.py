#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json

image_dir = '/home/ubuntu/weight_pruning_admm/data/Imagenet/image/val'
folder_dir_list = os.listdir(image_dir)

for folder_dir in folder_dir_list:
    os.rename(image_dir + '/' + folder_dir, image_dir + '/' + folder_dir + '_')
    
folder_dir_list = os.listdir(image_dir)
label_to_content_dir = '/home/ubuntu/weight_pruning_admm/data/Imagenet/label/label_to_content.json'
imagenet_class_index_dir = '/home/ubuntu/weight_pruning_admm/data/Imagenet/label/imagenet_class_index.json' 

with open(label_to_content_dir, 'r') as f:
    label_to_content = json.load(f)

with open(imagenet_class_index_dir, 'r') as f:
    imagenet_class_index = json.load(f)

for folder_dir in folder_dir_list:
    try:
        label_name = label_to_content[folder_dir[0:-1]]
        for i, item in enumerate(imagenet_class_index.items()):
            key, values = item
            name = values[1]
            if name == label_name:
                os.rename(image_dir + '/' + folder_dir, image_dir + '/' + key)
    except:
        continue

name_list = []
for name in label_to_content.values():
    name_list.append(name)


sorted(name_list)[:20]
