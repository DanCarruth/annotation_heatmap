# Author: Daniel Carruth (with assistance from ChatGPT4)
# Contact: dwc2@cavs.msstate.edu
# Date: April 9, 2023
# Description: This file contains functions for processing annotated images.
# License: GPLv3
import os
import numpy as np
import cv2
from glob import glob
from collections import defaultdict
from PIL import Image
import pandas as pd

def generate_probability_map(dataset, dataset_path, annotation_path, class_name, class_rgb):
    probability_map = None
    if dataset == 'CityScapes gtFine':
        annotation_files = glob(os.path.join(annotation_path, '*/*/*_color.png'))
    elif dataset == 'Rellis3D': 
        annotation_files = glob(os.path.join(annotation_path, '*/pylon_camera_node_label_color/*.png'))
    else:
        annotation_files = glob(os.path.join(annotation_path, '*.png' if dataset != 'CaSSed MAVS' else '*.bmp'))
    
    num_images = len(annotation_files)

    for annotation_file in annotation_files:
        annotation = cv2.imread(annotation_file)
        mask = np.all(annotation == class_rgb, axis=-1)

        if probability_map is None:
            probability_map = np.zeros(mask.shape, dtype=np.float32)

        probability_map += mask.astype(np.float32)

    # Calculate probability by dividing by the number of images
    probability_map /= num_images

    return probability_map


datasets = {
    'CityScapes gtFine': {
        'dataset_path': '/cavs/projects/ARC/Project1.38/datasets/cityscapes/gtFine_trainvaltest/gtFine/',
        'annotation_path': '/cavs/projects/ARC/Project1.38/datasets/cityscapes/gtFine_trainvaltest/gtFine/',
        'classes': [
            {'class_name': 'road', 'class_rgb': (128, 64,128)},
            {'class_name': 'sidewalk', 'class_rgb': (244, 35,232)},
            {'class_name': 'parking', 'class_rgb': (250,170,160)},
            {'class_name': 'rail track', 'class_rgb': (230,150,140)},
            {'class_name': 'building', 'class_rgb': ( 70, 70, 70)},
            {'class_name': 'wall', 'class_rgb': (102,102,156)},
            {'class_name': 'fence', 'class_rgb': (190,153,153)},
            {'class_name': 'guard rail', 'class_rgb': (180,165,180)},
            {'class_name': 'bridge', 'class_rgb': (150,100,100)},
            {'class_name': 'tunnel', 'class_rgb': (150,120, 90)},
            {'class_name': 'pole', 'class_rgb': (153,153,153)},
            {'class_name': 'traffic light', 'class_rgb': (250,170, 30)},
            {'class_name': 'traffic sign', 'class_rgb': (220,220,  0) },
            {'class_name': 'vegetation', 'class_rgb': (107,142, 35)},
            {'class_name': 'terrain', 'class_rgb': (152,251,152)},
            {'class_name': 'sky', 'class_rgb': ( 70,130,180)},
            {'class_name': 'person', 'class_rgb': (220, 20, 60)},
            {'class_name': 'rider', 'class_rgb':  (255,  0,  0)},
            {'class_name': 'car', 'class_rgb': (  0,  0,142)},
            {'class_name': 'truck', 'class_rgb': (  0,  0, 70)},
            {'class_name': 'bus', 'class_rgb': (  0, 60,100)},
            {'class_name': 'caravan', 'class_rgb': (  0,  0, 90)},
            {'class_name': 'trailer', 'class_rgb': (  0,  0,110) },
            {'class_name': 'train', 'class_rgb': (  0, 80,100)},
            {'class_name': 'motorcycle', 'class_rgb': (  0,  0,230) },
            {'class_name': 'bicycle', 'class_rgb': (119, 11, 32)},
        ],
    },
    'Rellis3D': {
        'dataset_path': '/cavs/projects/ARC/Project1.38/datasets/rellis3d/Rellis_3D_pylon_camera_node/Rellis-3D/',
        'annotation_path': '/cavs/projects/ARC/Project1.38/datasets/rellis3d/Rellis_3D_pylon_camera_node_label_color/Rellis-3D/',
        'classes': [
            {'class_name': 'dirt', 'class_rgb': (108,64,20)},
            {'class_name': 'grass', 'class_rgb': (0,102,0)},
            {'class_name': 'tree', 'class_rgb': (0,255,0)},
            {'class_name': 'pole', 'class_rgb': (0,153,153)},
            {'class_name': 'water', 'class_rgb': (0,128,255)},
            {'class_name': 'sky', 'class_rgb': (0,0,255)},
            {'class_name': 'vehicle', 'class_rgb': (255,255,0)},
            {'class_name': 'object', 'class_rgb': (255,0,127)},
            {'class_name': 'asphalt', 'class_rgb': (64,64,64)},
            {'class_name': 'building', 'class_rgb': (255,0,0)},
            {'class_name': 'log', 'class_rgb': (102,0,0)},
            {'class_name': 'person', 'class_rgb': (204,153,255)},
            {'class_name': 'fence', 'class_rgb': (102,0,204) },
            {'class_name': 'bush', 'class_rgb': (255,153,204)},
            {'class_name': 'concrete', 'class_rgb': (170,170,170)},
            {'class_name': 'barrier', 'class_rgb': (41,121,255)},
            {'class_name': 'uphill', 'class_rgb': (101,31,255)},
            {'class_name': 'downhill', 'class_rgb':  (137,149,9)},
            {'class_name': 'puddle', 'class_rgb': (134,255,239)},
            {'class_name': 'mud', 'class_rgb': (99,66,34)},
            {'class_name': 'rubble', 'class_rgb': (110,22,138)},
        ],
    },
    'RUGD': {
        'dataset_path': '/cavs/projects/ARC/Project1.38/datasets/rugd/RUGD_annotations/',
        'annotation_path': '/cavs/projects/ARC/Project1.38/datasets/rugd/RUGD_frames-with-annotations/',
        'classes': [
            {'class_name': 'dirt', 'class_rgb': (108,64,20)},
            {'class_name': 'sand', 'class_rgb': (255,229,204)},
            {'class_name': 'grass', 'class_rgb': (0,102,0)},
            {'class_name': 'tree', 'class_rgb': (0,255,0)},
            {'class_name': 'pole', 'class_rgb': (0,153,153)},
            {'class_name': 'water', 'class_rgb': (0,128,255)},
            {'class_name': 'sky', 'class_rgb': (0,0,255)},
            {'class_name': 'vehicle', 'class_rgb': (255,255,0)},
            {'class_name': 'object', 'class_rgb': (255,0,127)},
            {'class_name': 'asphalt', 'class_rgb': (64,64,64)},
            {'class_name': 'gravel', 'class_rgb': (255,128,0)},
            {'class_name': 'building', 'class_rgb': (255,0,0)},
            {'class_name': 'mulch', 'class_rgb': (153,76,0)},
            {'class_name': 'rock-bed', 'class_rgb': (102,102,0)},
            {'class_name': 'log', 'class_rgb': (102,0,0)},
            {'class_name': 'bicycle', 'class_rgb': (0,255,128)},
            {'class_name': 'person', 'class_rgb': (204,153,255)},
            {'class_name': 'fence', 'class_rgb': (102,0,204) },
            {'class_name': 'bush', 'class_rgb': (255,153,204)},
            {'class_name': 'sign', 'class_rgb': (0,102,102)},
            {'class_name': 'rock', 'class_rgb': (153,204,255)},
            {'class_name': 'bridge', 'class_rgb': (102,255,255)},
            {'class_name': 'concrete', 'class_rgb': (101,101,11)},
            {'class_name': 'picnic-table', 'class_rgb': (114,85,47)},
        ],
    },
    'CaSSed': {
        'dataset_path': '/cavs/projects/Halo/Autonomous-P/Halo_Offroad_Dataset/8_Ready/CaSSed_Dataset_Final/real_world_data/Train/imgs/',
        'annotation_path': '/cavs/projects/Halo/Autonomous-P/Halo_Offroad_Dataset/8_Ready/CaSSed_Dataset_Final/real_world_data/Train/annos/',
        'classes': [
            {'class_name': 'smooth-trail', 'class_rgb': (155,155,155)},
            {'class_name': 'rough-trail', 'class_rgb': (139,87,42)},
            {'class_name': 'vegetation', 'class_rgb': (209,255,158)},
            {'class_name': 'forest', 'class_rgb': (59,93,4)},
            {'class_name': 'sky', 'class_rgb': (74,144,226)},
            {'class_name': 'obstacle', 'class_rgb': (184,20,124)},
        ],
    },
    'CaSSed MAVS': {
        'dataset_path': '/cavs/projects/Halo/Autonomous-P/Halo_Offroad_Dataset/8_Ready/CaSSed_Dataset_Final/MAVS_Simulated_Data/Train/imgs/',
        'annotation_path': '/cavs/projects/Halo/Autonomous-P/Halo_Offroad_Dataset/8_Ready/CaSSed_Dataset_Final/MAVS_Simulated_Data/Train/annos/',
        'classes': [
            {'class_name': 'road', 'class_rgb': (255,192,0)},
            {'class_name': 'vegetation', 'class_rgb': (56,86,34)},
            {'class_name': 'forest', 'class_rgb': (145,208,80)},
            {'class_name': 'sky', 'class_rgb': (134,206,235)},
        ],
    },
}

probability_maps = defaultdict(dict)

for dataset_name, dataset_info in datasets.items():
    for class_info in dataset_info['classes']:
        print(f"Generating probability map for {dataset_name}, class {class_info['class_name']}...")
        probability_map = generate_probability_map(
            dataset=dataset_name,
            dataset_path=dataset_info['dataset_path'],
            annotation_path=dataset_info['annotation_path'],
            class_name=class_info['class_name'],
            class_rgb=class_info['class_rgb']
        )
        probability_maps[dataset_name][class_info['class_name']] = probability_map

        # Save probability map to an image file
        probability_map_img = Image.fromarray((probability_map * 255).astype(np.uint8), mode='L')
        probability_map_img.save(f"{dataset_name}_{class_info['class_name']}_probability_map.png")

        # Save probability map to a CSV file
        pd.DataFrame(probability_map).to_csv(f"{dataset_name}_{class_info['class_name']}_map.csv", index=False)

        print(f"Probability map for {dataset_name}, class {class_info['class_name']} saved.")
        break
