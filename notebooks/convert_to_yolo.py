import os
import xml.etree.ElementTree as ET
import random
import shutil

# Paths based on data/raw/ structure
raw_images_dir = 'data/raw/images'
raw_annotations_dir = 'data/raw/annotations'
processed_dir = 'data/processed'
images_subdirs = ['images/train', 'images/val', 'images/test']
labels_subdirs = ['labels/train', 'labels/val', 'labels/test']

# Create output folders
for subdir in images_subdirs + labels_subdirs:
    os.makedirs(os.path.join(processed_dir, subdir), exist_ok=True)

# Class mapping (person=0, helmet=1; ignore 'head' for now)
class_map = {'person': 0, 'helmet': 1}

# Get list of images (.png in this dataset)
image_files = [f for f in os.listdir(raw_images_dir) if f.endswith('.png')]

# Shuffle and split: 70% train, 20% val, 10% test
random.shuffle(image_files)
total = len(image_files)
train_end = int(0.7 * total)
val_end = int(0.9 * total)
splits = {
    'train': image_files[:train_end],
    'val': image_files[train_end:val_end],
    'test': image_files[val_end:]
}

# Convert each image/annotation
for split, files in splits.items():
    for img_file in files:
        # Copy image
        shutil.copy(os.path.join(raw_images_dir, img_file), os.path.join(processed_dir, f'images/{split}/{img_file}'))
        
        # Parse XML
        xml_file = img_file.replace('.png', '.xml')
        xml_path = os.path.join(raw_annotations_dir, xml_file)
        if not os.path.exists(xml_path):
            continue
        
        tree = ET.parse(xml_path)
        root = tree.getroot()
        width = int(root.find('size/width').text)
        height = int(root.find('size/height').text)
        
        # Create YOLO .txt label
        txt_file = img_file.replace('.png', '.txt')
        txt_path = os.path.join(processed_dir, f'labels/{split}/{txt_file}')
        with open(txt_path, 'w') as txt:
            for obj in root.iter('object'):
                cls_name = obj.find('name').text
                if cls_name not in class_map:
                    continue
                cls_id = class_map[cls_name]
                
                bndbox = obj.find('bndbox')
                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)
                
                # Normalize to 0-1
                bb_width = (xmax - xmin) / width
                bb_height = (ymax - ymin) / height
                center_x = (xmin + (xmax - xmin) / 2) / width
                center_y = (ymin + (ymax - ymin) / 2) / height
                
                txt.write(f"{cls_id} {center_x:.6f} {center_y:.6f} {bb_width:.6f} {bb_height:.6f}\n")

print("Conversion and split done! Check data/processed/")