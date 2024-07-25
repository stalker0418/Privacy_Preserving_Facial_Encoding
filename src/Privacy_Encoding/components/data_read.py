
from collections import defaultdict
from src.Privacy_Encoding.entity import DataReadConfig
import os
import cv2
import numpy as np
from src.Privacy_Encoding.logging import logger

class DataRead:
    def __init__(self, config: DataReadConfig):
        self.config = config

    def extract_numerical_value(self, file_name):
        return int(file_name.split('_')[self.config.image_id_index])
    


    # Read the images, and update the y_label so that missing images are removed from the labels
    def read_images(self, image_folder):
        image_files = os.listdir(image_folder)

        image_files = sorted(image_files, key= self.extract_numerical_value)

        # Initialize an empty list to store images
        x_train, y_train, label_count = [], [], defaultdict(int)

        # Loop through each image file, read it as grayscale, and append to x_train
        for image_file in image_files:
            image_path = os.path.join(image_folder, image_file)
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            # label = int(image_file.split('_')[4].split('.')[0])
            filename = image_file.split('.')[0]
            label = int(filename.split(self.config.index_separator)[self.config.label_index])

            # Check if the image is successfully loaded
            if img is not None: 
                # if label_count[label] <=cut_off_length:
                x_train.append(img)
                y_train.append(label)   
                label_count[label] += 1                                                                 
            else:
                # print(f"Failed to load {image_file}")
                logger.info(f"Failed to load {image_file}")

        # Convert the list of images to a NumPy array
        x_train, y_train = np.array(x_train), np.array(y_train)

        # expect_num = set(range(0,len(y_train)))
        # existing_num = {int(file.split('_')[2]) for file in image_files}
        # missing_num = expect_num - existing_num
        # y_train = np.delete(y_train, list(missing_num))
        # print("label count for {image_folder} is :", label_count)

        logger.info(f"label count for {image_folder} is : {label_count}")
        return x_train, y_train, label_count


    def label_count(self,image_folder):
        image_files = os.listdir(image_folder)

        image_files = sorted(image_files, key= self.extract_numerical_value)
        # Loop through each image file, read it as grayscale, and append to x_train
        label_count = defaultdict(int)

        for image_file in image_files:
            image_path = os.path.join(image_folder, image_file)
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            # label = int(image_file.split('_')[4].split('.')[0])
            filename = image_file.split('.')[0]
            label = int(filename.split(self.config.index_separator)[self.config.label_index])

            # Check if the image is successfully loaded
            if img is not None:    
                label_count[label] += 1                                                                 
            else:
                # print(f"Failed to load {image_file}")
                logger.info(f"Failed to load {image_file}")
            
            return label_count


    def normalize_dataset(self, dataset):
        # Assuming 'dataset' is a 3D NumPy array representing images
        # Normalize to the range [0, 1]
        normalized_dataset = (dataset.astype(np.float32) / 255.0).clip(0.0, 1.0)

        # Alternatively, normalize to the range [-1, 1]
        # normalized_dataset = (dataset.astype(np.float32) / 127.5) - 1.0

        return normalized_dataset

 