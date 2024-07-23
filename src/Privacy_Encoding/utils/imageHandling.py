import numpy as np
from ensure import ensure_annotations
from typing import List
import os
from PIL import Image
from src.Privacy_Encoding.logging import logger
import matplotlib.pyplot as plt


def normalize_dataset(dataset):
    # Assuming 'dataset' is a 3D NumPy array representing images
    # Normalize to the range [0, 1]
    normalized_dataset = (dataset.astype(np.float32) / 255.0).clip(0.0, 1.0)

    # Alternatively, normalize to the range [-1, 1]
    # normalized_dataset = (dataset.astype(np.float32) / 127.5) - 1.0

    return normalized_dataset




@ensure_annotations
def save_images_to_directory(directory_path: str, x_train, y_train):
    os.makedirs(directory_path, exist_ok=True)

    for count, (image, label) in enumerate(zip(x_train, y_train)):
        # Convert normalized values to the range [0, 255]
                


        # Create the file name
        file_name = f"{count}_{label}.jpg"
        file_path = os.path.join(directory_path, file_name)

        plt.subplot(1,1,1)
        plt.grid('off')
        plt.axis("off")
        plt.imshow(image, cmap='gray')
        plt.savefig(file_path, bbox_inches='tight', pad_inches=0)

    logger.info(f"All Images are saved to {directory_path} successfully.")