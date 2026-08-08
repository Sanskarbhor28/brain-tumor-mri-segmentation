import cv2
import numpy as np
import torch


def preprocess_image(image_path, image_size=256):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    image = cv2.resize(image, (image_size, image_size))

    image = image.astype(np.float32) / 255.0

    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=0)

    image = torch.tensor(image)

    return image