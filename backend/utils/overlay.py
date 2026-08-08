import cv2
import numpy as np


def create_overlay(original_path, mask_path, output_path):

    # Read original image
    image = cv2.imread(original_path)

    # Read predicted mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Resize mask to original image size
    mask = cv2.resize(
        mask,
        (image.shape[1], image.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )

    # Binary mask
    _, mask = cv2.threshold(
        mask,
        127,
        255,
        cv2.THRESH_BINARY
    )

    # Create red overlay
    red = np.zeros_like(image)
    red[:, :, 2] = mask

    alpha = 0.4

    overlay = cv2.addWeighted(
        image,
        1.0,
        red,
        alpha,
        0
    )

    cv2.imwrite(output_path, overlay)