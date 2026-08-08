import cv2
import numpy as np


def save_mask(mask_tensor, save_path):

    mask = mask_tensor.squeeze().cpu().numpy()

    mask = (mask > 0.5).astype(np.uint8) * 255

    cv2.imwrite(save_path, mask)