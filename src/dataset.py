from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import cv2


class BRISCDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_size=256):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.image_size = image_size

        self.images = sorted(self.image_dir.glob("*.jpg"))
        self.masks = sorted(self.mask_dir.glob("*.png"))

        if len(self.images) != len(self.masks):
            raise ValueError(
                f"Images ({len(self.images)}) and masks ({len(self.masks)}) "
                "have different counts."
            )

        # Verify matching filenames
        for image_path, mask_path in zip(self.images, self.masks):
            image_id = image_path.stem
            mask_id = mask_path.stem

            if image_id != mask_id:
                raise ValueError(
                    f"Mismatch:\n{image_path.name}\n{mask_path.name}"
                )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        mask_path = self.masks[index]

        # Load MRI image
        image = np.array(
            Image.open(image_path).convert("L"),
            dtype=np.float32
        )

        # Load mask
        mask = np.array(
            Image.open(mask_path).convert("L"),
            dtype=np.uint8
        )

        # Resize image
        image = cv2.resize(
            image,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_LINEAR
        )

        # Resize mask using nearest-neighbor
        mask = cv2.resize(
            mask,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_NEAREST
        )

        # Normalize MRI to [0, 1]
        image = image / 255.0

        # Convert mask to binary
        # 0 = background
        # 1 = tumor
        mask = (mask > 200).astype(np.float32)

        # Add channel dimension
        image = np.expand_dims(image, axis=0)
        mask = np.expand_dims(mask, axis=0)

        # Convert to PyTorch tensors
        image = torch.tensor(image, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)

        return image, mask