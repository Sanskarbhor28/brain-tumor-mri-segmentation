import matplotlib.pyplot as plt
import numpy as np

from dataset import BRISCDataset


IMAGE_DIR = r"C:\Project\BrainTumorResearch\data\BRISC\segmentation_task\train\images"
MASK_DIR = r"C:\Project\BrainTumorResearch\data\BRISC\segmentation_task\train\masks"


dataset = BRISCDataset(
    image_dir=IMAGE_DIR,
    mask_dir=MASK_DIR,
    image_size=256
)

image, mask = dataset[0]

# Remove channel dimension
image = image.squeeze(0).numpy()
mask = mask.squeeze(0).numpy()

# Temporary binary visualization ONLY
binary_mask = mask > 127

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(image, cmap="gray")
plt.title("MRI - 256x256")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(mask, cmap="gray")
plt.title("Resized Raw Mask")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(image, cmap="gray")
plt.imshow(binary_mask, cmap="Reds", alpha=0.4)
plt.title("Temporary Overlay")
plt.axis("off")

plt.tight_layout()
plt.show()