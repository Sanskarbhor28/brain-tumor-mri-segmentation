from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

IMAGE_DIR = Path(
    r"C:\Project\BrainTumorResearch\data\BRISC\segmentation_task\train\images"
)

MASK_DIR = Path(
    r"C:\Project\BrainTumorResearch\data\BRISC\segmentation_task\train\masks"
)

image_path = sorted(IMAGE_DIR.glob("*.jpg"))[0]
mask_path = sorted(MASK_DIR.glob("*.png"))[0]

image = np.array(Image.open(image_path).convert("RGB"))
mask = np.array(Image.open(mask_path).convert("L"))

print("Image:", image_path.name)
print("Mask:", mask_path.name)
print("Image shape:", image.shape)
print("Mask shape:", mask.shape)

# Show the raw mask and a binary version
binary_mask = mask > 0

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(image)
plt.title("MRI")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(mask, cmap="gray")
plt.title("Raw Mask")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(binary_mask, cmap="gray")
plt.title("Mask > 0")
plt.axis("off")

plt.tight_layout()
plt.show()