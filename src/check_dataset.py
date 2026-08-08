from pathlib import Path

BASE_DIR = Path(
    r"C:\Project\BrainTumorResearch\data\BRISC\segmentation_task\train"
)

IMAGE_DIR = BASE_DIR / "images"
MASK_DIR = BASE_DIR / "masks"

images = sorted(IMAGE_DIR.glob("*.jpg"))
masks = sorted(MASK_DIR.glob("*.png"))

print(f"Images: {len(images)}")
print(f"Masks:  {len(masks)}")

# Create lookup tables based on filename
image_names = {img.stem for img in images}
mask_names = {mask.stem for mask in masks}

# Check for missing masks
missing_masks = image_names - mask_names

# Check for images without corresponding image
missing_images = mask_names - image_names

print("\nDataset validation")
print("------------------")

if not missing_masks and not missing_images:
    print("✅ Every MRI image has a corresponding mask.")
else:
    print(f"❌ Images without masks: {len(missing_masks)}")
    print(f"❌ Masks without images: {len(missing_images)}")

# Check image dimensions
from PIL import Image

image_sizes = set()

for image_path in images:
    with Image.open(image_path) as img:
        image_sizes.add(img.size)

print("\nImage dimensions:")
print(image_sizes)

# Check mask dimensions
mask_sizes = set()

for mask_path in masks:
    with Image.open(mask_path) as mask:
        mask_sizes.add(mask.size)

print("Mask dimensions:")
print(mask_sizes)