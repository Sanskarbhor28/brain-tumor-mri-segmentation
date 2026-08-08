from pathlib import Path
from PIL import Image
import numpy as np

MASK_DIR = Path(
    r"C:\Project\BrainTumorResearch\data\BRISC\segmentation_task\train\masks"
)

masks = sorted(MASK_DIR.glob("*.png"))

for mask_path in masks[:3]:
    img = Image.open(mask_path)

    print("\n" + "=" * 60)
    print("File:", mask_path.name)
    print("PIL mode:", img.mode)
    print("Size:", img.size)

    mask = np.array(img)

    print("NumPy shape:", mask.shape)
    print("NumPy dtype:", mask.dtype)
    print("Unique values:", np.unique(mask))

    # Check whether the PNG has a palette
    if img.mode == "P":
        palette = img.getpalette()

        print("Palette detected: YES")

        # Print RGB colors corresponding to used palette indices
        used_values = np.unique(mask)

        print("\nPalette mapping:")
        for value in used_values:
            start = int(value) * 3
            rgb = tuple(palette[start:start + 3])
            print(f"{value:3d} -> RGB {rgb}")
    else:
        print("Palette detected: NO")