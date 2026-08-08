from pathlib import Path
from PIL import Image
import numpy as np
from collections import Counter

MASK_DIR = Path(
    r"C:\Project\BrainTumorResearch\data\BRISC\segmentation_task\train\masks"
)

masks = sorted(MASK_DIR.glob("*.png"))

for mask_path in masks[:3]:
    mask = np.array(Image.open(mask_path).convert("L"))

    counts = Counter(mask.flatten())

    print("\n", mask_path.name)
    print("-" * 50)

    for value, count in sorted(counts.items()):
        percentage = 100 * count / mask.size
        print(f"Value {value:3d}: {count:7d} pixels ({percentage:.3f}%)")