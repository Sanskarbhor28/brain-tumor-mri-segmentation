import os
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split

from dataset import BRISCDataset
from model import UNet


IMAGE_DIR = r"C:\Project\BrainTumorResearch\data\BRISC\segmentation_task\train\images"
MASK_DIR = r"C:\Project\BrainTumorResearch\data\BRISC\segmentation_task\train\masks"

CHECKPOINT = r"C:\Project\BrainTumorResearch\checkpoints\unet_epoch_4.pth"

IMAGE_SIZE = 256
BATCH_SIZE = 8
NUM_WORST = 10


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# =========================
# Dataset
# =========================

dataset = BRISCDataset(
    image_dir=IMAGE_DIR,
    mask_dir=MASK_DIR,
    image_size=IMAGE_SIZE
)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

_, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# =========================
# Model
# =========================

model = UNet().to(device)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("Loaded:", CHECKPOINT)


# =========================
# Evaluate
# =========================

results = []

image_index = 0

with torch.no_grad():

    for images, masks in val_loader:

        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)

        probabilities = torch.sigmoid(outputs)

        predictions = (probabilities > 0.5).float()

        for i in range(images.size(0)):

            pred = predictions[i, 0].cpu().numpy()
            target = masks[i, 0].cpu().numpy()
            probability = probabilities[i, 0].cpu().numpy()

            gt_pixels = np.count_nonzero(target)
            pred_pixels = np.count_nonzero(pred)

            max_probability = probability.max()

            intersection = (pred * target).sum()

            dice = (
                (2 * intersection + 1e-6)
                /
                (pred.sum() + target.sum() + 1e-6)
            )

            results.append({
                "index": image_index,
                "dice": float(dice),
                "gt_pixels": int(gt_pixels),
                "pred_pixels": int(pred_pixels),
                "max_probability": float(max_probability)
            })

            image_index += 1


# =========================
# Sort by Dice
# =========================

results.sort(key=lambda x: x["dice"])

print("\n" + "=" * 80)
print("WORST CASE ANALYSIS")
print("=" * 80)

for rank, item in enumerate(results[:NUM_WORST], start=1):

    print(
        f"\nRank {rank}"
        f"\nValidation index       : {item['index']}"
        f"\nDice                   : {item['dice']:.4f}"
        f"\nGround-truth pixels    : {item['gt_pixels']}"
        f"\nPredicted tumor pixels : {item['pred_pixels']}"
        f"\nMaximum probability    : {item['max_probability']:.4f}"
    )

print("\n" + "=" * 80)