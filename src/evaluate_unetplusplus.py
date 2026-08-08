import torch
import numpy as np
from torch.utils.data import DataLoader, random_split

from dataset import BRISCDataset
from unetplusplus_model import UNetPlusPlus


# =========================
# Configuration
# =========================

IMAGE_DIR = r"C:\Project\BrainTumorResearch\data\BRISC\segmentation_task\train\images"
MASK_DIR = r"C:\Project\BrainTumorResearch\data\BRISC\segmentation_task\train\masks"

CHECKPOINT = r"C:\Project\BrainTumorResearch\checkpoints\unetplusplus\best_unetplusplus.pth"

IMAGE_SIZE = 256
BATCH_SIZE = 4

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


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
    num_workers=0,
    pin_memory=True
)

print("Validation images:", len(val_dataset))


# =========================
# Model
# =========================

model = UNetPlusPlus(
    num_classes=1,
    input_channels=1,
    deep_supervision=False
).to(device)

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
# Metric accumulators
# =========================

total_dice = 0.0
total_iou = 0.0
total_precision = 0.0
total_recall = 0.0
total_specificity = 0.0

num_images = 0


# =========================
# Evaluation
# =========================

with torch.no_grad():

    for images, masks in val_loader:

        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        outputs = model(images)

        probabilities = torch.sigmoid(outputs)

        predictions = (probabilities > 0.5).float()

        # Flatten each image separately
        predictions = predictions.view(predictions.size(0), -1)
        masks = masks.view(masks.size(0), -1)

        for pred, target in zip(predictions, masks):

            tp = (pred * target).sum().item()

            fp = (pred * (1 - target)).sum().item()

            fn = ((1 - pred) * target).sum().item()

            tn = ((1 - pred) * (1 - target)).sum().item()

            # Dice
            dice = (
                (2 * tp + 1e-6)
                /
                (2 * tp + fp + fn + 1e-6)
            )

            # IoU
            iou = (
                (tp + 1e-6)
                /
                (tp + fp + fn + 1e-6)
            )

            # Precision
            precision = (
                (tp + 1e-6)
                /
                (tp + fp + 1e-6)
            )

            # Recall / Sensitivity
            recall = (
                (tp + 1e-6)
                /
                (tp + fn + 1e-6)
            )

            # Specificity
            specificity = (
                (tn + 1e-6)
                /
                (tn + fp + 1e-6)
            )

            total_dice += dice
            total_iou += iou
            total_precision += precision
            total_recall += recall
            total_specificity += specificity

            num_images += 1


# =========================
# Final results
# =========================

mean_dice = total_dice / num_images
mean_iou = total_iou / num_images
mean_precision = total_precision / num_images
mean_recall = total_recall / num_images
mean_specificity = total_specificity / num_images


print("\n" + "=" * 50)
print("FINAL VALIDATION RESULTS")
print("=" * 50)

print(f"Images evaluated : {num_images}")
print(f"Mean Dice        : {mean_dice:.4f}")
print(f"Mean IoU         : {mean_iou:.4f}")
print(f"Mean Precision   : {mean_precision:.4f}")
print(f"Mean Recall      : {mean_recall:.4f}")
print(f"Mean Specificity : {mean_specificity:.4f}")

print("=" * 50)