import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from dataset import BRISCDataset
from model import UNet


# =========================
# Configuration
# =========================

IMAGE_DIR = r"C:\Project\BrainTumorResearch\data\BRISC\segmentation_task\train\images"
MASK_DIR = r"C:\Project\BrainTumorResearch\data\BRISC\segmentation_task\train\masks"

IMAGE_SIZE = 256
BATCH_SIZE = 8
EPOCHS = 5
LEARNING_RATE = 1e-4

CHECKPOINT_DIR = r"C:\Project\BrainTumorResearch\checkpoints"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# =========================
# Device
# =========================

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

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

print("Total:", len(dataset))
print("Train:", len(train_dataset))
print("Validation:", len(val_dataset))


# =========================
# Model
# =========================

model = UNet().to(device)


# =========================
# Loss Functions
# =========================

bce_loss = nn.BCEWithLogitsLoss()


def dice_loss(logits, targets, smooth=1e-6):
    probabilities = torch.sigmoid(logits)

    probabilities = probabilities.view(-1)
    targets = targets.view(-1)

    intersection = (probabilities * targets).sum()

    dice = (
        (2.0 * intersection + smooth)
        /
        (probabilities.sum() + targets.sum() + smooth)
    )

    return 1.0 - dice


def combined_loss(logits, targets):
    bce = bce_loss(logits, targets)
    dice = dice_loss(logits, targets)

    return bce + dice


# =========================
# Dice Score
# =========================

def dice_score(logits, targets, threshold=0.5, smooth=1e-6):
    probabilities = torch.sigmoid(logits)

    predictions = (probabilities > threshold).float()

    predictions = predictions.view(-1)
    targets = targets.view(-1)

    intersection = (predictions * targets).sum()

    dice = (
        (2.0 * intersection + smooth)
        /
        (predictions.sum() + targets.sum() + smooth)
    )

    return dice.item()


# =========================
# Optimizer
# =========================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# =========================
# Training
# =========================

for epoch in range(EPOCHS):

    model.train()

    train_loss = 0.0
    train_dice = 0.0

    for images, masks in train_loader:

        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad()

        outputs = model(images)

        loss = combined_loss(outputs, masks)

        loss.backward()

        optimizer.step()

        train_loss += loss.item()
        train_dice += dice_score(outputs.detach(), masks)

    train_loss /= len(train_loader)
    train_dice /= len(train_loader)


    # =========================
    # Validation
    # =========================

    model.eval()

    val_loss = 0.0
    val_dice = 0.0

    with torch.no_grad():

        for images, masks in val_loader:

            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            outputs = model(images)

            loss = combined_loss(outputs, masks)

            val_loss += loss.item()
            val_dice += dice_score(outputs, masks)

    val_loss /= len(val_loader)
    val_dice /= len(val_loader)


    print(
        f"\nEpoch [{epoch + 1}/{EPOCHS}]"
        f"\nTrain Loss: {train_loss:.4f}"
        f"\nTrain Dice: {train_dice:.4f}"
        f"\nVal Loss:   {val_loss:.4f}"
        f"\nVal Dice:   {val_dice:.4f}"
    )


    # =========================
    # Save checkpoint
    # =========================

    checkpoint_path = os.path.join(
        CHECKPOINT_DIR,
        f"unet_epoch_{epoch + 1}.pth"
    )

    torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_dice": val_dice
        },
        checkpoint_path
    )

    print("Saved:", checkpoint_path)


print("\nTraining complete.")