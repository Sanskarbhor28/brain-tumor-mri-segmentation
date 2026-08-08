import os
import torch
from torch.utils.data import DataLoader, random_split

from dataset import BRISCDataset
from residual_model import ResidualUNet
from losses import FocalDiceLoss


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_DIR = r"C:\Project\BrainTumorResearch\data\BRISC\segmentation_task\train\images"
MASK_DIR = r"C:\Project\BrainTumorResearch\data\BRISC\segmentation_task\train\masks"

CHECKPOINT_DIR = r"C:\Project\BrainTumorResearch\checkpoints\residual_unet"

IMAGE_SIZE = 256
BATCH_SIZE = 8
EPOCHS = 5

LEARNING_RATE = 1e-4

# Focal Loss parameters
FOCAL_ALPHA = 0.75
FOCAL_GAMMA = 2.0


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# CHECKPOINT DIRECTORY
# ============================================================

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ============================================================
# DATASET
# ============================================================

dataset = BRISCDataset(
    image_dir=IMAGE_DIR,
    mask_dir=MASK_DIR,
    image_size=IMAGE_SIZE
)

print("Total:", len(dataset))


# ============================================================
# TRAIN / VALIDATION SPLIT
# IMPORTANT:
# Same 80/20 split and same seed as baseline
# ============================================================

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

print("Train:", len(train_dataset))
print("Validation:", len(val_dataset))


# ============================================================
# DATA LOADERS
# ============================================================

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


# ============================================================
# MODEL
# ============================================================

model = ResidualUNet().to(device)


# ============================================================
# LOSS
# Focal Loss + Dice Loss
# ============================================================

criterion = FocalDiceLoss(
    alpha=FOCAL_ALPHA,
    gamma=FOCAL_GAMMA,
    focal_weight=1.0,
    dice_weight=1.0
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# DICE METRIC
# ============================================================

def dice_score(predictions, targets, smooth=1e-6):

    predictions = (predictions > 0.5).float()

    predictions = predictions.view(predictions.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    intersection = (predictions * targets).sum(dim=1)

    dice = (
        (2 * intersection + smooth)
        /
        (
            predictions.sum(dim=1)
            + targets.sum(dim=1)
            + smooth
        )
    )

    return dice.mean().item()


# ============================================================
# TRAINING
# ============================================================

best_val_dice = 0.0

for epoch in range(EPOCHS):

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    train_loss = 0.0
    train_dice = 0.0
    train_batches = 0

    for images, masks in train_loader:

        images = images.to(
            device,
            non_blocking=True
        )

        masks = masks.to(
            device,
            non_blocking=True
        )

        # Clear gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Calculate Focal + Dice Loss
        loss = criterion(
            outputs,
            masks
        )

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        # Calculate Dice
        probabilities = torch.sigmoid(outputs)

        dice = dice_score(
            probabilities,
            masks
        )

        train_loss += loss.item()
        train_dice += dice

        train_batches += 1


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    val_loss = 0.0
    val_dice = 0.0
    val_batches = 0

    with torch.no_grad():

        for images, masks in val_loader:

            images = images.to(
                device,
                non_blocking=True
            )

            masks = masks.to(
                device,
                non_blocking=True
            )

            # Forward pass
            outputs = model(images)

            # Validation loss
            loss = criterion(
                outputs,
                masks
            )

            # Dice
            probabilities = torch.sigmoid(outputs)

            dice = dice_score(
                probabilities,
                masks
            )

            val_loss += loss.item()
            val_dice += dice

            val_batches += 1


    # --------------------------------------------------------
    # AVERAGES
    # --------------------------------------------------------

    train_loss /= train_batches
    train_dice /= train_batches

    val_loss /= val_batches
    val_dice /= val_batches


    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print()
    print(f"Epoch [{epoch + 1}/{EPOCHS}]")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Train Dice: {train_dice:.4f}")
    print(f"Val Loss:   {val_loss:.4f}")
    print(f"Val Dice:   {val_dice:.4f}")


    # --------------------------------------------------------
    # SAVE EVERY EPOCH
    # --------------------------------------------------------

    checkpoint_path = os.path.join(
        CHECKPOINT_DIR,
        f"residual_unet_epoch_{epoch + 1}.pth"
    )

    torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "train_dice": train_dice,
            "val_loss": val_loss,
            "val_dice": val_dice,
        },
        checkpoint_path
    )

    print("Saved:", checkpoint_path)


    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if val_dice > best_val_dice:

        best_val_dice = val_dice

        best_path = os.path.join(
            CHECKPOINT_DIR,
            "best_residual_unet.pth"
        )

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "train_dice": train_dice,
                "val_loss": val_loss,
                "val_dice": val_dice,
            },
            best_path
        )

        print("⭐ New best model saved!")


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 60)
print("Training complete.")
print("=" * 60)
print(f"Best Validation Dice: {best_val_dice:.4f}")
print(f"Checkpoints: {CHECKPOINT_DIR}")