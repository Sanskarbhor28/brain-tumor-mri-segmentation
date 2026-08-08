import torch
import matplotlib.pyplot as plt
import numpy as np

from dataset import BRISCDataset
from model import UNet


IMAGE_DIR = r"C:\Project\BrainTumorResearch\data\BRISC\segmentation_task\train\images"
MASK_DIR = r"C:\Project\BrainTumorResearch\data\BRISC\segmentation_task\train\masks"

CHECKPOINT = r"C:\Project\BrainTumorResearch\checkpoints\unet_epoch_4.pth"

IMAGE_SIZE = 256

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# -------------------------
# Dataset
# -------------------------

dataset = BRISCDataset(
    image_dir=IMAGE_DIR,
    mask_dir=MASK_DIR,
    image_size=IMAGE_SIZE
)


# Use the same deterministic split as training
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

_, val_dataset = torch.utils.data.random_split(
    dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)


# -------------------------
# Model
# -------------------------

model = UNet().to(device)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("Loaded checkpoint:", CHECKPOINT)


# -------------------------
# Select validation image
# -------------------------

image, mask = val_dataset[0]

image_input = image.unsqueeze(0).to(device)

with torch.no_grad():
    output = model(image_input)

probability = torch.sigmoid(output)

prediction = (probability > 0.5).float()


# -------------------------
# Convert to NumPy
# -------------------------

image_np = image.squeeze(0).numpy()
mask_np = mask.squeeze(0).numpy()
prediction_np = prediction.squeeze().cpu().numpy()


# -------------------------
# Calculate Dice
# -------------------------

intersection = (prediction_np * mask_np).sum()

dice = (
    (2 * intersection + 1e-6)
    /
    (prediction_np.sum() + mask_np.sum() + 1e-6)
)

print(f"Dice Score: {dice:.4f}")


# -------------------------
# Visualization
# -------------------------

plt.figure(figsize=(16, 4))


plt.subplot(1, 4, 1)
plt.imshow(image_np, cmap="gray")
plt.title("MRI")
plt.axis("off")


plt.subplot(1, 4, 2)
plt.imshow(mask_np, cmap="gray")
plt.title("Ground Truth")
plt.axis("off")


plt.subplot(1, 4, 3)
plt.imshow(prediction_np, cmap="gray")
plt.title("U-Net Prediction")
plt.axis("off")


plt.subplot(1, 4, 4)
plt.imshow(image_np, cmap="gray")
plt.imshow(prediction_np, cmap="Reds", alpha=0.4)
plt.title("Prediction Overlay")
plt.axis("off")


plt.tight_layout()
plt.show()