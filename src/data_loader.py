from torch.utils.data import DataLoader, random_split

from dataset import BRISCDataset


IMAGE_DIR = r"C:\Project\BrainTumorResearch\data\BRISC\segmentation_task\train\images"
MASK_DIR = r"C:\Project\BrainTumorResearch\data\BRISC\segmentation_task\train\masks"


# Create complete dataset
dataset = BRISCDataset(
    image_dir=IMAGE_DIR,
    mask_dir=MASK_DIR,
    image_size=256
)

# 80% training / 20% validation
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size]
)

# Create DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=0
)


if __name__ == "__main__":
    print("Total images:", len(dataset))
    print("Training images:", len(train_dataset))
    print("Validation images:", len(val_dataset))

    images, masks = next(iter(train_loader))

    print("\nBatch information")
    print("----------------")
    print("Images:", images.shape)
    print("Masks:", masks.shape)
    print("Image dtype:", images.dtype)
    print("Mask dtype:", masks.dtype)