from dataset import BRISCDataset


IMAGE_DIR = r"C:\Project\BrainTumorResearch\data\BRISC\segmentation_task\train\images"
MASK_DIR = r"C:\Project\BrainTumorResearch\data\BRISC\segmentation_task\train\masks"


dataset = BRISCDataset(
    image_dir=IMAGE_DIR,
    mask_dir=MASK_DIR,
    image_size=256
)

print("Dataset size:", len(dataset))

image, mask = dataset[0]

print("Image shape:", image.shape)
print("Image dtype:", image.dtype)
print("Image min:", image.min().item())
print("Image max:", image.max().item())

print("Mask shape:", mask.shape)
print("Mask dtype:", mask.dtype)
print("Mask min:", mask.min().item())
print("Mask max:", mask.max().item())