import huggingface_hub
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from torchvision.datasets import CocoCaptions
from torch.utils.data import DataLoader
from datasets import load_dataset
from PIL import Image


# Function to load COCO dataset and prepare DataLoader
def get_coco_dataloader(root, split, batch_size=2, shuffle=True):

    transform = lambda x: F.to_tensor(x)

    coco_dataset = CocoCaptions(root, split, transform=transform)

    dataloader = DataLoader(coco_dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader


huggingface_hub.login()

# Set device for GPU usage (if available)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_data = load_dataset("detection-datasets/coco", split="train")
print(train_data.data.keys())

# Load a pre-trained Faster R-CNN model
model = fasterrcnn_resnet50_fpn(pretrained=True)