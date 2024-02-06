import huggingface_hub
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.utils.data import DataLoader, SubsetRandomSampler
from datasets import load_dataset
from PIL import Image


# Function to process COCO dataset into a DataLoader
def get_coco_dataloader(data, batch_size=2, shuffle=True):
    # Define transformation function
    transform = lambda x: F.to_tensor(x)

    # Define collate function for DataLoader
    def collate_fn(batch):
        images = [item['image'] for item in batch]
        captions = [item['caption'] for item in batch]
        return transform(images), captions

    sampler = SubsetRandomSampler(range(int(0.2 * len(data))))
    # Create DataLoader
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=shuffle,
                            collate_fn=collate_fn, sampler=sampler)

    return dataloader


huggingface_hub.login()

# Set device for GPU usage (if available)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_data = load_dataset("detection-datasets/coco", split="train")
train_eval = train_data.train_test_split(test_size=0.1)
train_data = train_data.get("train")
valid_data = train_data.get("test")
test_data = load_dataset("detection-datasets/coco", split="test")

train_loader = get_coco_dataloader(train_data)
valid_loader = get_coco_dataloader(valid_data)
test_loader = get_coco_dataloader(test_data)

# Load a pre-trained Faster R-CNN model
image_model = fasterrcnn_resnet50_fpn(pretrained=True)
multimodel = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-base")
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")

for image, caption in train_loader:
    with torch.no_grad:
        input_ids = tokenizer(caption, return_tensors="pt")["input_ids"]
        img_features = image_model(image)
        inputs = {
            "input_ids": input_ids,
            "attention_mask": input_ids.ne(tokenizer.pad_token_id),
            "decoder_input_ids": torch.zeros_like(input_ids),
            "image_features": img_features,
        }
        outputs = multimodel(**inputs)
        generated_caption = tokenizer.decode(outputs.sequences, skip_special_tokens=True)
        print(generated_caption)

