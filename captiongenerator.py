from datasets import load_dataset
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoProcessor
from transformers import AutoModelForCausalLM
from evaluate import load
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ds = load_dataset("lambdalabs/pokemon-blip-captions")
ds = ds["train"].train_test_split(test_size=0.1)
train_ds = ds["train"]
test_ds = ds["test"]

checkpoint = "microsoft/git-base"
processor = AutoProcessor.from_pretrained(checkpoint)


def transforms(example_batch):
    images = [x for x in example_batch["image"]]
    captions = [x for x in example_batch["text"]]
    inputs = processor(images=images, text=captions, return_tensors="pt", padding="max_length").to(device)
    inputs.update({"labels": inputs["input_ids"]})
    return inputs


train_ds.set_transform(transforms)
test_ds.set_transform(transforms)

train_dataloader = DataLoader(train_ds, batch_size=16)
test_dataloader = DataLoader(test_ds, batch_size=16)

model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

wer = load("wer")

optimizer = torch.optim.AdamW(model.parameters())

for inp in train_dataloader:

    output = model(**inp)
    loss = output["loss"]
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predicted = logits.argmax(-1)
    decoded_labels = processor.batch_decode(labels, skip_special_tokens=True)
    decoded_predictions = processor.batch_decode(predicted, skip_special_tokens=True)
    wer_score = wer.compute(predictions=decoded_predictions, references=decoded_labels)
    return {"wer_score": wer_score}


from PIL import Image
import requests

url = "https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/pokemon.png"
image = Image.open(requests.get(url, stream=True).raw)

device = "cuda" if torch.cuda.is_available() else "cpu"

inputs = processor(images=image, return_tensors="pt").to(device)
pixel_values = inputs.pixel_values

generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print("generated caption:", generated_caption)