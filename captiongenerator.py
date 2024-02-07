import numpy as np
from datasets import load_dataset
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoProcessor, AutoTokenizer
from evaluate import load
import torch
from transformers import GitVisionModel

CUDA_LAUNCH_BLOCKING=1

class GitVisionModelClassifier(nn.Module):
    def __init__(self, gitvisionmodel, d_model, num_classes):
        super(GitVisionModelClassifier, self).__init__()
        self.gitvisionmodel = gitvisionmodel
        self.proj_down = nn.Linear(d_model, num_classes)

    def forward(self, inputs):
        outputs = self.gitvisionmodel(**inputs)
        last_hidden_state = outputs.last_hidden_state
        outputs = self.proj_down(last_hidden_state[:, -1, :])
        return outputs


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ds = load_dataset("lambdalabs/pokemon-blip-captions")
ds = ds["train"].train_test_split(test_size=0.1)
train_ds = ds["train"]
test_ds = ds["test"]

processor = AutoProcessor.from_pretrained("microsoft/git-base")
gitmodel = GitVisionModel.from_pretrained("microsoft/git-base")
tokenizer = AutoTokenizer.from_pretrained("microsoft/git-base")

d_model = gitmodel.config.hidden_size

model = GitVisionModelClassifier(gitmodel, d_model, num_classes=8)
wer = load("wer")

optimizer = torch.optim.AdamW(model.parameters())

n_uniques = 0
max_length = 0

for row in train_ds:

    caption = row["text"]
    encoded_data = tokenizer(
        caption, padding=True, truncation=True
    )
    padded_sequences = encoded_data["input_ids"]
    padded_sequences = torch.tensor(padded_sequences, device=device)
    n_uniques_in = len(torch.unique(padded_sequences))
    n_uniques = n_uniques_in if n_uniques < n_uniques_in else n_uniques
    max_length_in = len(padded_sequences)
    max_length = max_length_in if max_length < max_length_in else max_length


def collate_fn(batch):

    images = [x["image"] for x in batch]
    captions = [x["text"] for x in batch]
    inputs = processor(images=images, return_tensors="pt")

    encoded_data = tokenizer(
        captions, padding=True, truncation=True
    )
    # Access padded input_ids and labels
    padded_sequences = encoded_data["input_ids"]
    padded_sequences = torch.tensor(padded_sequences, device=device)
    one_hot = torch.nn.functional.one_hot(padded_sequences, n_uniques)
    return inputs, one_hot


train_dataloader = DataLoader(train_ds, batch_size=64, collate_fn=collate_fn)

loss_fn = nn.CrossEntropyLoss()

tot_loss = 0
for epoch in range(15):
    for image, caption in train_dataloader:

        outputs = model(image)
        print(caption.shape)
        # loss = loss_fn(outputs, caption)
        # tot_loss += loss.item()
        # loss.backward()
        # optimizer.step()
        # optimizer.zero_grad()

    print("loss: {:3f}", tot_loss)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predicted = logits.argmax(-1)
    decoded_labels = processor.batch_decode(labels, skip_special_tokens=True)
    decoded_predictions = processor.batch_decode(predicted, skip_special_tokens=True)
    wer_score = wer.compute(predictions=decoded_predictions, references=decoded_labels)
    return {"wer_score": wer_score}
#
#
# from PIL import Image
# import requests
#
# url = "https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/pokemon.png"
# image = Image.open(requests.get(url, stream=True).raw)
#
# device = "cuda" if torch.cuda.is_available() else "cpu"
#
# inputs = processor(images=image, return_tensors="pt").to(device)
# pixel_values = inputs.pixel_values
#
# generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
# generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
# print("generated caption:", generated_caption)
# import torch
#
#
# # Initializing a GitVisionConfig with microsoft/git-base style configuration
# configuration = GitVisionConfig()
#
# # Initializing a GitVisionModel (with random weights) from the microsoft/git-base style configuration
# model = GitVisionModel(configuration)
#
# # Accessing the model configuration
# configuration = model.config
#
# from PIL import Image
# import requests
# from transformers import AutoProcessor, GitVisionModel
#
# processor = AutoProcessor.from_pretrained("microsoft/git-base")
# model = GitVisionModel.from_pretrained("microsoft/git-base")
#
# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)
#
# inputs = processor(images=image, return_tensors="pt")
#
# outputs = model(**inputs)
# last_hidden_state = outputs.last_hidden_state
#
# from transformers import GitConfig, GitModel
#
# # Initializing a GIT microsoft/git-base style configuration
# configuration = GitConfig()
#
# # Initializing a model (with random weights) from the microsoft/git-base style configuration
# model = GitModel(configuration)
#
# # Accessing the model configuration
# configuration = model.config
#
# from transformers import AutoProcessor, AutoModel
# import requests
# from PIL import Image
#
# processor = AutoProcessor.from_pretrained("microsoft/git-base")
# model = AutoModel.from_pretrained("microsoft/git-base")
#
# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)
#
# text = "this is an image of two cats"
#
# inputs = processor(text, images=image, return_tensors="pt")
#
# outputs = model(**inputs)
# last_hidden_state = outputs.last_hidden_state
#
# from transformers import AutoProcessor, AutoModelForCausalLM
# import requests
# from PIL import Image
#
# processor = AutoProcessor.from_pretrained("microsoft/git-base-coco")
# model = AutoModelForCausalLM.from_pretrained("microsoft/git-base-coco")
#
# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)
#
# pixel_values = processor(images=image, return_tensors="pt").pixel_values
#
# generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
# generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
# print(generated_caption)



