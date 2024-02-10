import random

import numpy as np
from datasets import load_dataset
from torch import nn
from transformers import Adafactor, AutoModelForCausalLM, AutoModel
from evaluate import load
import torch
from transformers.optimization import AdafactorSchedule
from process_data.Image_to_caption import ImageCaptionData
from transformers import AutoProcessor


torch.random.manual_seed(1234)
random.seed(1234)
np.random.seed(1234)


class GitVisionModelClassifier(nn.Module):
    def __init__(self, auto_model):
        super(GitVisionModelClassifier, self).__init__()
        self.auto_model = auto_model

    def forward(self, inputs):
        outputs = self.auto_model(**inputs)
        outputs = outputs.logits[:, -8:, :]

        return outputs


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ds = load_dataset("lambdalabs/pokemon-blip-captions")
ds = ds["train"].train_test_split(test_size=0.1)
train_ds = ds["train"]
test_ds = ds["test"]

imgC_data = ImageCaptionData(train_ds, test_ds)

gitmodel = AutoModelForCausalLM.from_pretrained("microsoft/git-base").to(device)
processor = AutoProcessor.from_pretrained("microsoft/git-base")

model = GitVisionModelClassifier(gitmodel).to(device)
wer = load("wer")

optimizer = Adafactor(model.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
lr_scheduler = AdafactorSchedule(optimizer)

loss_fn = nn.CrossEntropyLoss()


for epoch in range(1):

    tot_loss = 0
    for image, ids in imgC_data.get_train_loader():

        outputs = model(image)
        loss = loss_fn(outputs[:, :, -ids.shape[-1]:], ids)
        tot_loss += loss.item()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    print("loss: {:.3f}".format(tot_loss))

for image, ids in imgC_data.get_test_loader():

    model.eval()
    labels = model(image)
    predicted = labels.argmax(-1)
    decoded_labels = processor.batch_decode(ids, skip_special_tokens=True)
    decoded_predictions = processor.batch_decode(predicted, skip_special_tokens=True)
    wer_score = wer.compute(predictions=decoded_predictions, references=decoded_labels)
    print("wer_score {:.3f}".format(wer_score))

