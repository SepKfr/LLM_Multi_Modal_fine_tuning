from datasets import load_dataset
from torch import nn
from transformers import Adafactor, AutoModelForCausalLM
from evaluate import load
import torch
from transformers import AutoModel
from transformers.optimization import AdafactorSchedule
from collect_data.Image_to_caption import ImageCaptionData
from transformers import AutoProcessor


class GitVisionModelClassifier(nn.Module):
    def __init__(self, gitvisionmodel, d_model, num_classes=8):
        super(GitVisionModelClassifier, self).__init__()
        self.gitvisionmodel = gitvisionmodel
        self.proj_down = nn.Linear(d_model, num_classes)

    def forward(self, inputs):
        outputs = self.gitvisionmodel(**inputs)
        print(outputs.keys())
        last_hidden_state = outputs.last_hidden_state
        outputs = self.proj_down(last_hidden_state)
        return outputs


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ds = load_dataset("lambdalabs/pokemon-blip-captions")
ds = ds["train"].train_test_split(test_size=0.1)
train_ds = ds["train"]
test_ds = ds["test"]

imgC_data = ImageCaptionData(train_ds, test_ds)

gitmodel = AutoModelForCausalLM.from_pretrained("microsoft/git-base").to(device)
processor = AutoProcessor.from_pretrained("microsoft/git-base")

d_model = gitmodel.config.hidden_size

model = GitVisionModelClassifier(gitmodel, d_model).to(device)
wer = load("wer")

optimizer = Adafactor(model.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
lr_scheduler = AdafactorSchedule(optimizer)

loss_fn = nn.CrossEntropyLoss()


for epoch in range(50):

    tot_loss = 0
    for image in imgC_data.get_train_loader():

        outputs = model(image)
        print(outputs.shape)
        print(ids.shape)
        loss = loss_fn(outputs, ids)

        tot_loss += loss.item()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    print("loss: {:.3f}".format(tot_loss))

for image, caption in imgC_data.get_test_loader():

    model.eval()
    labels = model(image)
    predicted = labels[:, :, :caption.shape[-1]].argmax(-1)
    decoded_labels = processor.batch_decode(caption, skip_special_tokens=True)
    decoded_predictions = processor.batch_decode(predicted, skip_special_tokens=True)
    wer_score = wer.compute(predictions=decoded_predictions, references=decoded_labels)
    print("wer_score {:.3f}".format(wer_score))

