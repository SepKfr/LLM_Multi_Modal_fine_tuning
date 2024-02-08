from datasets import load_dataset
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoProcessor, AutoTokenizer, Adafactor
from evaluate import load
import torch
from transformers import AutoModel
from transformers.optimization import AdafactorSchedule


class GitVisionModelClassifier(nn.Module):
    def __init__(self, gitvisionmodel, d_model, num_classes=100):
        super(GitVisionModelClassifier, self).__init__()
        self.gitvisionmodel = gitvisionmodel
        self.proj_down = nn.Linear(d_model, num_classes)

    def forward(self, inputs):
        outputs = self.gitvisionmodel(**inputs)
        last_hidden_state = outputs.last_hidden_state
        outputs = self.proj_down(last_hidden_state[:, -8:, :])
        return outputs


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ds = load_dataset("lambdalabs/pokemon-blip-captions")
ds = ds["train"].train_test_split(test_size=0.1)
train_ds = ds["train"]
test_ds = ds["test"]

processor = AutoProcessor.from_pretrained("microsoft/git-base")
gitmodel = AutoModel.from_pretrained("microsoft/git-base").to(device)
tokenizer = AutoTokenizer.from_pretrained("microsoft/git-base")

d_model = gitmodel.config.hidden_size

model = GitVisionModelClassifier(gitmodel, d_model).to(device)
wer = load("wer")

optimizer = Adafactor(model.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
lr_scheduler = AdafactorSchedule(optimizer)


def collate_fn(batch):

    images = [x["image"] for x in batch]
    captions = [x["text"] for x in batch]
    inputs = processor(images=images, text=captions, return_tensors="pt",
                       padding=True, truncation=True, max_length=8)
    inputs.to(device)

    encoded_data = tokenizer(
        captions, padding=True, truncation=True, max_length=8
    )
    # Access padded input_ids and labels
    padded_sequences = encoded_data["input_ids"]
    padded_sequences = torch.tensor(padded_sequences, device=device)
    unique_labels = torch.tensor(list(set(label for sublist in padded_sequences for label in sublist))).to(device)
    unique_labels = torch.unique(unique_labels)
    n_unique = len(unique_labels)
    one_hot_encoded = torch.zeros((padded_sequences.shape[0], n_unique), device=device)

    # Iterate through each sample and set the corresponding index to 1
    for i, sample in enumerate(padded_sequences):
        indices = torch.tensor([unique_labels.tolist().index(label) for label in sample]).to(device)
        one_hot_encoded[i].scatter_(0, indices, 1)
    one_hot_encoded = one_hot_encoded.to(torch.long)
    return inputs, one_hot_encoded


def collate_fn_test(batch):

    images = [x["image"] for x in batch]
    captions = [x["text"] for x in batch]
    inputs = processor(images=images, return_tensors="pt")
    inputs.to(device)

    encoded_data = tokenizer(
        captions, padding=True, truncation=True, max_length=8
    )
    # Access padded input_ids and labels
    padded_sequences = encoded_data["input_ids"]
    padded_sequences = torch.tensor(padded_sequences, device=device)

    return inputs, padded_sequences


train_dataloader = DataLoader(train_ds, batch_size=64, collate_fn=collate_fn)
test_dataloader = DataLoader(test_ds, batch_size=64, collate_fn=collate_fn)

loss_fn = nn.CrossEntropyLoss()


for epoch in range(50):

    tot_loss = 0
    for image, caption in train_dataloader:

        outputs = model(image)
        loss = loss_fn(outputs[:, :, :caption.shape[-1]], caption)
        tot_loss += loss.item()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    print("loss: {:.3f}".format(tot_loss))

for image, caption in test_dataloader:

    model.eval()
    labels = model(image)
    predicted = labels[:, :, :caption.shape[-1]].argmax(-1)
    decoded_labels = processor.batch_decode(caption, skip_special_tokens=True)
    decoded_predictions = processor.batch_decode(predicted, skip_special_tokens=True)
    wer_score = wer.compute(predictions=decoded_predictions, references=decoded_labels)
    print("wer_score {:.3f}".format(wer_score))
