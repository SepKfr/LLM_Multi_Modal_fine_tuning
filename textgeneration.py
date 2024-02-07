import evaluate
import torch
from datasets import load_dataset
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Adafactor
from transformers.optimization import AdafactorSchedule

imdb = load_dataset("imdb")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}

accuracy = evaluate.load("accuracy")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2, id2label=id2label, label2id=label2id).to(device)

optimizer = Adafactor(model.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
lr_scheduler = AdafactorSchedule(optimizer)


def collate_fn(batch):
    # Extract sequences
    sequences = [item["text"] for item in batch]

    # Pad sequences using tokenizer directly
    encoded_data = tokenizer(sequences, truncation=True)

    # Access padded input_ids and labels
    padded_sequences = encoded_data["input_ids"]

    padded_sequences = torch.tensor(padded_sequences, device=device)

    # Filter out None values from labels:

    labels = [item.get("label") for item in batch]
    labels = torch.tensor(labels, device=device)

    return padded_sequences, labels


train_dataloader = DataLoader(imdb["train"], batch_size=64, collate_fn=collate_fn)
test_dataloader = DataLoader(imdb["test"], batch_size=64, collate_fn=collate_fn)

loss_fn = nn.CrossEntropyLoss()
epochs = 15

for epoch in range(epochs):
    model.train()
    for i, batch in enumerate(train_dataloader):

        input_ids, labels = batch
        outputs = model(input_ids).squeeze(1)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

model.eval()
for batch in test_dataloader:
    input_ids, labels = batch
    outputs = model(input_ids).squeeze(1)
    predicted = torch.argmax(outputs, dim=-1)

    acc = accuracy.compute(predictions=predicted, references=labels)
    print(acc)

