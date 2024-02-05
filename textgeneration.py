import os

from transformers import GPT2Tokenizer, GPT2LMHeadModel
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
import huggingface_hub

from coarse_fine_grained import GPT2Classifier2
from gptclassifier import GPT2Classifier

huggingface_hub.login()

# Set device for GPU usage (if available)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_data = load_dataset("yelp_review_full", split="train")
train_eval = train_data.train_test_split(test_size=0.1, stratify_by_column="label")
train_data = train_eval.get("train")
valid_data = train_eval.get("test")
test_data = load_dataset("yelp_review_full", split="test")

product_reviews_train = train_data.filter(lambda x: "laptop" in x["text"])
product_reviews_valid = valid_data.filter(lambda x: "laptop" in x["text"])
product_reviews_test = test_data.filter(lambda x: "laptop" in x["text"])

max_value = max(product_reviews_train, key=lambda x: x["label"])["label"]

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token


def collate_fn(batch):
    # Extract sequences
    sequences = [item["text"] for item in batch]

    # Pad sequences using tokenizer directly
    encoded_data = tokenizer(
      sequences, padding=True, truncation=True, max_length=512
    )

    # Access padded input_ids and labels
    padded_sequences = encoded_data["input_ids"]

    padded_sequences = torch.tensor(padded_sequences, device=device)

    # Filter out None values from labels:

    labels = [item.get("label") for item in batch]
    labels = torch.tensor(labels, device=device)

    return padded_sequences, labels


loss_fn = torch.nn.CrossEntropyLoss()

train_dataloader = DataLoader(product_reviews_train, batch_size=8, collate_fn=collate_fn)
valid_dataloader = DataLoader(product_reviews_valid, batch_size=8, collate_fn=collate_fn)
test_dataloader = DataLoader(product_reviews_test, batch_size=8, collate_fn=collate_fn)


def run(model, optim):
    epochs = 15
    for epoch in range(epochs):
        model.train()
        for i, batch in enumerate(train_dataloader):
            input_ids, labels = batch
            outputs = model(input_ids)
            labels = torch.eye(max_value+1, device=device)[labels].to(torch.long)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optim.step()
            optim.zero_grad()

        print("train loss {:.3f}".format(loss.item()))

        model.eval()
        for batch in valid_dataloader:
            input_ids, labels = batch
            outputs = model(input_ids)
            labels = torch.eye(max_value + 1, device=device)[labels].to(torch.long)
            loss = loss_fn(outputs, labels)

        print("valid loss {:.3f}".format(loss.item()))

    model.eval()

    tot_loss = 0
    for batch in test_dataloader:
        input_ids, labels = batch
        outputs = model(input_ids)
        labels = torch.eye(max_value + 1, device=device)[labels].to(torch.long)
        loss = loss_fn(outputs, labels)
        tot_loss += loss.item()

    print("test loss {:.3f}".format(loss))

    return tot_loss


gpt_model = GPT2LMHeadModel.from_pretrained("gpt2")

gptclassifier = GPT2Classifier(gpt_model, max_value+1).to(device)
optimizer = torch.optim.AdamW(gptclassifier.parameters(), lr=1e-5)

test_loss1 = run(gptclassifier, optimizer)

# gptclassifier2 = GPT2Classifier2(d_model=64, gpt_model=gpt_model, num_classes=max_value+1)
# optimizer = torch.optim.AdamW(gptclassifier2.parameters(), lr=1e-5)
#
# test_loss2 = run(gptclassifier2, optimizer)

print("loss of GPT: {:.3f}".format(test_loss1))
#print("loss of GPT BlurDenoise: {:.3f}".format(test_loss2))

# prompt = "Write a creative description for a product:"
# new_description = model.generate(
#     tokenizer.encode(prompt, return_tensors="pt").to(device),
#     max_length=100, num_beams=2, early_stopping=True
# )
# decoded_description = tokenizer.decode(new_description[0], skip_special_tokens=True)
# print(decoded_description)
