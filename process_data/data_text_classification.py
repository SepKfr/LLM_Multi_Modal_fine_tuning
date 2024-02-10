import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TextClassificationData:
    def __init__(self, train, test, val=None, check_point="distilbert-base-uncased"):

        self.tokenizer = AutoTokenizer.from_pretrained(check_point)

        self._train_dataloader = DataLoader(train, batch_size=64, collate_fn=self.collate_fn)

        if val is not None:

            self._val_dataloader = DataLoader(val, batch_size=64, collate_fn=self.collate_fn)

        self._test_dataloader = DataLoader(test, batch_size=64, collate_fn=self.collate_fn)

    def get_train_loader(self):
        return self._train_dataloader

    def get_test_loader(self):
        return self._test_dataloader

    def get_val_loader(self):
        return self._val_dataloader

    def collate_fn(self, batch):
        # Extract sequences
        sequences = [item["text"] for item in batch]

        # Pad sequences using tokenizer directly
        inputs = self.tokenizer(sequences, return_tensors="pt", truncation=True, max_length=64, padding="max_length")

        return inputs.to(device)