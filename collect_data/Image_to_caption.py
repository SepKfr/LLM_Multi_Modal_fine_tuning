import torch
from datasets import load_dataset
from sympy.physics.control.control_plots import np
from torch.utils.data import DataLoader
from transformers import AutoProcessor, AutoTokenizer

ds = load_dataset("lambdalabs/pokemon-blip-captions")
ds = ds["train"].train_test_split(test_size=0.1)
train_ds = ds["train"]
test_ds = ds["test"]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ImageCaptionData:

    def __init__(self, train, test, val=None, check_point="microsoft/git-base"):

        self.processor = AutoProcessor.from_pretrained(check_point)
        self.tokenizer = AutoTokenizer.from_pretrained(check_point)
        self._train = train
        self._test = test
        self._val = val

    def get_train_loader(self):
        return DataLoader(self._train, batch_size=64, collate_fn=self.transforms)

    def get_test_loader(self):
        return self._test

    def get_val_loader(self):
        return self._val

    def transforms(self, example_batch):
        images = [x["image"] for x in example_batch]
        captions = [x["text"] for x in example_batch]
        inputs = self.processor(images=images, text=captions, max_length=16,
                                truncation=True, padding="max_length", return_tensors="pt")
        return inputs.to(device)






