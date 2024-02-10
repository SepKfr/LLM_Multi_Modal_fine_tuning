import torch
from torch import nn
from transformers import AutoModelForSequenceClassification

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TextClassifier(nn.Module):
    def __init__(self):
        super(TextClassifier, self).__init__()

        id2label = {0: "NEGATIVE", 1: "POSITIVE"}
        label2id = {"NEGATIVE": 0, "POSITIVE": 1}

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=2, id2label=id2label, label2id=label2id).to(device)

    def forward(self, inputs):

        outputs = self.auto_model(**inputs)
        outputs = outputs.logits

        return outputs