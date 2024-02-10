import torch
from torch import nn
from transformers import AutoModelForSequenceClassification

from modules.Transformers import Transformer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TextClassifierFineTune(nn.Module):
    def __init__(self):
        super(TextClassifierFineTune, self).__init__()

        id2label = {0: "NEGATIVE", 1: "POSITIVE"}
        label2id = {"NEGATIVE": 0, "POSITIVE": 1}

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=2, id2label=id2label, label2id=label2id).to(device)

        d_model = self.model.config.hidden_size
        self.fine_tune_model = Transformer(d_model=d_model, attn_type="ATA")

    def forward(self, inputs):

        outputs = self.auto_model(**inputs)
        outputs = outputs.logits
        outputs = self.fine_tune_model(outputs)
        return outputs