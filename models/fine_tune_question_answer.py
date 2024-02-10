import torch
from torch import nn
from transformers import AutoModelForQuestionAnswering

from modules.Transformers import Transformer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class QuestionAnswerFineTune(nn.Module):
    def __init__(self):
        super(QuestionAnswerFineTune, self).__init__()

        self.model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-uncased").to(device)
        d_model = self.model.config.hidden_size
        self.fine_tune_model = Transformer(d_model=d_model, attn_type="ATA")

    def forward(self, inputs):

        outputs = self.model(**inputs)
        return outputs
