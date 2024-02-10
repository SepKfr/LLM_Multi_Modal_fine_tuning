import torch
from torch import nn
from transformers import AutoModelForQuestionAnswering

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class QuestionAnswer(nn.Module):
    def __init__(self):

        super(QuestionAnswer, self).__init__()

        self.model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-uncased").to(device)

    def forward(self, inputs):

        outputs = self.model(**inputs)
        return outputs
