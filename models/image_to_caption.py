import torch
from torch import nn
from transformers import AutoModelForCausalLM

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ImageToCaption(nn.Module):
    def __init__(self):
        super(ImageToCaption, self).__init__()

        self.auto_model = AutoModelForCausalLM.from_pretrained("microsoft/git-base").to(device)

    def forward(self, inputs):

        outputs = self.auto_model(**inputs)
        outputs = outputs.logits[:, -8:, :]

        return outputs

