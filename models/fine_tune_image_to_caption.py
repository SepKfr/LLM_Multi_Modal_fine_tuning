import torch
from torch import nn
from transformers import AutoModel

from modules.Transformers import Transformer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ImageToCaptionFineTune(nn.Module):
    def __init__(self):
        super(ImageToCaptionFineTune, self).__init__()

        self.auto_model = AutoModel.from_pretrained("microsoft/git-base").to(device)
        d_model = self.auto_model.config.hidden_size
        self.proj_down = nn.Linear(d_model, 128)
        self.fine_tune_model = Transformer(d_model=128, attn_type="ATA")
        self.proj_up = nn.Linear(128, d_model)

    def forward(self, inputs):

        outputs = self.auto_model(**inputs)
        outputs = outputs.logits[:, -8:, :]
        inputs_to_fine_tune = self.proj_up(outputs)
        outputs_fine_tune = self.fine_tune_model(inputs_to_fine_tune)
        outputs = self.proj_up(outputs_fine_tune)
        return outputs

