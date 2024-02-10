import torch
from torch import nn
from transformers import AutoModel

from modules.Transformers import Transformer
from modules.coarse_fine_grained import PredictBlurDenoise

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ImageToCaptionFineTune(nn.Module):
    def __init__(self):
        super(ImageToCaptionFineTune, self).__init__()

        self.auto_model = AutoModel.from_pretrained("microsoft/git-base").to(device)
        d_model = self.auto_model.config.hidden_size
        self.fine_tune_model = PredictBlurDenoise(d_model=d_model, num_inducing=8)

    def forward(self, inputs):

        outputs = self.auto_model(**inputs)
        outputs = outputs.last_hidden_state[:, -2:, :]
        print(outputs.shape)
        outputs_fine_tune = self.fine_tune_model(outputs)
        return outputs_fine_tune

