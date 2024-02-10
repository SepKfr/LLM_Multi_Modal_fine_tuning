import torch
from torch import nn
from transformers import AutoModelForQuestionAnswering

from modules.Transformers import Transformer
from modules.coarse_fine_grained import PredictBlurDenoise

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class QuestionAnswerFineTune(nn.Module):
    def __init__(self, fine_tune_type=1):
        super(QuestionAnswerFineTune, self).__init__()

        self.model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-uncased").to(device)
        d_model = int(self.model.config.hidden_size/2)

        if fine_tune_type == 1:
            self.fine_tune_model = Transformer(d_model=d_model, attn_type="ATA")
        else:
            self.fine_tune_model = PredictBlurDenoise(d_model=d_model, num_inducing=8)

    def forward(self, inputs):

        outputs = self.model(**inputs)
        outputs_start = outputs.start_logits
        outputs_end = outputs.end_logits
        outputs_start_fine = self.fine_tune_model(outputs_start.unsqueeze(1))
        outputs_end_fine = self.fine_tune_model(outputs_end.unsqueeze(1))
        output_dict = dict()
        output_dict["start_logits"] = outputs_start_fine
        output_dict["end_logits"] = outputs_end_fine
        return outputs
