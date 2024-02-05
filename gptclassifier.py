import torch.nn as nn


class GPT2Classifier(nn.Module):
    def __init__(self, gpt_model, num_classes):

        super(GPT2Classifier, self).__init__()

        self.gpt_model = gpt_model
        self.classification_head = nn.Linear(gpt_model.config.vocab_size, num_classes)

    def forward(self, input_ids, attention_mask=None):

        outputs = self.gpt_model(input_ids, attention_mask=attention_mask)

        logits = self.classification_head(outputs.logits[:, -1:, :])

        return logits
