import torch.nn as nn
import torch

torch.manual_seed(42)


class CustomModel(nn.Module):
    def __init__(self, base_model, labels):
        super(CustomModel, self).__init__()
        self.base_model = base_model
        self.classifier = nn.Linear(768, len(labels))

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(
            input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(
            # Use the output of the [CLS] token
            outputs.last_hidden_state[:, 0, :])
        return logits
