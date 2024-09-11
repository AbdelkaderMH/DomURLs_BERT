import torch.nn as nn
import torch
from transformers import AutoModel
import torch.nn.functional as F


class PLMEncoder(nn.Module):
    def __init__(self, output_size, drop_prob=0.2, pretrained_path='amahdaouy/BERT_DOMURLS'):
        super(PLMEncoder, self).__init__()
        self.transformer = AutoModel.from_pretrained(pretrained_path)
        self.dropout = nn.Dropout(drop_prob)
        self.Classifier = nn.Linear(self.transformer.config.hidden_size, output_size)


    def forward(self, input_ids=None, attention_mask=None):
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        pooled = outputs[1]
        pooled = self.dropout(pooled)
        logits = self.Classifier(pooled)
        return logits