import torch.nn as nn
import torch.nn.functional as F
import torch

from transformers import (
    AutoConfig,
    AutoModelForMaskedLM,
)

import os

# Get the path of the current file
file_path = os.path.abspath(__file__)

# Get the directory containing the file
directory_path = os.path.dirname(file_path)

config_kwargs = {
    "cache_dir": None,
    "revision": 'main',
    "use_auth_token": None,
    "hidden_dropout_prob": 0.2,
    "vocab_size": 5000,
}

class URLBertForSequenceClassification(nn.Module):
    def __init__(self, output_size, drop_prob=0.2, pretrained_path=None):
        super(URLBertForSequenceClassification, self).__init__()
        config = AutoConfig.from_pretrained(f"{directory_path}/urlbert_model/", **config_kwargs)


        self.bert_model = AutoModelForMaskedLM.from_config(
            config=config,
        )
        self.bert_model.resize_token_embeddings(config_kwargs["vocab_size"])
        
        bert_dict = torch.load(f"{directory_path}/urlbert_model/urlBERT.pt", map_location='cpu')
        
        self.bert_model.load_state_dict(bert_dict)
        
        self.dropout = nn.Dropout(p=drop_prob)
        self.classifier = nn.Linear(768, output_size)

    def forward(self, input_ids=None, attention_mask=None):

        outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        
        hidden_states = outputs.hidden_states[-1][:,0,:]
        out = self.dropout(hidden_states)
        out = self.classifier(out)
        
        return out