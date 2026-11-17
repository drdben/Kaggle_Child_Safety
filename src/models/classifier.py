from __future__ import annotations
import torch.nn as nn
from transformers import AutoModel


class EncoderClassifier(nn.Module):
    def __init__(self, model_name: str, dropout: float = .1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, 2)

    def pooled(self, input_ids, attention_mask, **kwargs):
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        return output.pooler_output if output.pooler_output is not None else output.last_hidden_state[:, 0]

    def forward(self, input_ids, attention_mask, **kwargs):
        return self.classifier(self.dropout(self.pooled(input_ids, attention_mask, **kwargs)))
