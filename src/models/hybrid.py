from __future__ import annotations
import torch.nn as nn
from transformers import AutoModel


class HybridClassifier(nn.Module):
    def __init__(self, muril_name: str, xlmr_name: str, dropout: float = .1):
        super().__init__()
        self.muril = AutoModel.from_pretrained(muril_name)
        self.xlmr = AutoModel.from_pretrained(xlmr_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.muril.config.hidden_size + self.xlmr.config.hidden_size, 2)

    @staticmethod
    def _pool(encoder, ids, mask):
        x = encoder(input_ids=ids, attention_mask=mask)
        return x.pooler_output if x.pooler_output is not None else x.last_hidden_state[:, 0]

    def forward(self, muril_input_ids, muril_attention_mask, xlmr_input_ids, xlmr_attention_mask):
        a = self._pool(self.muril, muril_input_ids, muril_attention_mask)
        b = self._pool(self.xlmr, xlmr_input_ids, xlmr_attention_mask)
        return self.classifier(self.dropout(__import__('torch').cat((a, b), dim=-1)))
