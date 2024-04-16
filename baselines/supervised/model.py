import torch.nn as nn
import torch
from transformers import AutoConfig, AutoModel, DistilBertModel, T5EncoderModel


class BaselineModel(nn.Module):
    def __init__(self, pretrained: str):
        super(BaselineModel, self).__init__()
        self.pretrained = pretrained
        self.config = AutoConfig.from_pretrained(self.pretrained)
        self.classifier = nn.Linear(self.config.hidden_size, 2)
        if pretrained == 't5-base':
            self.encoder = T5EncoderModel.from_pretrained(self.pretrained, config=self.config)
        else:
            self.encoder = AutoModel.from_pretrained(self.pretrained, config=self.config)

    def forward(self, input_ids, attention_mask, token_type_ids):
        if isinstance(self.encoder, DistilBertModel):
            output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden_state = output.last_hidden_state
            logits = last_hidden_state[:, 0, :]
            score = self.classifier(logits).squeeze(-1)
            return score, logits
        elif isinstance(self.encoder, T5EncoderModel):
            last_hidden_state = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            pooled = torch.mean(last_hidden_state, dim=1)  # Mean pooling
            score = self.classifier(pooled).squeeze(-1)
            return score, pooled
        else:
            output = self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            last_hidden_state = output.last_hidden_state
            logits = last_hidden_state[:, 0, :]
            score = self.classifier(logits).squeeze(-1)
            return score, logits
