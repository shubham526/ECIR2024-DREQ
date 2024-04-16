import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, DistilBertModel, T5EncoderModel


class TextEmbedding(nn.Module):
    def __init__(self, pretrained: str) -> None:
        super(TextEmbedding, self).__init__()
        self.pretrained = pretrained
        self.config = AutoConfig.from_pretrained(self.pretrained)
        if pretrained == 't5-base':
            self.encoder = T5EncoderModel.from_pretrained(self.pretrained, config=self.config)
        else:
            self.encoder = AutoModel.from_pretrained(self.pretrained, config=self.config)

    def forward(self, input_ids, attention_mask, token_type_ids) -> torch.Tensor:
        if isinstance(self.encoder, DistilBertModel):
            output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            return output[0][:, 0, :]
        elif isinstance(self.encoder, T5EncoderModel):
            last_hidden_state = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            return torch.mean(last_hidden_state, dim=1)  # Mean pooling
        else:
            output = self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            return output[0][:, 0, :]


class QueryEmbedding(nn.Module):
    def __init__(self, pretrained: str) -> None:
        super(QueryEmbedding, self).__init__()
        self.encoder = TextEmbedding(pretrained=pretrained)
        self.fc = nn.Linear(in_features=768, out_features=100)

    def forward(self, input_ids, attention_mask, token_type_ids) -> torch.Tensor:
        text_emb = self.encoder(input_ids, attention_mask, token_type_ids)
        final_emb = self.fc(text_emb)
        return final_emb


class DocEmbedding(nn.Module):
    def __init__(self) -> None:
        super(DocEmbedding, self).__init__()
        self.fc = nn.Linear(in_features=768 + 300, out_features=100)

    def forward(self, text_emb, entity_emb) -> torch.Tensor:
        concat_embedding = torch.cat((text_emb, entity_emb), dim=1)
        final_emb = self.fc(concat_embedding)
        return final_emb


class DocRankingModel(nn.Module):
    def __init__(self, pretrained: str):
        super(DocRankingModel, self).__init__()
        self.query_encoder = QueryEmbedding(pretrained=pretrained)
        self.doc_encoder = DocEmbedding()
        self.score = nn.Linear(in_features=500, out_features=1)

    def forward(
            self,
            query_input_ids,
            query_attention_mask,
            query_token_type_ids,
            doc_text_emb,
            doc_entity_emb
    ):
        query_emb = self.query_encoder(query_input_ids, query_attention_mask, query_token_type_ids)
        doc_emb = self.doc_encoder(doc_text_emb, doc_entity_emb)
        emb_sub = torch.sub(input=query_emb, other=doc_emb, alpha=1)
        emb_add = torch.add(input=query_emb, other=doc_emb, alpha=1)
        emb_mul = query_emb * doc_emb
        concat_embedding = torch.cat((query_emb, doc_emb, emb_add, emb_sub, emb_mul), dim=1)
        score = self.score(concat_embedding)
        return score.squeeze(dim=1), concat_embedding
