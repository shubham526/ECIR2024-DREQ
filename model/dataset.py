from typing import Dict, Any
import json
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class DocRankingDataset(Dataset):
    def __init__(self, dataset, tokenizer, train, max_len):

        self._dataset = dataset
        self._tokenizer = tokenizer
        self._max_len = max_len
        self._train = train
        self._read_data()
        self._count = len(self._examples)

    def _read_data(self):
        with open(self._dataset, 'r') as f:
            self._examples = [json.loads(line) for line in tqdm(f)]

    def _create_input(self, text):
        encoded_dict = self._tokenizer.encode_plus(
            text=text,
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=self._max_len,  # Pad & truncate all sentences.
            padding='max_length',
            truncation=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_token_type_ids=True  # Construct token type ids
        )

        return encoded_dict['input_ids'], encoded_dict['attention_mask'], encoded_dict['token_type_ids']

    def collate(self, batch):
        query_input_ids = torch.tensor([item['query_input_ids'] for item in batch])
        query_attention_mask = torch.tensor([item['query_attention_mask'] for item in batch])
        query_token_type_ids = torch.tensor([item['query_token_type_ids'] for item in batch])
        doc_text_emb = torch.tensor([item['doc_text_emb'] for item in batch])
        doc_entity_emb = torch.tensor([item['doc_entity_emb'] for item in batch])
        label = torch.tensor([item['label'] for item in batch], dtype=torch.float)
        if self._train:
            return {
                'query_input_ids': query_input_ids,
                'query_attention_mask': query_attention_mask,
                'query_token_type_ids': query_token_type_ids,
                'doc_text_emb': doc_text_emb,
                'doc_entity_emb': doc_entity_emb,
                'label': label
            }
        else:
            query_id = [item['query_id'] for item in batch]
            doc_id = [item['doc_id'] for item in batch]
            return {
                'query_id': query_id,
                'doc_id': doc_id,
                'label': label,
                'query_input_ids': query_input_ids,
                'query_attention_mask': query_attention_mask,
                'query_token_type_ids': query_token_type_ids,
                'doc_text_emb': doc_text_emb,
                'doc_entity_emb': doc_entity_emb,
            }

    def __len__(self) -> int:
        return self._count

    def __getitem__(self, index: int) -> Dict[str, Any]:
        example = self._examples[index]
        query_input_ids, query_attention_mask, query_token_type_ids = self._create_input(example['query'])
        doc_entity_emb = example['doc_ent_emb']
        doc_text_emb = example['doc_chunk_embeddings']
        if self._train:
            return {
                'query_input_ids': query_input_ids,
                'query_attention_mask': query_attention_mask,
                'query_token_type_ids': query_token_type_ids,
                'doc_text_emb': doc_text_emb,
                'doc_entity_emb': doc_entity_emb,
                'label': example['label']
            }
        else:
            return {
                'query_id': example['query_id'],
                'doc_id': example['doc_id'],
                'label': example['label'],
                'query_input_ids': query_input_ids,
                'query_attention_mask': query_attention_mask,
                'query_token_type_ids': query_token_type_ids,
                'doc_text_emb': doc_text_emb,
                'doc_entity_emb': doc_entity_emb,
            }
