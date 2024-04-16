from typing import Dict, Any
import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from tqdm import tqdm


class BaselineDataset(Dataset):
    def __init__(
            self,
            dataset,
            tokenizer: AutoTokenizer,
            train: bool,
            max_len: int
    ) -> None:

        self._dataset = dataset
        self._tokenizer = tokenizer
        self._train = train
        self._max_len = max_len
        self._read_data()
        self._count = len(self._examples)

    def _read_data(self):
        with open(self._dataset, 'r') as f:
            self._examples = [json.loads(line) for line in tqdm(f)]

    def _create_input(self, query, document):
        # Tokenize all the sentences and map the tokens to their word IDs.
        encoded_dict = self._tokenizer.encode_plus(
            text=query,
            text_pair=document,
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=self._max_len,  # Pad & truncate all sentences.
            padding='max_length',
            truncation=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_token_type_ids=True  # Construct token type ids
        )

        return encoded_dict['input_ids'], encoded_dict['attention_mask'], encoded_dict['token_type_ids']

    def __len__(self) -> int:
        return self._count

    def collate(self, batch):
        input_ids = torch.tensor([item['input_ids'] for item in batch])
        attention_mask = torch.tensor([item['attention_mask'] for item in batch])
        token_type_ids = torch.tensor([item['token_type_ids'] for item in batch])
        label = torch.tensor([item['label'] for item in batch])
        if self._train:
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids,
                'label': label
            }
        else:
            query_id = [item['query_id'] for item in batch]
            doc_id = [item['doc_id'] for item in batch]
            return {
                'query_id': query_id,
                'doc_id': doc_id,
                'label': label,
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids,
            }

    def __getitem__(self, index: int) -> Dict[str, Any]:
        example = self._examples[index]
        input_ids, attention_mask, token_type_ids = self._create_input(example['query'], example['doc'])
        if self._train:
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids,
                'label': example['label']
            }
        else:
            return {
                'query_id': example['query_id'],
                'doc_id': example['doc_id'],
                'label': example['label'],
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids
            }
