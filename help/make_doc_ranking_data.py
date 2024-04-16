import json
import sys
import numpy as np
import argparse
import collections
from tqdm import tqdm
import gzip
from spacy_passage_chunker import SpacyPassageChunker
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, AutoModel, DistilBertModel, T5EncoderModel


class Encoder(nn.Module):
    def __init__(self, pretrained: str) -> None:
        super(Encoder, self).__init__()
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


def inverse_rank_weighting(rank):
    return 1 / rank


def get_weight(weight_method, rank, score):
    if weight_method == 'score':
        return score
    elif weight_method == 'uniform':
        return 1
    elif weight_method == 'recip-rank':
        return inverse_rank_weighting(rank)


def read_json_file(json_file):
    with open(json_file, 'r') as f:
        return json.load(f)


def load_docs(doc_file):
    docs = {}
    with open(doc_file, 'r') as f:
        for line in tqdm(f):
            d = json.loads(line)
            docs[d['doc_id']] = (d['entities'], d['text'])

    return docs


def read_qrels(qrels_file: str):
    qrels = collections.defaultdict(dict)

    with open(qrels_file, 'r') as f:
        for line in f:
            query_id, _, object_id, relevance = line.strip().split()
            assert object_id not in qrels[query_id]
            qrels[query_id][object_id] = int(relevance)

    return qrels


def read_run(run_file):
    run = collections.defaultdict(dict)

    with open(run_file, 'r') as f:
        for line in f:
            query_id = line.strip().split()[0]
            object_id = line.strip().split()[2]
            score = line.strip().split()[4]
            if object_id not in run[query_id]:
                run[query_id][object_id] = float(score)

    return run


def load_embeddings(embedding_file):
    emb = {}
    with gzip.open(embedding_file, 'r') as f:
        for line in tqdm(f, total=13032425):
            d = json.loads(line)
            emb[d['entity_id']] = d['embedding']
    return emb


def load_queries(queries_file):
    with open(queries_file, 'r') as f:
        return dict({
            (line.strip().split('\t')[0], line.strip().split('\t')[1])
            for line in f
        })


def write_to_file(data, save):
    with open(save, 'w') as f:
        for line in data:
            f.write("%s\n" % line)


def get_entity_centric_doc_embedding(doc_entities, query_entities, query_entity_embeddings, weight_method):
    embeddings = []
    for entity_id in doc_entities:
        entity_id = str(entity_id)
        if entity_id in query_entity_embeddings and entity_id in query_entities:
            entity_embedding = query_entity_embeddings[entity_id]
            if len(entity_embedding) >= 300:
                entity_embedding = entity_embedding[:300]
                entity_rank = list(query_entities).index(entity_id) + 1
                entity_score = query_entities[entity_id]
                entity_weight = get_weight(weight_method=weight_method, rank=entity_rank, score=entity_score)
                embeddings.append(entity_weight * np.array(entity_embedding))

    if not embeddings:
        return None

    return np.sum(embeddings, axis=0, dtype=np.float32).tolist()


def create_input(text, tokenizer, max_len):
    encoded_dict = tokenizer.encode_plus(
        text=text,
        add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
        max_length=max_len,  # Pad & truncate all sentences.
        padding='max_length',
        truncation=True,
        return_attention_mask=True,  # Construct attn. masks.
        return_token_type_ids=True,  # Construct token type ids
        return_tensors='pt'
    )

    return encoded_dict['input_ids'], encoded_dict['attention_mask'], encoded_dict['token_type_ids']


def get_doc_chunk_embeddings(doc_chunks, encoder, tokenizer, max_len, device):
    embeddings = []
    for chunk in doc_chunks:
        input_ids, attention_mask, token_type_ids = create_input(chunk, tokenizer, max_len)
        with torch.no_grad():
            emb = encoder(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                token_type_ids=token_type_ids.to(device)
            )
        embeddings.append(emb.squeeze().detach().cpu().tolist())
    return np.mean(np.array(embeddings), axis=0).tolist()


def get_docs(docs, qrels, query_entities, query_entity_embeddings, positive, query_docs, weight_method, chunker,
             encoder, tokenizer, max_len, device):
    d = {}

    # Iterate through all the documents in query_docs
    for doc_id in query_docs:

        # Check if the current document ID is in the docs dictionary
        if doc_id in docs:
            # Determine if the current document is a positive (relevant) document based on the qrels dictionary
            is_positive = doc_id in qrels and qrels[doc_id] >= 1

            # Process the document if it matches the desired relevance status (positive or negative)
            if is_positive == positive:
                # Compute the entity-centric document embedding for the current document
                doc_ent_emb = get_entity_centric_doc_embedding(
                    doc_entities=docs[doc_id][0],
                    query_entities=query_entities,
                    query_entity_embeddings=query_entity_embeddings,
                    weight_method=weight_method
                )
                # Tokenize document chunks
                document = docs[doc_id][1].replace('\n', ' ')
                chunker.tokenize_document(document)
                doc_chunks = chunker.chunk_document()
                doc_chunk_embeddings = get_doc_chunk_embeddings(doc_chunks, encoder, tokenizer, max_len, device)

                # Check if the document entity embedding is not None
                if doc_ent_emb is not None:
                    d[doc_id] = (doc_chunk_embeddings, doc_ent_emb)

    # Return the dictionary containing the documents (positive or negative) and their entity embeddings
    return d


def make_data_strings(query, query_id, docs, label, data):
    for doc_id in docs:
        data.append(json.dumps({
            'query': query,
            'query_id': query_id,
            'doc_id': doc_id,
            'doc_chunk_embeddings': docs[doc_id][0],
            'doc_ent_emb': docs[doc_id][1],
            'label': label
        }))


def get_query_entity_embeddings(query_entities, entity_embeddings):
    emb = {}
    for entity_id, score in query_entities.items():
        if entity_id in entity_embeddings:
            emb[entity_id] = entity_embeddings[entity_id]
    return emb


def create_data(queries, docs, doc_qrels, doc_run, entity_run, entity_embeddings,
                train, balance, weight_method, chunker, max_len, device, encoder, tokenizer):
    data = []
    for query_id, query in tqdm(queries.items(), total=len(queries)):
        if query_id in doc_run and query_id in entity_run and query_id in doc_qrels:
            query_docs = doc_run[query_id]
            query_entities = entity_run[query_id]
            query_entity_embeddings = get_query_entity_embeddings(
                query_entities=query_entities,
                entity_embeddings=entity_embeddings
            )
            qrels = doc_qrels[query_id]
            if train:
                # If this is train data then we are going to get the positive examples from the qrels file.
                # Any document that is explicitly annotated as relevant in the qrels in considered positive.
                pos_docs = get_docs(
                    docs=docs,
                    qrels=qrels,
                    query_entities=query_entities,
                    query_entity_embeddings=query_entity_embeddings,
                    positive=True,
                    query_docs=set(qrels.keys()),
                    weight_method=weight_method,
                    encoder=encoder,
                    tokenizer=tokenizer,
                    max_len=max_len,
                    chunker=chunker,
                    device=device
                )
            else:
                # If this is test data, then we are going to get the positive examples from the document run file.
                # In this case we set the `query_docs` parameter.
                # Any document that is explicitly annotated as relevant in the qrels in considered positive.
                pos_docs = get_docs(
                    docs=docs,
                    qrels=qrels,
                    query_entities=query_entities,
                    query_entity_embeddings=query_entity_embeddings,
                    positive=True,
                    query_docs=set(query_docs.keys()),
                    weight_method=weight_method,
                    encoder=encoder,
                    tokenizer=tokenizer,
                    max_len=max_len,
                    chunker=chunker,
                    device=device
                )
            # We always choose the negative examples from the run file.
            # A document is negative if either:
            #   (1) It is explicitly annotated as non-relevant in the qrels, OR
            #   (2) The document is not in the qrels AND present in the run file.
            neg_docs = get_docs(
                docs=docs,
                qrels=qrels,
                query_entities=query_entities,
                query_entity_embeddings=query_entity_embeddings,
                positive=False,
                query_docs=set(query_docs.keys()),
                weight_method=weight_method,
                encoder=encoder,
                tokenizer=tokenizer,
                max_len=max_len,
                chunker=chunker,
                device=device
            )

            if balance:
                # If this is true, then we balance the number of positive and negative examples
                n = min(len(pos_docs), len(neg_docs))
                pos_docs = dict(list(pos_docs.items())[:n])
                neg_docs = dict(list(neg_docs.items())[:n])

            make_data_strings(
                query=query,
                query_id=query_id,
                docs=pos_docs,
                label=1,
                data=data
            )
            make_data_strings(
                query=query,
                query_id=query_id,
                docs=neg_docs,
                label=0,
                data=data
            )

    return data


def main():
    parser = argparse.ArgumentParser("Make train/test data.")
    parser.add_argument("--queries", help="Queries file.", required=True, type=str)
    parser.add_argument("--docs", help="Document file.", required=True, type=str)
    parser.add_argument("--qrels", help="Document qrels.", required=True, type=str)
    parser.add_argument("--doc-run", help="Document run file.", required=True, type=str)
    parser.add_argument("--entity-run", help="Entity run file.", required=True, type=str)
    parser.add_argument("--embeddings", help="Wikipedia2Vec entity embeddings file.", required=True, type=str)
    parser.add_argument('--max-len', help='Maximum length for truncation/padding. Default: 512', default=512, type=int)
    parser.add_argument('--max-sent-len', default=10, help='Maximum sentence length per passage. Default=10', type=int)
    parser.add_argument('--stride', default=5,
                        help='Distance between each beginning sentence of passage in a document. Default=5', type=int)
    parser.add_argument('--encoder', help='Name of model (bert|distilbert|roberta|deberta|ernie|electra|conv-bert|t5)'
                                          'Default: bert.', type=str, default='bert')
    parser.add_argument('--train', help='Is this train data? Default: False.', action='store_true')
    parser.add_argument('--balance', help='Should the data be balanced? Default: False.', action='store_true')
    parser.add_argument('--weight-method', help='Type of entity weighing method (uniform|recip-rank|inv-log|log|score)',
                        default='score')
    parser.add_argument("--save", help="Output file.", required=True, type=str)
    parser.add_argument('--cuda', help='CUDA device number. Default: 0.', type=int, default=0)
    parser.add_argument('--use-cuda', help='Whether or not to use CUDA. Default: False.', action='store_true')
    args = parser.parse_args(args=None if sys.argv[1:] else ['--random'])

    if args.train:
        print('Creating train data...')
    else:
        print('Creating test data...')

    if args.balance:
        print('Balance dataset? --> True.')
    else:
        print('Balance dataset? --> False.')

    print('Entity weighing method --> {}'.format(args.weight_method))

    if args.avg:
        print('Averaging document chunk embeddings --> True')
    else:
        print('Averaging document chunk embeddings --> False')

    print('Maximum sentence length per passage = {}'.format(args.max_sent_len))
    print('Document chunking with stride = {}'.format(args.stride))

    model_map = {
        'bert': 'bert-base-uncased',
        'distilbert': 'distilbert-base-uncased',
        'roberta': 'roberta-base',
        'deberta': 'microsoft/deberta-base',
        'ernie': 'nghuyong/ernie-2.0-base-en',
        'electra': 'google/electra-small-discriminator',
        'conv-bert': 'YituTech/conv-bert-base',
        't5': 't5-base'
    }
    cuda_device = 'cuda:' + str(args.cuda)
    print('CUDA Device: {} '.format(cuda_device))
    device = torch.device(cuda_device if torch.cuda.is_available() and args.use_cuda else 'cpu')
    print('Using device: {}'.format(device))
    print('Model ==> {}'.format(args.encoder))
    pretrain = vocab = model_map[args.encoder]
    tokenizer = AutoTokenizer.from_pretrained(vocab, model_max_length=args.max_len)
    encoder = Encoder(pretrained=pretrain)
    chunker = SpacyPassageChunker(max_len=args.max_sent_len, stride=args.stride)
    encoder.to(device)
    encoder.eval()

    print('Loading queries...')
    queries = load_queries(queries_file=args.queries)
    print('[Done].')

    print('Loading documents...')
    docs = load_docs(args.docs)
    print('[Done].')

    print('Loading qrels...')
    qrels = read_qrels(args.qrels)
    print('[Done].')

    print('Loading document run...')
    doc_run = read_run(args.doc_run)
    print('[Done].')

    print('Loading entity run...')
    entity_run = read_run(args.entity_run)
    print('[Done].')

    print('Loading embeddings...')
    embeddings = load_embeddings(args.embeddings)
    print('[Done].')

    print('Creating data...')
    data = create_data(
        queries=queries,
        docs=docs,
        doc_qrels=qrels,
        doc_run=doc_run,
        entity_run=entity_run,
        entity_embeddings=embeddings,
        train=args.train,
        balance=args.balance,
        weight_method=args.weight_method,
        chunker=chunker,
        tokenizer=tokenizer,
        encoder=encoder,
        max_len=args.max_len,
        device=device
    )
    print('[Done].')

    print('Writing to file...')
    write_to_file(data, args.save)
    print('[Done].')


if __name__ == '__main__':
    main()
