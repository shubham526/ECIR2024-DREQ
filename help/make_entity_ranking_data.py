import json
import sys
from tqdm import tqdm
import argparse
from typing import List, Dict, Tuple


def read_entity_data_file(file: str) -> Dict[str, str]:
    docs: Dict[str, str] = {}
    with open(file, 'r') as f:
        for line in tqdm(f):
            d = json.loads(line)
            docs[d['id']] = d['contents'].strip()
    return docs


def read_qrels(file: str) -> Dict[str, Tuple[List[str], List[str]]]:
    qrels: Dict[str, Tuple[List[str], List[str]]] = {}
    with open(file, 'r') as f:
        for line in f:
            line_parts = line.split()
            query_id, doc_id, relevance = line_parts[0], line_parts[2], int(line_parts[3])

            if query_id in qrels:
                if relevance >= 1:
                    qrels[query_id][0].append(doc_id)
                else:
                    qrels[query_id][1].append(doc_id)
            else:
                if relevance >= 1:
                    qrels[query_id] = ([doc_id], [])
                else:
                    qrels[query_id] = ([], [doc_id])

    return qrels


def read_queries(file: str):
    with open(file, 'r') as f:
        return dict({
            (line.strip().split('\t')[0], line.strip().split('\t')[1])
            for line in f
        })


def to_pointwise_data_string(query_id, query, entities, desc, label, data):
    for entity_id in entities:
        if entity_id in desc:
            data.append(json.dumps({
                'query_id': query_id,
                'query': query,
                'doc': desc[entity_id],
                'doc_id': entity_id,
                'label': label,
            }))


def write_to_file(data: List[str], save: str):
    with open(save, 'w') as f:
        for d in data:
            f.write('%s\n' % d)


def to_data(query_id, query, pos_entities, neg_entities, desc, data):
    k = min(len(pos_entities), len(neg_entities))
    pos_entities = list(pos_entities)[:k]
    neg_entities = list(neg_entities)[:k]
    to_pointwise_data_string(query_id=query_id, query=query, entities=pos_entities, desc=desc, label=1, data=data)
    to_pointwise_data_string(query_id=query_id, query=query, entities=neg_entities, desc=desc, label=0, data=data)


def create_data(desc, qrels, queries):
    data: List[str] = []

    for query_id in tqdm(queries, total=len(queries)):
        if query_id in qrels:
            to_data(
                query_id=query_id,
                query=queries[query_id],
                pos_entities=qrels[query_id][0],
                neg_entities=qrels[query_id][1],
                desc=desc,
                data=data
            )

    return data


def main():
    parser = argparse.ArgumentParser("Create a train/test data.")
    parser.add_argument("--queries", help="Queries file.", required=True, type=str)
    parser.add_argument("--qrels", help="Entity ground truth file.", required=True, type=str)
    parser.add_argument("--save", help="Output directory.", required=True, type=str)
    parser.add_argument("--desc", help='File containing entity description.', required=True, type=str)
    args = parser.parse_args(args=None if sys.argv[1:] else [' --random'])

    print('Reading description file....')
    desc: Dict[str, str] = read_entity_data_file(args.desc)
    print('[Done].')

    print('Reading entity ground truth file...')
    qrels: Dict[str, Tuple[List[str], List[str]]] = read_qrels(args.qrels)
    print('[Done].')

    print('Reading queries file...')
    queries = read_queries(args.queries)
    print('[Done].')

    print('Creating data...')
    data = create_data(
        desc=desc,
        qrels=qrels,
        queries=queries
    )
    print('[Done].')

    print('Writing to file...')
    write_to_file(data, args.save)
    print('[Done].')

    print('File written to ==> {}'.format(args.save))


if __name__ == '__main__':
    main()
