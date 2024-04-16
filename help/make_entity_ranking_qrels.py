import json
import sys
import argparse
from typing import List, Dict, Set
import itertools
import collections
from tqdm import tqdm


def create_qrels(
        qrels: Dict[str, Dict[str, int]],
        docs: Dict[str, List[str]],
):
    data: List[str] = []

    for query_id, doc_dict in qrels.items():
        pos_docs = [doc_id for doc_id in doc_dict if doc_dict[doc_id] >= 1]
        neg_docs = [doc_id for doc_id in doc_dict if doc_dict[doc_id] < 1]
        pos_entities = set(itertools.chain.from_iterable([docs[doc_id] for doc_id in pos_docs if doc_id in docs]))
        neg_entities = set(itertools.chain.from_iterable([docs[doc_id] for doc_id in neg_docs if doc_id in docs]))
        # There seems to be some entities that are both pos and neg!
        # Don't know what to do with them; I just remove them
        common = list(set(pos_entities).intersection(neg_entities))
        pos_entities = set([e_id for e_id in pos_entities if e_id not in common])
        neg_entities = set([e_id for e_id in neg_entities if e_id not in common])
        create_qrels_file_strings(query_id=query_id, entities=pos_entities, data=data, label='1')
        create_qrels_file_strings(query_id=query_id, entities=neg_entities, data=data, label='0')

    return data


def create_qrels_file_strings(query_id: str, entities: Set[str], data: List[str], label: str) -> None:
    for entity_id in entities:
        data.append(query_id + ' Q0 ' + str(entity_id) + ' ' + label)


def write_to_file(data: List[str], save: str):
    with open(save, 'w') as f:
        for d in data:
            f.write('%s\n' % d)


def read_qrels(file: str):
    qrels = collections.defaultdict(dict)

    with open(file, 'r') as f:
        for line in f:
            query_id, _, object_id, relevance = line.strip().split()
            assert object_id not in qrels[query_id]
            qrels[query_id][object_id] = int(relevance)

    return qrels


def read_docs(file: str) -> Dict[str, List[str]]:
    docs: Dict[str, List[str]] = {}
    with open(file, 'r') as f:
        for line in tqdm(f):
            d = json.loads(line)
            docs[d['doc_id']] = d['entities']
    return docs


def main():
    parser = argparse.ArgumentParser("Create an entity ground truth file from document ground truth file.")
    parser.add_argument("--qrels", help="Document ground truth file.", required=True, type=str)
    parser.add_argument("--docs", help='Corpus file.', type=str, required=True)
    parser.add_argument("--save", help='File to save.', type=str, required=True)
    args = parser.parse_args(args=None if sys.argv[1:] else ['--random'])

    print('Reading document qrels...')
    qrels = read_qrels(args.qrels)
    print('[Done].')

    print('Reading document corpus...')
    docs = read_docs(args.docs)
    print('[Done].')

    print('Creating entity qrels file...')
    entity_qrels = create_qrels(qrels=qrels, docs=docs)
    print('[Done].')

    print('Writing to file...')
    write_to_file(entity_qrels, args.save)
    print('[Done].')


if __name__ == '__main__':
    main()
