import numpy as np
from typing import Dict, List
import argparse
import sys
import json
import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import operator
import gzip
from tqdm import tqdm

# Total number of documents in each collection
doc_map = {
    'robust': 528024,
    'codec': 729337,
    'core': 595037,
    'news': 728626
}


def calculate_document_score(
        query_entities: Dict[str, float],
        doc_entities: List[str],
        embeddings: Dict[str, List[float]],
        metric: str,
        method: str
):
    """Calculate the document score as the sum of cosine similarities
     between every pair of entities in the query and the document,
     multiplied by the confidence associated with each query entity.
     """
    try:
        # Pre-compute entity vectors
        query_vectors = np.array([embeddings[entity] for entity in query_entities.keys() if entity in embeddings])
        doc_vectors = np.array([embeddings[str(entity)] for entity in doc_entities if str(entity) in embeddings])

        # Compute the similarities
        if metric == 'dot':
            similarities = np.dot(query_vectors, doc_vectors.T)
        else:
            similarities = cosine_similarity(query_vectors, doc_vectors)

        if method == 'max':
            # Take the maximum score for each query entity
            max_scores = np.max(similarities, axis=1)
            # Sum up all the maximum scores to get the final score
            score = np.sum(max_scores)
        else:
            score = np.sum(similarities)

        return score
    except Exception as e:
        print(f'Exception: {e}')
        print(f"Query Vectors: {query_vectors.shape}")
        print(f"Doc Vectors: {doc_vectors.shape}")
        return 0.0


def get_query_annotations(query_annotations: str) -> Dict[str, float]:
    annotations = json.loads(query_annotations)
    res: Dict[str, float] = {}
    for ann in annotations:
        res[str(ann['entity_id'])] = ann['score']
    return res


def re_rank(
        run: Dict[str, List[str]],
        docs: Dict[str, List[str]],
        query_annotations: Dict[str, str],
        embeddings: Dict[str, List[float]],
        out_file: str,
        metric: str,
        method: str
) -> None:
    for query_id, candidate_docs in tqdm(run.items(), total=len(run)):
        ranked_docs: Dict[str, float] = rank_docs_for_query(
            candidate_docs=candidate_docs,
            docs=docs,
            query_annotations=get_query_annotations(query_annotations[query_id]),
            embeddings=embeddings,
            metric=metric,
            method=method
        )
        if not ranked_docs:
            print('Empty ranking for query: {}'.format(query_id))
        else:
            run_file_strings: List[str] = to_run_file_strings(query_id, ranked_docs, metric, method)
            write_to_file(run_file_strings, out_file)


def rank_docs_for_query(
        candidate_docs: List[str],
        docs: Dict[str, List[str]],
        query_annotations: Dict[str, float],
        embeddings: Dict[str, List[float]],
        metric: str,
        method: str
) -> Dict[str, float]:
    ranking: Dict[str, float] = dict(
        (doc_id, calculate_document_score(
            query_entities=query_annotations,
            doc_entities=docs[doc_id],
            embeddings=embeddings,
            metric=metric,
            method=method
        ))
        for doc_id in candidate_docs if doc_id in docs
    )

    return dict(sorted(ranking.items(), key=operator.itemgetter(1), reverse=True))


def to_run_file_strings(query: str, doc_ranking: Dict[str, float], metric: str, method: str) -> List[str]:
    run_file_strings: List[str] = []
    tag = metric + '_' + method
    rank: int = 1
    for entity, score in doc_ranking.items():
        if score > 0.0:
            run_file_string: str = query + ' Q0 ' + entity + ' ' + str(rank) + ' ' + str(score) + ' ' + tag
            run_file_strings.append(run_file_string)
            rank += 1

    return run_file_strings


def write_to_file(run_file_strings: List[str], run_file: str) -> None:
    with open(run_file, 'a') as f:
        for item in run_file_strings:
            f.write("%s\n" % item)


def load_embeddings(embedding_file):
    emb = {}
    with gzip.open(embedding_file, 'r') as f:
        for line in tqdm(f, total=13032425):
            d = json.loads(line)
            embedding = d['embedding'][:300]
            emb[d['entity_id']] = embedding
    return emb


def read_docs(file: str, name: str) -> Dict[str, List[str]]:
    docs: Dict[str, List[str]] = {}
    with open(file, 'r') as f:
        for line in tqdm(f, total=doc_map[name]):
            d = json.loads(line)
            docs[d['doc_id']] = d['entities']
    return docs


def read_run(run_file: str) -> Dict[str, List[str]]:
    run: Dict[str, List[str]] = {}

    with open(run_file, 'r') as f:
        for line in f:
            query_id = line.strip().split()[0]
            doc_id = line.strip().split()[2]
            if query_id in run:
                run[query_id].append(doc_id)
            else:
                run[query_id] = [doc_id]
    return run


def read_tsv(file: str) -> Dict[str, str]:
    with open(file, 'r') as f:
        return dict({
            (line.strip().split('\t')[0], line.strip().split('\t')[1])
            for line in f
        })


def main():
    """
    Main method to run code.
    """
    parser = argparse.ArgumentParser("Document ranking using pairwise similarity")
    parser.add_argument("--run", help="Document run file to re-rank.", required=True)
    parser.add_argument("--docs", help="Document corpus containing entities.", required=True)
    parser.add_argument("--annotations", help="File containing TagMe annotations for queries.", required=True)
    parser.add_argument("--embeddings", help="Entity embedding file", required=True)
    parser.add_argument("--name", help="Name of dataset (robust|core|news|codec).", required=True)
    parser.add_argument("--metric", help="Similarity metric (cos|dot). Default: cos", default='cos', type=str)
    parser.add_argument("--method", help="Score method (sum|max). Default: sum", default='sum', type=str)
    parser.add_argument("--save", help="Output run file (re-ranked).", required=True)
    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])

    print('Loading run file...')
    run: Dict[str, List[str]] = read_run(run_file=args.run)
    print('[Done].')

    print('Loading docs...')
    docs: Dict[str, List[str]] = read_docs(file=args.docs, name=args.name)
    print('[Done].')

    print('Loading query annotations...')
    query_annotations: Dict[str, str] = read_tsv(file=args.annotations)
    print('[Done].')

    print('Loading entity embeddings...')
    embeddings: Dict[str, List[float]] = load_embeddings(embedding_file=args.embeddings)
    print('[Done].')

    print("Re-Ranking run...")
    re_rank(
        run=run,
        docs=docs,
        query_annotations=query_annotations,
        embeddings=embeddings,
        out_file=args.save,
        metric=args.metric,
        method=args.method
    )
    print('[Done].')

    print('New run file written to {}'.format(args.save))


if __name__ == '__main__':
    main()
