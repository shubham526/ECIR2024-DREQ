from pyserini.search import LuceneSearcher
from pyserini.index.lucene import IndexReader
import numpy as np
import json
import argparse
import sys
from math import log
from tqdm import tqdm


def divide_queries(queries, k):
    # Assuming that you have indexed your documents using Pyserini and have an instance of the searcher
    searcher = LuceneSearcher.from_prebuilt_index('robust04')
    index_reader = IndexReader.from_prebuilt_index('robust04')

    def get_avg_idf(query):
        terms = query.split()
        idf_scores = []
        for term in terms:
            df, cf = index_reader.get_term_counts(term)  # get document frequency (df)
            if df > 0:  # Ensure we're not dividing by zero
                idf = log(index_reader.stats()['documents'] / df)  # calculate IDF
                idf_scores.append(idf)
        return 1 / max(idf_scores) if idf_scores else 0  # return reciprocal of max IDF

    def get_wig_score(query):
        hits = searcher.search(query, k)
        reciprocal_max_idf = get_avg_idf(query)
        rsvs = np.array([hit.score for hit in hits])
        wig = np.sum(rsvs - reciprocal_max_idf)
        return wig / (len(query.split()) * len(rsvs))

    # For each query, compute WIG scores
    wig_scores = {query: get_wig_score(query) for query in tqdm(queries)}

    # Sort queries based on WIG scores in descending order
    sorted_queries = sorted(wig_scores.items(), key=lambda item: item[1], reverse=True)

    # Divide queries into three difficulty levels: easy, medium, hard
    num_queries = len(sorted_queries)
    easy_queries = [query for query, score in sorted_queries[:num_queries // 3]]
    medium_queries = [query for query, score in sorted_queries[num_queries // 3:2 * num_queries // 3]]
    hard_queries = [query for query, score in sorted_queries[2 * num_queries // 3:]]

    return {
        'easy': easy_queries,
        'medium': medium_queries,
        'hard': hard_queries
    }


def load_queries(queries_file):
    with open(queries_file, 'r') as f:
        return dict({
            (line.strip().split('\t')[0], line.strip().split('\t')[1])
            for line in f
        })


def main():
    """
    Main method to run code.
    """
    parser = argparse.ArgumentParser("Divide the queries into easy, medium, hard using QPP.")
    parser.add_argument("--queries", help="TSV file containing queries.", required=True)
    parser.add_argument("--k", help="Number of documents to retrieve for QPP. Default=100.", default=100, type=int)
    parser.add_argument("--save", help="Output run file (re-ranked).", required=True)
    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])

    print('Loading queries...')
    queries = load_queries(queries_file=args.queries)
    print('[Done].')

    print('Dividing queries...')
    res = divide_queries(queries=queries, k=args.k)
    with open(args.save, 'w') as f:
        json.dump(res, f, indent=4)
    print('[Done].')


if __name__ == '__main__':
    main()
