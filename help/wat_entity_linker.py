import json
from tqdm import tqdm
import requests
from joblib import Parallel, delayed
import contextlib
import argparse
import joblib
import sys

MY_GCUBE_TOKEN = "YOUR TOKEN HERE"


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """
    Context manager to patch joblib to report into tqdm progress bar given as argument
    """

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


class WATAnnotation:
    # An entity annotated by WAT

    def __init__(self, d):
        # char offset (included)
        self.start = d['start']
        # char offset (not included)
        self.end = d['end']

        # annotation accuracy
        self.rho = d['rho']
        # spot-entity probability
        self.prior_prob = d['explanation']['prior_explanation']['entity_mention_probability']

        # annotated text
        self.spot = d['spot']

        # Wikipedia's entity info
        self.wikipedia_id = d['id']
        self.wikipedia_title = d['title']

    def json_dict(self):
        # Simple dictionary representation
        return {
            'wikipedia_title': self.wikipedia_title,
            'wikipedia_id': self.wikipedia_id,
            'start': self.start,
            'end': self.end,
            'rho': self.rho,
            'prior_prob': self.prior_prob
        }


def read_docs(doc_file):
    docs = {}
    with open(doc_file, 'r') as f:
        for line in tqdm(f, total=729337):
            d = json.loads(line)
            docs[d['doc_id']] = d['text']
    return docs


def wat_entity_linking(doc_id, text):
    try:
        # Main method, text annotation with WAT entity linking system
        wat_url = 'https://wat.d4science.org/wat/tag/tag'
        payload = [("gcube-token", MY_GCUBE_TOKEN),
                   ("text", text),
                   ("lang", 'en'),
                   ("tokenizer", "nlp4j"),
                   ('debug', 9),
                   ("method",
                    "spotter:includeUserHint=true:includeNamedEntity=true:includeNounPhrase=true,prior:k=50,filter-valid,"
                    "centroid:rescore=true,topk:k=5,voting:relatedness=lm,ranker:model=0046.model,"
                    "confidence:model=pruner-wiki.linear")]

        response = requests.get(wat_url, params=payload, timeout=30)
        annotations = [WATAnnotation(a) for a in response.json()['annotations']]
        json_list = [w.json_dict() for w in annotations]
        return json.dumps({
            'doc_id': doc_id,
            'entities': json_list
        })
    except json.decoder.JSONDecodeError:
        print('ERROR IN RESPONSE. NO ENTITIES FOUND.')
        return json.dumps({
            'doc_id': doc_id,
            'entities': []
        })
    except requests.exceptions.ConnectionError as e:
        print('Network problem occurred', e)
        return json.dumps({
            'doc_id': doc_id,
            'entities': []
        })
    except requests.exceptions.RequestException as e:
        print('HTTP Request failed', e)
        return json.dumps({
            'doc_id': doc_id,
            'entities': []
        })


def entity_link_corpus(docs, workers):
    with tqdm_joblib(tqdm(desc="Progress", total=len(docs))) as progress_bar:
        data = Parallel(n_jobs=workers, backend='multiprocessing')(
            delayed(wat_entity_linking)(doc_id, text)
            for doc_id, text in docs.items()
        )
    return data


def write_to_file(res, save):
    with open(save, 'w') as f:
        for item in res:
            f.write("%s\n" % item)


def main():
    """
    Main method to run code.
    """
    parser = argparse.ArgumentParser("Annotate documents using WAT. Annotations are stored in TSV format.")
    parser.add_argument("--docs", help="File containing documents.", required=True)
    parser.add_argument("--save", help="Output file containing query annotations.", required=True)
    parser.add_argument("--workers", help="Number of threads to use. Default=4", default=4, type=int)
    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])

    print('Loading corpus...')
    docs = read_docs(doc_file=args.docs)
    print('[Done].')

    print('Entity linking corpus using WAT...')
    print('Using {} threads'.format(args.workers))
    res = entity_link_corpus(docs=docs, workers=args.workers)
    print('[Done].')

    print('Writing to file...')
    write_to_file(res=res, save=args.save)
    print('[Done].')


if __name__ == '__main__':
    main()
