import json
import sys
from tqdm import tqdm
import argparse
from typing import List, Dict, Any


def read_folds(fold_file: str) -> Dict[str, Dict[str, List[str]]]:
    with open(fold_file, 'r') as f:
        return json.load(f)


def read_data(data: str, total: int):
    with open(data, 'r') as f:
        return [json.loads(line) for line in tqdm(f, total=total)]


def get_queries(fold_number: str, data: Dict[str, Dict[str, List[str]]], train: bool) -> List[str]:
    return data[fold_number]['training'] if train else data[fold_number]['testing']


def write_to_file(file_path: str, data: List[str]) -> None:
    with open(file_path, 'a') as f:
        for d in data:
            f.write('%s\n' % d)


def create_fold_data(
        fold_data_dict: Dict[str, Dict[str, List[str]]],
        data: List[Dict[str, Any]],
        save_dir: str,
        save_file: str,
        train: bool,
) -> None:
    for fold_num in range(5):
        # Get train and test queries for this fold
        fold_queries: List[str] = get_queries(fold_number=str(fold_num), data=fold_data_dict, train=train)

        # Divide data into train and test
        fold_data: List[str] = [json.dumps(d) for d in data if d['query_id'] in fold_queries]

        # Save
        save_file_path: str = save_dir + '/' + 'fold-' + str(fold_num) + '/' + save_file
        write_to_file(file_path=save_file_path, data=fold_data)

        print('Done Fold-{}'.format(fold_num))


def main():
    parser = argparse.ArgumentParser("Split data by fold for k-fold CV.")
    parser.add_argument("--folds", help='Path to file containing the fold queries.', required=True)
    parser.add_argument("--data", help='Path to data file.', required=True)
    parser.add_argument("--save", help='Path to directory where data will be saved.', required=True)
    parser.add_argument("--out", help='Name of file to save.', required=True)
    parser.add_argument("--total", help='Total lines in data file.', required=True, type=int)
    parser.add_argument('--train', help='Whether or not this is train data. Default: False.', action='store_true')
    args = parser.parse_args(args=None if sys.argv[1:] else ['--random'])

    print('Loading fold queries..')
    fold_query_dict: Dict[str, Dict[str, List[str]]] = read_folds(args.folds)
    print('[Done].')

    print('Loading data...')
    data: List[Dict[str, Any]] = read_data(args.data, args.total)
    print('[Done].')

    if args.train:
        print('Creating fold-wise train data...')
    else:
        print('Creating fold-wise test data...')
    create_fold_data(
        fold_data_dict=fold_query_dict,
        data=data,
        save_dir=args.save,
        save_file=args.out,
        train=args.train,
    )
    print('[Done].')


if __name__ == '__main__':
    main()
