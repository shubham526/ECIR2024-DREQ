import os
import numpy as np
from scipy import stats
from pytrec_eval import RelevanceEvaluator, parse_run, parse_qrel
from tabulate import tabulate
import argparse
import sys
import glob


def read_run_file(file_path):
    with open(file_path, 'r') as f:
        run_data = parse_run(f)
    return run_data


def read_qrels_file(file_path):
    with open(file_path, 'r') as f:
        qrels_data = parse_qrel(f)
    return qrels_data


def calc_standard_error(differences):
    return np.std(differences, ddof=1) / np.sqrt(len(differences))


def paired_t_test(reference_run_file, test_run_files_dir, qrels_file, eval_measure, output_file):
    test_run_files = glob.glob(os.path.join(test_run_files_dir, "*.run"))

    reference_run_data = read_run_file(reference_run_file)
    qrels_data = read_qrels_file(qrels_file)
    evaluator = RelevanceEvaluator(qrels_data, {eval_measure})

    reference_results = evaluator.evaluate(reference_run_data)
    results = []
    headers = ["Test File", "T-Statistic", "P-Value", "Standard Error", "Significance"]

    for test_run_file in test_run_files:
        try:
            test_run_data = read_run_file(test_run_file)
            test_results = evaluator.evaluate(test_run_data)

            common_query_ids = sorted(set(reference_results.keys()) & set(test_results.keys()))

            reference_values = np.array([reference_results[qid][eval_measure] for qid in common_query_ids])
            test_values = np.array([test_results[qid][eval_measure] for qid in common_query_ids])
            differences = reference_values - test_values
            standard_error = calc_standard_error(differences)

            t_stat, p_value = stats.ttest_rel(reference_values, test_values)
            significant = p_value < 0.05
            results.append([os.path.basename(test_run_file), t_stat, p_value, standard_error, significant])
        except ValueError:
            print('Error while reading run file: {}'.format(test_run_file))

    with open(output_file, 'w') as f:
        f.write(tabulate(results, headers=headers, tablefmt="grid"))


def main():
    parser = argparse.ArgumentParser("Paired-t-test between a reference run file and test files.")
    parser.add_argument("--reference", help='Path to reference run file.', required=True)
    parser.add_argument("--test", help='Path to directory containing test runs.', required=True)
    parser.add_argument("--qrels", help='Path to qrels file.', required=True)
    parser.add_argument("--eval", help='Evaluation measure to use.', required=True)
    parser.add_argument("--output", help='Path to output file for saving the results.', required=True)
    args = parser.parse_args(args=None if sys.argv[1:] else ['--random'])
    paired_t_test(
        reference_run_file=args.reference,
        test_run_files_dir=args.test,
        qrels_file=args.qrels,
        eval_measure=args.eval,
        output_file=args.output
    )


if __name__ == '__main__':
    main()
