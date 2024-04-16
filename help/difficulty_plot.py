import os
import json
import matplotlib as mpl
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from argparse import ArgumentParser


def fetch_values(run, metric, queries):
    lines = [line.split() for line in open(run, 'r')]
    tsv = [[line[1], line[0]] + line[2:] for line in lines]
    data = {row[0]: float(row[2]) for row in tsv if row[1] == metric and not math.isnan(float(row[2]))}
    return {key: data.get(key, 0.0) for key in queries}


def main():
    mpl.use("Agg")
    parser = ArgumentParser('Difficulty plot using QPP.')
    parser.add_argument('--save', help='Output file name', required=True)
    parser.add_argument('--metric', help='Metric for comparison.', required=True)
    parser.add_argument('--queries', help='JSON file with queries.', required=True)
    parser.add_argument('--y-label', help='Label for the y-axis.', required=True)
    parser.add_argument('--format', help='Format to save the figure. Default: PDF.', default='pdf', type=str)
    parser.add_argument(dest='runs', nargs='+')
    args = parser.parse_args()

    # Load JSON data
    with open(args.queries) as f:
        queries = json.load(f)

    datas = {
        run: fetch_values(run, args.metric, queries['easy'] + queries['medium'] + queries['hard'])
        for run in args.runs
    }

    queries_diff = {
        key: [(query, datas[args.runs[0]].get(query, 0.0)) for query in queries]
        for key, queries in queries.items()
    }
    series_dict = {key: dict() for key in queries_diff}

    for run in datas:
        data = datas[run]
        for label, queriesByD in queries_diff.items():
            series_dict[label][run] = np.average([data[key] for key, _ in queriesByD])

    df1 = pd.DataFrame(series_dict, index=args.runs)
    df2 = df1
    df2.index = [os.path.basename(label) for label in df1.index]

    df3 = df2.transpose()
    colors = ['black', 'olivedrab', 'red',
              'indigo', 'hotpink', 'darkorchid',
              'mediumspringgreen', 'green', 'teal',
              'tomato',
              'black', 'slategrey',
              'navy', 'royalblue', 'deepskyblue']

    plt.figure()
    df3.plot(kind='bar', label=args.metric, color=colors, figsize=(7, 7), width=0.75)
    leg = plt.legend(loc='best', fancybox=True)
    leg.get_frame().set_alpha(0.5)
    plt.tick_params(axis='both', which='major', labelsize=11)
    plt.xticks(rotation=0)
    plt.ylabel(args.y_label, fontsize=20)
    plt.savefig(args.save, bbox_inches='tight', dpi=2400, format=args.format)
    print('Figure saved to ==> {}'.format(args.save))


if __name__ == '__main__':
    main()
