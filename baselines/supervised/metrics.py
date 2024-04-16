import pytrec_eval


def get_metric(qrels: str, run: str, metric: str = 'map') -> float:

    # Read the qrels file
    with open(qrels, 'r') as f:
        qrels_dict = pytrec_eval.parse_qrel(f)

    # Read the run file
    with open(run, 'r') as f:
        run_dict = pytrec_eval.parse_run(f)

    # Evaluate
    evaluator = pytrec_eval.RelevanceEvaluator(qrels_dict, pytrec_eval.supported_measures)
    results = evaluator.evaluate(run_dict)
    mes = {}
    for _, query_measures in sorted(results.items()):
        for measure, value in sorted(query_measures.items()):
            mes[measure] = pytrec_eval.compute_aggregated_measure(
                measure,
                [query_measures[measure] for query_measures in results.values()]
            )
    return mes[metric]


