# DREQ: Document Re-Ranking Using Entity-based Query Understanding

Shubham Chatterjee, Iain Mackie, Jeff Dalton. 2024. [DREQ: Document Re-Ranking Using Entity-based Query Understanding](https://arxiv.org/abs/2401.05939). In _Proceedings of the 46th European Conference on Information Retrieval (ECIR 2024)_.

This repository contains the code associated with this paper. 

---

> ### ⚠️ Before using these results
>
> **Corrections.** Five cells in the published tables do not match the runs that
> produced them, and both papers' results rest on a form of entity supervision
> whose consequences were not stated. All of it is set out on the
> [Errata](https://github.com/shubham526/SIGIR2025-QDER/wiki/Errata) page, which
> covers this paper and QDER together.
>
> **Verify anything for yourself.** The run file behind every published row is
> released, together with a script that scores each one against the value
> printed in the paper. No GPU, no model, no Python — only `trec_eval`. See
> [Data](https://github.com/shubham526/SIGIR2025-QDER/wiki/Data).

---

## Data

All data for this paper is on the **[Data page in the QDER
wiki](https://github.com/shubham526/SIGIR2025-QDER/wiki/Data)**. Both papers
share a first stage, the same collections and the same candidate rankings, so
the data was consolidated there rather than duplicated and left to drift apart.

That page has:

| | |
|---|---|
| **Collection inputs** | qrels, queries, query entity annotations, fold splits and the candidate ranking, one bundle per collection |
| **Entity-linked corpora** | entity annotations for each collection, in JSONL |
| **Entity resources** | entity metadata and Wikipedia2Vec embeddings |
| **Artifact packages** | one archive per collection: the run behind every published row, the shared entity-ranking stage, the per-fold configurations, and a `verify.sh` that reproduces every figure |

This paper reports TREC Robust 2004, CODEC, TREC News 2021 and TREC Core 2018.
The archives for those four contain a `dreq/` directory with this paper's runs.
(The fifth, TREC CAR, is QDER only.)

```bash
tar xzf robust04.tar.gz && cd robust04
./verify.sh /path/to/docs.graded.qrels /path/to/title.BM25_RM3_TUNED.run
```

**Note on scoring.** CODEC is scored with `trec_eval -Jc`; the other three use
`-c`. The flag is applied uniformly within each collection, so no comparison
inside a table is affected, but it must be used to reproduce a figure.

## Baselines

The neural re-ranking baselines are no longer in this repository. They were
shared with QDER — the released runs are byte-identical between the two papers
— so keeping one copy in each invited them to diverge.

**[`ir_baselines`](https://github.com/shubham526/ir_baselines), tag
[`v1.0-as-published`](https://github.com/shubham526/ir_baselines/releases/tag/v1.0-as-published)**

That tag is the code that produced the released runs. It contains only fixes
that cannot change a score; anything that would is listed in its
`docs/known-issues.md` and fixed in a later release. Use the tag to reproduce a
published row, and the later release to build on the code.

Not everything in the tables came from there:

| Row | Where it came from                                                                                |
|---|---------------------------------------------------------------------------------------------------|
| RoBERTa, DeBERTa, ELECTRA, ConvBERT, ERNIE, RankT5 | `ir_baselines`, `v1.0-as-published`                                                               |
| KNRM, ConvKNRM, EDRM | [OpenMatch](https://github.com/thunlp/OpenMatch/tree/master/v1), `master` branch, `v1/` directory |
| CEDR, PARADE, BERT-MaxP, EQFE, ColBERT v2, SPLADE, ANCE-MaxP | public runs or the original authors' implementations                                              |
| MaxSimCos | `ir_baselines`, under `ir_baselines.entity`                                                       |

The run file for every one of these is in the artifact packages regardless of
where the code lives, so a published figure can be checked without any of them.

## Running the pipeline

DREQ has two stages. The first ranks entities for a query; the second uses that
ranking to weight entity embeddings when re-ranking documents. The second
depends on the output of the first, so they run in order.

Everything below assumes the collection inputs and the entity-linked corpus
from the [Data](https://github.com/shubham526/SIGIR2025-QDER/wiki/Data) page.

### 0. Inputs

| | |
|---|---|
| `corpus.entities.jsonl` | the entity-linked corpus: `{"doc_id", "entities"}` per line |
| `docs.graded.qrels`, `docs.binary.qrels` | relevance judgments; `rank-lips` needs the binary one |
| `queries.tsv`, `bm25+rm3.run` | queries and the candidate ranking to re-rank |
| `entity_metadata.jsonl.gz` | entity descriptions, for the entity ranker |
| `mmead_entities.wikipedia2vec.jsonl.gz` | entity embeddings, for the document ranker |
| `folds.json` | the 5-fold splits |

To link a corpus yourself rather than using the released one, `wat_entity_linker.py`
does it — put a gcube token in `MY_GCUBE_TOKEN` first. `make_folds.py` builds
fold splits from a queries file.

### 1. Entity ranking

**Derive entity judgments from the document judgments.** An entity is positive
if it occurs only in relevant documents, negative if only in non-relevant ones,
and discarded if it occurs in both:

```bash
python make_entity_ranking_qrels.py \
  --qrels docs.binary.qrels \
  --docs corpus.entities.jsonl \
  --save entity.qrels
```

Read [Errata §1](https://github.com/shubham526/SIGIR2025-QDER/wiki/Errata) before
building on this. The rule selects for rarity rather than topical relevance, and
what it implies for the results is set out there.

**Build the training pairs.** Each is a query and an entity description, balanced
1:1 per query:

```bash
python make_entity_ranking_data.py \
  --queries queries.tsv \
  --qrels entity.qrels \
  --desc entity_metadata.jsonl \
  --save entity_data.jsonl
```

**Split by fold, then train.** The entity ranker is a monoBERT cross-encoder over
(query, entity description) pairs, and the file above is already in the format
[`ir_baselines`](https://github.com/shubham526/ir-baselines) consumes:

```bash
python split_data_by_fold.py --folds folds.json --data entity_data.jsonl \
  --save entity_folds --out train.jsonl --total $(wc -l < entity_data.jsonl) --train
python split_data_by_fold.py --folds folds.json --data entity_data.jsonl \
  --save entity_folds --out test.jsonl  --total $(wc -l < entity_data.jsonl)

for k in 0 1 2 3 4; do
  python -m ir_baselines.train --model bert \
    --train entity_folds/fold-$k/train.jsonl \
    --dev   entity_folds/fold-$k/test.jsonl \
    --qrels entity_folds/fold-$k/test.qrels \
    --save-dir entity_model/fold-$k --use-cuda
  python -m ir_baselines.test --model bert \
    --test entity_folds/fold-$k/test.jsonl \
    --checkpoint entity_model/fold-$k/model.bin \
    --save-dir entity_runs --run fold-$k.run --use-cuda
done
cat entity_runs/fold-*.run > entity.run
```

**Check the concatenation.** A missing fold leaves the master run short, and it
still scores:

```bash
awk '{print $1}' entity.run | sort -u | wc -l    # match your topic count
```

### 2. Document ranking

**Build the training data.** This is where the entity run enters: the top entities
per query weight the entity embeddings that go into the document representation.

```bash
python make_doc_ranking_data.py \
  --queries queries.tsv --docs corpus.entities.jsonl \
  --qrels docs.binary.qrels --doc-run bm25+rm3.run \
  --entity-run entity.run \
  --embeddings mmead_entities.wikipedia2vec.jsonl.gz \
  --encoder t5 --weight-method score \
  --train --balance \
  --save doc_data.train.jsonl --use-cuda
```

Drop `--train --balance` for the test data. `--weight-method` selects how entity
rank becomes a weight (`uniform`, `recip-rank`, `inv-log`, `log`, `score`);
`score` is what the paper used. Passage chunking uses a 10-sentence window with
stride 5 (`--max-sent-len`, `--stride`).

**Split by fold and train:**

```bash
python split_data_by_fold.py --folds folds.json --data doc_data.train.jsonl \
  --save doc_folds --out train.jsonl --total $(wc -l < doc_data.train.jsonl) --train
python split_data_by_fold.py --folds folds.json --data doc_data.test.jsonl \
  --save doc_folds --out test.jsonl --total $(wc -l < doc_data.test.jsonl)

for k in 0 1 2 3 4; do
  python train.py --text-enc t5 \
    --train doc_folds/fold-$k/train.jsonl --dev doc_folds/fold-$k/test.jsonl \
    --qrels docs.graded.qrels --save-dir doc_model/fold-$k --use-cuda
  python test.py --text-enc t5 \
    --test doc_folds/fold-$k/test.jsonl \
    --checkpoint doc_model/fold-$k/model.bin \
    --save doc_runs/fold-$k.run --use-cuda
done
cat doc_runs/fold-*.run > dreq.run
```

### 3. Score

```bash
trec_eval -c -m map -m ndcg_cut.20 -m P.20 docs.graded.qrels dreq.run
```

**CODEC uses `-Jc`.** The other three collections use `-c`. Getting this wrong is
the most common reason a Table 2 figure does not reproduce.

`split_run_or_qrels_by_fold.py` splits a run or qrels file per fold.

### Analysis scripts

`qpp.py` divides queries into easy/medium/hard by query performance prediction,
`difficulty_plot.py` draws the per-difficulty comparison, and `paired-t-test.py`
runs significance tests between a reference run and a directory of others.

---

Shield: [![CC BY-SA 4.0][cc-by-sa-shield]][cc-by-sa]

All data associated with this work is licensed and released under a
[Creative Commons Attribution-ShareAlike 4.0 International License][cc-by-sa].

[![CC BY-SA 4.0][cc-by-sa-image]][cc-by-sa]

[cc-by-sa]: http://creativecommons.org/licenses/by-sa/4.0/
[cc-by-sa-image]: https://licensebuttons.net/l/by-sa/4.0/88x31.png
[cc-by-sa-shield]: https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg

## Acknowledgement

This material is based upon work supported by the Engineering and Physical Sciences Research Council (EPSRC) grant EP/V025708/1. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the EPSRC.

## Cite

```bibtex
@inproceedings{chatterjee2024dreq,
  author    = {Chatterjee, Shubham and Mackie, Iain and Dalton, Jeff},
  title     = {DREQ: Document Re-ranking Using Entity-Based Query Understanding},
  year      = {2024},
  isbn      = {978-3-031-56026-2},
  publisher = {Springer-Verlag},
  address   = {Berlin, Heidelberg},
  url       = {https://doi.org/10.1007/978-3-031-56027-9_13},
  doi       = {10.1007/978-3-031-56027-9_13},
  booktitle = {Advances in Information Retrieval: 46th European Conference on Information Retrieval, ECIR 2024, Glasgow, UK, March 24–28, 2024, Proceedings, Part I},
  pages     = {210–229},
  numpages  = {20},
  location  = {Glasgow, United Kingdom}
}
```

## Contact

Shubham Chatterjee — <shubham.chatterjee@mst.edu>