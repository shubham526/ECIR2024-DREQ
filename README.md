# DREQ: Document Re-Ranking Using Entity-based Query Understanding

Shubham Chatterjee, Iain Mackie, Jeff Dalton. 2024. [DREQ: Document Re-Ranking Using Entity-based Query Understanding](https://arxiv.org/abs/2401.05939). In _Proceedings of the 46th European Conference on Information Retrieval (ECIR 2024)_.

This repository contains the code associated with this paper. For instructions on running it, [read the documentation](https://github.com/shubham526/ECIR2024-DREQ/wiki/DREQ:-Document-Re%E2%80%90Ranking-Using-Entity%E2%80%90based-Query-Understanding).

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
                                                      

The run file for every one of these is in the artifact packages regardless of
where the code lives, so a published figure can be checked without any of them.

## What this repository still contains

The DREQ model itself, the entity-ranking stage it shares with QDER, and the
data-preparation pipeline. 
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