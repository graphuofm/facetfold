"""What actually governs coverage?

A first analysis over two corpora suggested coverage collapsed onto a
single curve when plotted against k/N, i.e. that answering the query
needs a constant FRACTION of the corpus. Adding a third corpus
(StackExchange, 150 facets) refuted that: at the same corpus fraction
the corpora differ by ~40 percentage points. The two-corpus agreement
was an artifact of Amazon and IMDb happening to have a similar number
of facets (33 and 28). That claim is retracted and is not made in the
paper.

This script replaces it with a model that does explain the spread.

Reference model (uniform retrieval). If the k retrieved items were a
uniform random sample of the admitted set, a facet holding n_f of the
N admitted items is covered with probability 1 - (1 - n_f/N)^k, so
    expected coverage = mean_f [1 - (1 - n_f/N)^k].
This depends on the facet size distribution and on k -- it is exactly
the "budget" part of the problem, with no semantics in it.

The gap between this reference and the measured coverage of real
similarity retrieval isolates the second cause: semantic retrieval
CONCENTRATES on a few facets, so it covers fewer facets than a random
draw of the same size would. Writing both down separates "the budget
is too small" from "retrieval looks in too few places".
"""
import glob, json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
ROOT = HERE.parent
EPS = 0.05
CORPUS_RUNS = {"results_amazon.json", "results_imdb.json",
               "results_stackexchange.json"}


def facet_sizes(corpus_dir, thr):
    """Admitted rows per facet, under the run's own predicate."""
    df = pd.read_parquet(ROOT / "corpora" / corpus_dir / "corpus.parquet",
                         columns=["facet", "filter_num", "filter_bool"])
    adm = df.filter_bool.values & (df.filter_num.values >= thr)
    return df.facet.values[adm], int(adm.sum())


def uniform_reference(sizes, N, k):
    p = sizes / N
    return float(np.mean(1.0 - np.power(1.0 - p, k)))


out = {}
for f in sorted(glob.glob(str(HERE / "results_*.json"))):
    if Path(f).name not in CORPUS_RUNS:
        continue
    r = json.load(open(f))
    name = r.get("corpus_dir") or Path(f).stem.replace("results_", "")
    thr = float(r["predicate"].rsplit(">=", 1)[1])
    fac, N = facet_sizes(name, thr)
    sizes = pd.Series(fac).value_counts().values.astype(float)
    rows = sorted([x for x in r["results"] if x["eps"] == EPS], key=lambda x: x["k"])
    pts = []
    for x in rows:
        ref = uniform_reference(sizes, N, x["k"])
        pts.append(dict(k=x["k"], frac=x["k"] / N,
                        coverage=x["coverage_mean"],
                        uniform_reference=ref,
                        concentration_gap_pp=100 * (ref - x["coverage_mean"]),
                        top1=x["top1_agree_separable"],
                        spearman=x["spearman_all"]))
    out[name] = dict(
        N=N, n_facets=int(len(sizes)),
        facet_size_min=int(sizes.min()), facet_size_median=int(np.median(sizes)),
        facet_size_max=int(sizes.max()),
        facet_skew=float(sizes.max() / sizes.min()),
        gt_spread_median=r["separability"]["gt_spread_median"],
        frac_separable=r["separability"]["frac_queries_separable"],
        points=pts)

res = dict(
    eps=EPS,
    retracted_claim=("coverage collapses onto one curve against k/N -- "
                     "refuted by the third corpus; see module docstring"),
    reference_model="expected coverage under uniform random retrieval: "
                    "mean_f [1 - (1 - n_f/N)^k]",
    corpora=out,
    finding=("Coverage is governed by two separable causes. (1) Budget: "
             "even uniform random retrieval of k items misses facets "
             "whose share of the corpus is below ~1/k, so corpora with "
             "many small facets are harder at any k. (2) Concentration: "
             "similarity retrieval covers FEWER facets than a uniform "
             "draw of the same size, because semantically similar items "
             "cluster into a few facets. The concentration gap is "
             "reported per corpus and is the part a bigger k does not "
             "fix cheaply."),
)
(HERE / "scaling.json").write_text(json.dumps(res, indent=2))

print(f"{'corpus':<16}{'N':>10}{'facets':>8}{'min':>8}{'skew':>8}")
for n, c in out.items():
    print(f"{n:<16}{c['N']:>10,}{c['n_facets']:>8}{c['facet_size_min']:>8}"
          f"{c['facet_skew']:>8.0f}x")
print(f"\n{'corpus':<16}{'k':>7}{'measured':>10}{'uniform ref':>13}{'concentration gap':>19}")
for n, c in out.items():
    for p in c["points"]:
        print(f"{n:<16}{p['k']:>7}{p['coverage']:>9.1%}{p['uniform_reference']:>12.1%}"
              f"{p['concentration_gap_pp']:>16.1f} pp")
print(f"\nwrote {HERE/'scaling.json'}")
