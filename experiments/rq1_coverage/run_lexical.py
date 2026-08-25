"""Does facet concentration survive a change of retrieval geometry?

Section 7 shows that similarity retrieval covers far fewer facets than
a uniform draw of the same size, and tests that this is not an artifact
of one dense encoder by repeating it under a second. Two dense encoders
share a geometry, though, so the finding could still be a property of
embedding space rather than of retrieval.

This repeats the coverage measurement under BM25, a lexical retriever
with no embedding geometry at all, and under a hybrid that fuses the
two rankings. If concentration persists across a dense, a lexical and a
hybrid retriever, the claim generalises past the representation.

Fusion is reciprocal rank fusion, the standard parameter-free choice,
so the hybrid introduces no tuned weight.

The uniform-retrieval reference is computed from the facet size
distribution alone and is therefore the same for every retriever, which
is what makes the three comparable.
"""
import argparse, json, re, time
from pathlib import Path

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
np.seterr(invalid="ignore", divide="ignore")

ap = argparse.ArgumentParser()
ap.add_argument("--corpus", default="imdb")
ap.add_argument("--n-queries", type=int, default=100)
ap.add_argument("--k", type=int, nargs="+", default=[10, 100, 1000, 10000])
ap.add_argument("--rrf-k", type=int, default=60)
ap.add_argument("--max-rows", type=int, default=0,
                help="subsample the admitted set; BM25 indexing is the "
                     "bottleneck and 0 means use all of it")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

CORP = ROOT / "corpora" / a.corpus
df = pd.read_parquet(CORP / "corpus.parquet")
emb = np.load(CORP / "emb.npy").astype(np.float32)
thr = {"amazon": 2015.0, "imdb": 2000.0}.get(a.corpus, 3.0)
adm = df.filter_num.values >= thr
pos = np.flatnonzero(adm)
if a.max_rows and len(pos) > a.max_rows:
    rs = np.random.RandomState(a.seed)
    pos = np.sort(rs.choice(pos, size=a.max_rows, replace=False))
codes, facets = pd.factorize(df.facet.values[pos])
texts = df.text.values[pos]
E = emb[pos]
E = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
G, N = len(facets), len(pos)
sizes = np.bincount(codes, minlength=G)
print(f"{a.corpus}: {N:,} admitted, {G} facets", flush=True)

TOK = re.compile(r"[a-z0-9]+")
tok = lambda s: TOK.findall(s.lower())
t0 = time.perf_counter()
bm25 = BM25Okapi([tok(t) for t in texts])
print(f"  BM25 index built in {time.perf_counter()-t0:.0f}s", flush=True)

rng = np.random.RandomState(a.seed)
titles = pd.Series(texts).str.split(".").str[0].str.strip()
ok = titles.str.split().str.len().between(4, 20) & ~titles.duplicated()
qsel = rng.choice(np.flatnonzero(ok.values), size=a.n_queries, replace=False)

# uniform-retrieval reference: identical for every retriever
p = sizes / N
ref = {k: float(np.mean(1.0 - (1.0 - p) ** k)) for k in a.k}


def cov(order, k):
    top = order[:k]
    return len(set(codes[top])) / G


rows = []
for qn, qi in enumerate(qsel):
    q = tok(str(texts[qi]))
    s_lex = bm25.get_scores(q)
    s_den = E @ E[qi]
    r_lex = np.argsort(-s_lex, kind="stable")
    r_den = np.argsort(-s_den, kind="stable")
    # reciprocal rank fusion over the two orderings
    rr = np.zeros(N)
    rr[r_lex] += 1.0 / (a.rrf_k + 1 + np.arange(N))
    rr[r_den] += 1.0 / (a.rrf_k + 1 + np.arange(N))
    r_hyb = np.argsort(-rr, kind="stable")
    for k in a.k:
        kk = min(k, N)
        for name, order in (("dense", r_den), ("bm25", r_lex), ("hybrid", r_hyb)):
            rows.append(dict(query_i=qn, k=k, retriever=name,
                             coverage=cov(order, kk)))
    if (qn + 1) % 25 == 0:
        print(f"  {qn+1}/{len(qsel)}", flush=True)

R = pd.DataFrame(rows)
agg = R.groupby(["k", "retriever"]).coverage.mean().reset_index()
agg["uniform_reference"] = agg.k.map(ref)
agg["gap_pp"] = (agg.uniform_reference - agg.coverage) * 100
summary = dict(corpus=a.corpus, n_admitted=N, n_facets=G,
               n_queries=int(len(qsel)), rrf_k=a.rrf_k,
               note="coverage under a dense encoder, BM25, and their "
                    "reciprocal-rank fusion, against the same "
                    "uniform-retrieval reference. gap_pp is the "
                    "concentration cost in percentage points.",
               uniform_reference=ref,
               results=agg.to_dict(orient="records"))
(OUT / f"lexical_{a.corpus}.json").write_text(json.dumps(summary, indent=2))
R.to_parquet(OUT / f"lexical_per_query_{a.corpus}.parquet", index=False)
pd.set_option("display.width", 200)
print("\n" + agg.to_string(index=False))
