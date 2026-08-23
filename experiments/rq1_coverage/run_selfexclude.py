"""Control: does a query's own source item inflate the finding?

The default workload takes each query from the opening sentence of a
corpus item, and that item stays in the corpus. Every query therefore
has one guaranteed near-perfect match, which could concentrate the
retrieved set around that item's facet and manufacture the coverage
result we report.

This reruns the coverage study with the source item excluded from the
admitted set, and additionally with its near duplicates excluded
(every admitted row whose similarity to the query exceeds a threshold,
which removes the item itself and any copy of it). If the coverage
trajectory is unchanged, the finding is not an artifact of the query
construction.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
np.seterr(invalid="ignore", divide="ignore")

ap = argparse.ArgumentParser()
ap.add_argument("--corpus", default="amazon")
ap.add_argument("--n-queries", type=int, default=200)
ap.add_argument("--k", type=int, nargs="+", default=[10, 100, 1000, 10000])
ap.add_argument("--eps", type=float, default=0.05)
ap.add_argument("--dup-sim", type=float, default=0.9)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

CORP = ROOT / "corpora" / a.corpus
df = pd.read_parquet(CORP / "corpus.parquet")
emb = np.load(CORP / "emb.npy").astype(np.float64)
thr = {"amazon": 2015.0, "imdb": 2000.0}.get(a.corpus, 3.0)
adm = df.filter_num.values >= thr
adm_pos = np.flatnonzero(adm)
row_of = -np.ones(len(df), dtype=np.int64)
row_of[adm_pos] = np.arange(len(adm_pos))
codes, facets = pd.factorize(df.facet.values[adm])
vals = df.value.values[adm].astype(np.float64)
E = emb[adm]
G, N = len(facets), int(adm.sum())
print(f"{a.corpus}: {N:,} admitted, {G} facets", flush=True)

rng = np.random.RandomState(a.seed)
titles = df.text.str.split(".").str[0].str.strip()
ok = titles.str.split().str.len().between(4, 20) & ~titles.duplicated()
# restrict to items the predicate admits, so that "the query's own row"
# is actually in the set being retrieved from and the control bites
ok &= adm
qidx = rng.choice(np.flatnonzero(ok.values), size=a.n_queries, replace=False)

# how similar is a query to its own source item, and to the rest?
self_sims = []
for qi in qidx:
    r = row_of[qi]
    if r >= 0:
        self_sims.append(float(E[r] @ emb[qi]))
self_sims = np.array(self_sims)
print(f"  similarity of a query to its own item: median "
      f"{np.median(self_sims):.3f}, min {self_sims.min():.3f}", flush=True)

MODES = ["keep", "drop_self", "drop_near_dupes"]
acc = {m: {k: [] for k in a.k} for m in MODES}
nrem = {m: [] for m in MODES}
for qn, qi in enumerate(qidx):
    s = E @ emb[qi]
    self_row = row_of[qi]
    for mode in MODES:
        live = np.ones(N, bool)
        if mode == "drop_self" and self_row >= 0:
            live[self_row] = False
        elif mode == "drop_near_dupes":
            live &= s < a.dup_sim
        nrem[mode].append(int(N - live.sum()))
        sl = np.where(live, s, -np.inf)
        cl = codes[live]
        present = np.zeros(G, bool); present[cl] = True
        n_live = present.sum()
        for k in a.k:
            kk = min(k, int(live.sum()) - 1)
            top = np.argpartition(-sl, kk)[:kk]
            seen = np.zeros(G, bool); seen[codes[top]] = True
            acc[mode][k].append((seen & present).sum() / max(n_live, 1))
    if (qn + 1) % 50 == 0:
        print(f"  {qn+1}/{len(qidx)}", flush=True)

res = []
for mode in MODES:
    for k in a.k:
        res.append(dict(mode=mode, k=k,
                        coverage_mean=float(np.mean(acc[mode][k])),
                        rows_removed_mean=float(np.mean(nrem[mode]))))
R = pd.DataFrame(res)
piv = R.pivot(index="k", columns="mode", values="coverage_mean")
piv["delta_self_pp"] = (piv["drop_self"] - piv["keep"]) * 100
piv["delta_dupes_pp"] = (piv["drop_near_dupes"] - piv["keep"]) * 100
summary = dict(corpus=a.corpus, n_admitted=N, n_facets=G, eps=a.eps,
               n_queries=int(len(qidx)), dup_sim_threshold=a.dup_sim,
               self_similarity_median=float(np.median(self_sims)),
               self_similarity_min=float(self_sims.min()),
               note="coverage with the query's own source item present, "
                    "removed, and with near duplicates removed",
               results=res,
               max_abs_delta_pp=float(np.abs(
                   np.r_[piv.delta_self_pp.values,
                         piv.delta_dupes_pp.values]).max()))
(OUT / f"selfexclude_{a.corpus}.json").write_text(json.dumps(summary, indent=2))
pd.set_option("display.width", 200)
print("\n" + piv.to_string())
print(f"\nlargest change in coverage: {summary['max_abs_delta_pp']:.2f} pp")
