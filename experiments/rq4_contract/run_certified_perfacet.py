"""Can the certificate be paid for before the scan, not after it?

Section 5 certifies a truncation only after the pass has scored every
admitted row, so the guarantee is a correctness result and buys no
time. Section 5.3 names the way out: if a candidate generator returns
rows in score order together with a threshold on everything it did not
return, the omitted mass is bounded BEFORE the omitted rows are read.

This measures that plan, using the retrieval Section 7 already found to
be the strong one. For each facet independently, take its b best-scoring
rows; let tau_g be the b-th score, so every unread row of the facet
scores at most tau_g and

    omitted mass  <=  (n_g - b) * exp((tau_g - m_g) / eps)

with n_g from the group statistics and m_g the facet's best score, both
known without touching the tail. That gives a pre-execution delta, hence
a pre-execution bound delta * (v_max - v_min) over the facet's range.

Reported per query and budget: the smallest per-facet b whose
PRE-EXECUTION bound meets the declared budget, the total rows that plan
reads as a fraction of the admitted set, and -- as the reference the
saving is measured against -- the smallest b whose bound computed from
the TRUE omitted mass would have sufficed. The gap between the two is
what the tail bound costs in conservatism; the second number is what
Section 5 certifies today, after reading everything.

A facet that cannot be certified at any b below its own size is
reported as declined rather than silently truncated, exactly as the
planner does.
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
np.seterr(invalid="ignore", divide="ignore", over="ignore")

ap = argparse.ArgumentParser()
ap.add_argument("--corpus", default="imdb")
ap.add_argument("--n-queries", type=int, default=100)
ap.add_argument("--eps", type=float, nargs="+", default=[0.02, 0.05, 0.5])
ap.add_argument("--budget-fracs", type=float, nargs="+",
                default=[0.10, 0.05, 0.01, 0.005])
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

CORP = ROOT / "corpora" / a.corpus
df = pd.read_parquet(CORP / "corpus.parquet")
emb = np.load(CORP / "emb.npy").astype(np.float64)
thr = {"amazon": 2015.0, "imdb": 2000.0}.get(a.corpus, 3.0)
adm = df.filter_num.values >= thr
codes, facets = pd.factorize(df.facet.values[adm])
vals = df.value.values[adm].astype(np.float64)
E = emb[adm]
G, N = len(facets), int(adm.sum())
rows_of = [np.flatnonzero(codes == g) for g in range(G)]
g_lo = np.array([vals[r].min() if len(r) else np.nan for r in rows_of])
g_hi = np.array([vals[r].max() if len(r) else np.nan for r in rows_of])
g_rng = g_hi - g_lo
col_range = float(vals.max() - vals.min())
budgets = [f * col_range for f in a.budget_fracs]
print(f"{a.corpus}: {N:,} admitted, {G} facets, value range {col_range:g}",
      flush=True)

rng = np.random.RandomState(a.seed)
titles = df.text.str.split(".").str[0].str.strip()
ok = titles.str.split().str.len().between(4, 20) & ~titles.duplicated() & adm
qidx = rng.choice(np.flatnonzero(ok.values), size=a.n_queries, replace=False)

recs = []
t0 = time.time()
for qn, qi in enumerate(qidx):
    s = E @ emb[qi]
    for eps in a.eps:
        for bf, beta in zip(a.budget_fracs, budgets):
            read_pre = read_true = 0
            dec_pre = dec_true = 0
            live = 0
            for g in range(G):
                idx = rows_of[g]
                n_g = idx.size
                if n_g == 0:
                    continue
                live += 1
                sg = np.sort(s[idx])[::-1]
                m_g = sg[0]
                w = np.exp((sg - m_g) / eps)
                cw = np.cumsum(w)
                z = cw[-1]
                rem = n_g - np.arange(1, n_g + 1)          # rows unread at b

                # (1) PRE-EXECUTION: the tail is bounded by its
                #     threshold, never read. tau_g = sg[b-1].
                tail_hat = rem * np.exp((sg - m_g) / eps)
                d_pre = tail_hat / (cw + tail_hat)
                hit = np.flatnonzero(d_pre * g_rng[g] <= beta)
                if hit.size:
                    read_pre += int(hit[0]) + 1
                else:
                    read_pre += n_g
                    dec_pre += 1

                # (2) what Section 5 certifies today, from the TRUE
                #     omitted mass, i.e. after reading everything
                d_true = 1.0 - cw / z
                hit = np.flatnonzero(d_true * g_rng[g] <= beta)
                if hit.size:
                    read_true += int(hit[0]) + 1
                else:
                    read_true += n_g
                    dec_true += 1

            recs.append(dict(
                corpus=a.corpus, query_i=qn, eps=eps, budget_frac=bf,
                budget=beta, n_live_facets=live,
                rows_pre=read_pre, frac_pre=read_pre / N,
                declined_pre=dec_pre / max(live, 1),
                rows_true=read_true, frac_true=read_true / N,
                declined_true=dec_true / max(live, 1),
                conservatism=read_pre / max(read_true, 1)))
    if (qn + 1) % 20 == 0:
        print(f"  {qn+1}/{len(qidx)}  ({time.time()-t0:.0f}s)", flush=True)

R = pd.DataFrame(recs)
agg = (R.groupby(["eps", "budget_frac"])
        .agg(budget=("budget", "first"),
             frac_pre_median=("frac_pre", "median"),
             frac_true_median=("frac_true", "median"),
             conservatism_median=("conservatism", "median"),
             declined_pre=("declined_pre", "mean"),
             declined_true=("declined_true", "mean"))
        .reset_index())
agg["speedup_if_indexed"] = 1.0 / agg.frac_pre_median
summary = dict(
    corpus=a.corpus, n_admitted=N, n_facets=G, n_queries=int(len(qidx)),
    value_range=col_range,
    note="frac_pre is the share of admitted rows a per-facet candidate "
         "generator would have to return for the PRE-EXECUTION tail "
         "bound to certify the budget without reading the rest; "
         "frac_true is what suffices when the true omitted mass is "
         "known, which today costs a full pass. speedup_if_indexed is "
         "1/frac_pre, the scan reduction such a plan would achieve, "
         "before index overhead.",
    results=agg.to_dict(orient="records"))
(OUT / f"certified_perfacet_{a.corpus}.json").write_text(json.dumps(summary, indent=2))
R.to_parquet(OUT / f"certified_perfacet_per_query_{a.corpus}.parquet", index=False)
pd.set_option("display.width", 220)
print("\n" + agg.to_string(index=False))
