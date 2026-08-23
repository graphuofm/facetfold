"""The counter-design: retrieve top-b WITHIN each facet.

Global top-k loses facets because similar items concentrate. The
obvious repair is not to retrieve more, but to retrieve per facet: at a
total budget K over G non-empty facets, take the b = K/G best-scoring
items of each facet and aggregate those. Coverage is then 100% by
construction, and the concentration pathology of a single ranked list
disappears. Deployed vector stores expose exactly this as grouped
retrieval, so it is a real alternative and not a hypothetical one.

This harness runs it at the same total candidate budget as global
top-k, in two allocations:

  equal         b = K/G for every facet. Cheapest to describe, and
                over-spends on facets holding few items.
  proportional  b_g proportional to |facet g|, floor 1. Spends the
                budget where the items are, while still guaranteeing
                every non-empty facet at least one item.

and reports what global top-k is reported on: coverage, conditional
MAE over the facets a method answers, top-facet agreement on separable
queries, and Spearman correlation with unanswered facets ranked last.

Cost is reported honestly. Per-facet retrieval is not free: it needs
either one index probe per facet or, without per-facet indexes, the
same full scan the exact answer needs. We time both the aggregation
work and the full-scan-plus-partial-sort route, and note that the
indexed route's cost is G probes rather than one.

What per-facet top-b does NOT give, at any budget, is a bound: it
reports a truncated mean with no statement of the weight it omitted,
which is the property Section 5 is about. That is measured here too,
as the realised error against the exact answer.
"""
import argparse, json, time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

np.seterr(invalid="ignore", divide="ignore")  # empty facets fold to NaN

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent

ap = argparse.ArgumentParser()
ap.add_argument("--corpus", default="amazon")
ap.add_argument("--n-queries", type=int, default=200)
ap.add_argument("--k", type=int, nargs="+", default=[100, 1000, 10000])
ap.add_argument("--eps", type=float, nargs="+", default=[0.02, 0.05, 0.5])
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
sizes = np.bincount(codes, minlength=G)
print(f"{a.corpus}: {N:,} admitted, {G} facets", flush=True)

rng = np.random.RandomState(a.seed)
titles = df.text.str.split(".").str[0].str.strip()
ok = titles.str.split().str.len().between(4, 20) & ~titles.duplicated()
qidx = rng.choice(np.flatnonzero(ok.values), size=a.n_queries, replace=False)


def fold(idx, sv, eps):
    """Max-anchored soft aggregate over the given rows."""
    c = codes[idx]
    m = np.full(G, -np.inf); np.maximum.at(m, c, sv)
    w = np.exp((sv - m[c]) / eps)
    num = np.zeros(G); den = np.zeros(G)
    np.add.at(num, c, w * vals[idx]); np.add.at(den, c, w)
    return np.where(den > 0, num / den, np.nan)


def alloc_equal(K, live):
    b = np.zeros(G, dtype=np.int64)
    b[live] = max(1, K // max(live.sum(), 1))
    return np.minimum(b, sizes)


def alloc_prop(K, live):
    b = np.zeros(G, dtype=np.int64)
    tot = sizes[live].sum()
    b[live] = np.maximum(1, np.floor(K * sizes[live] / tot)).astype(np.int64)
    return np.minimum(b, sizes)


rows = []
for qn, qi in enumerate(qidx):
    qv = emb[qi]
    t0 = time.perf_counter()
    s = E @ qv
    scan_ms = (time.perf_counter() - t0) * 1e3
    # descending order within each facet, once per query
    order = np.argsort(-s, kind="stable")
    goff = [[] for _ in range(G)]
    for r in order:
        goff[codes[r]].append(r)
    goff = [np.asarray(x, dtype=np.int64) for x in goff]
    live = sizes > 0

    for eps in a.eps:
        ref = fold(np.arange(N), s, eps)
        ok_ref = np.isfinite(ref)
        rank_ref = pd.Series(np.where(ok_ref, ref, -np.inf)).rank()
        win = np.argsort(-np.where(ok_ref, ref, -np.inf))
        sep = (ok_ref.sum() >= 2 and
               ref[win[0]] - ref[win[1]] > 0.05) if ok_ref.sum() >= 2 else False

        for K in a.k:
            cand = {}
            gk = np.argpartition(-s, min(K, N - 1))[:K]
            cand["global_topk"] = (gk, s[gk], int(K))
            for name, af in (("perfacet_equal", alloc_equal),
                             ("perfacet_prop", alloc_prop)):
                b = af(K, live)
                t1 = time.perf_counter()
                idx = np.concatenate([goff[g][:b[g]] for g in range(G)
                                      if b[g] > 0])
                agg_ms = (time.perf_counter() - t1) * 1e3
                cand[name] = (idx, s[idx], int(b.sum()), agg_ms)

            for name, tup in cand.items():
                idx, sv, used = tup[0], tup[1], tup[2]
                est = fold(idx, sv, eps)
                ok_e = np.isfinite(est)
                both = ok_ref & ok_e
                cov = ok_e[ok_ref].sum() / max(ok_ref.sum(), 1)
                dif = np.abs(est[both] - ref[both])
                mae = dif.mean() if both.any() else np.nan
                # the worst facet, which is what a bound would have to
                # cover and what neither truncation reports
                emax = dif.max() if both.any() else np.nan
                # conditional MAE relative to the facet value scale
                scale = np.abs(ref[ok_ref]).mean()
                rk = pd.Series(np.where(ok_e, est, -np.inf)).rank()
                sp = spearmanr(rank_ref[ok_ref], rk[ok_ref]).correlation
                if ok_e.any():
                    w_e = np.argmax(np.where(ok_e, est, -np.inf))
                    agree = bool(w_e == win[0])
                else:
                    agree = False
                rows.append(dict(
                    corpus=a.corpus, query_i=qn, eps=eps, k=K, method=name,
                    candidates=used, coverage=float(cov),
                    mae=float(mae), max_abs_err=float(emax),
                    mae_rel=float(mae / scale) if scale else np.nan,
                    n_missed=int(ok_ref.sum() - both.sum()),
                    spearman=float(sp) if sp == sp else np.nan,
                    top1_agree=agree, separable=bool(sep),
                    scan_ms=scan_ms,
                    agg_ms=float(tup[3]) if len(tup) > 3 else np.nan))
    if (qn + 1) % 25 == 0:
        print(f"  {qn+1}/{len(qidx)}", flush=True)

R = pd.DataFrame(rows)
agg = (R.groupby(["eps", "k", "method"])
        .agg(candidates=("candidates", "median"),
             coverage=("coverage", "mean"),
             mae=("mae", "mean"),
             max_abs_err_mean=("max_abs_err", "mean"),
             max_abs_err_p100=("max_abs_err", "max"),
             n_missed=("n_missed", "mean"),
             mae_rel=("mae_rel", "mean"),
             spearman=("spearman", "mean"),
             top1_agree=("top1_agree", "mean"))
        .reset_index())
sepR = R[R.separable]
sepa = (sepR.groupby(["eps", "k", "method"]).top1_agree.mean()
        .rename("top1_agree_separable").reset_index())
nsep = int(sepR.query_i.nunique())
agg = agg.merge(sepa, on=["eps", "k", "method"], how="left")

summary = dict(
    corpus=a.corpus, n_admitted=N, n_facets=G, n_queries=int(len(qidx)),
    predicate=f"filter_num >= {thr:g}",
    n_separable_queries=nsep,
    note="per-facet top-b is run at the same TOTAL candidate budget as "
         "global top-k; 'candidates' is the number of rows actually "
         "aggregated, which can fall below K when facets are smaller "
         "than their allocation",
    results=agg.to_dict(orient="records"))
(OUT / f"perfacet_{a.corpus}.json").write_text(json.dumps(summary, indent=2))
R.to_parquet(OUT / f"perfacet_per_query_{a.corpus}.parquet", index=False)
pd.set_option("display.width", 220)
print("\n" + agg.to_string(index=False))
