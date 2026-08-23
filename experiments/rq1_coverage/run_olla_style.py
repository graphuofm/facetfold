"""The closest published rival's sampling strategy, under our statistic.

OLLA (Hui et al., DASFAA 2026) is the nearest published system: online
aggregation over text, made to converge quickly by SEMANTIC STRATIFIED
SAMPLING. Its own target is an unweighted aggregate (COUNT/SUM/AVG)
over records an LLM predicate admits, so it does not compute the
relevance-weighted per-facet mean this paper is about, and running its
pipeline unmodified would not produce a comparable number.

What is comparable, and what the paper needs, is its ESTIMATOR applied
to our target. So this reproduces the strategy as described:

  * embed, then partition the embedding space into strata with K-means;
    the paper's rule of thumb is H0 = K log N strata for K categories
    over N records, capped at 2K log N;
  * sample uniformly within strata;
  * report a progressive estimate with a confidence interval
    eps_n = (z^2 V_n / n)^(1/2).

Two deliberate handicaps are removed, so the baseline is the strongest
version of itself rather than a straw man:

  1. OLLA must infer group membership with an LLM because its groups
     are semantic. Here the facet is a relational column, so we hand it
     the true facet labels for free.
  2. Equation 1 is a plain sample-mean interval. Our target is a RATIO
     (sum w v / sum w), so we give it the textbook stratified ratio
     estimator with a linearised variance and a finite-population
     correction, which is strictly tighter than Equation 1 applied
     naively.
  3. The rule H0 = K log N puts more strata than a per-facet budget can
     fill: at H = K log N most (facet, stratum) cells receive a single
     unit, and a stratified variance is not estimable from one unit.
     Rather than let the interval collapse to zero -- which would make
     the baseline look absurdly overconfident for a reason that is an
     artefact of our budget, not of the method -- singleton cells are
     COLLAPSED into one pooled stratum per facet, the standard remedy
     for a one-unit-per-stratum design.

Reported per (query, facet) at matched sample budgets: the realised
error, whether the nominal 95% interval covers the exact answer, the
interval's half-width, and facet coverage. The question is not whether
sampling converges -- it does -- but whether it converges at a budget
comparable to the one truncation is given, and whether its interval can
be believed at the sharp temperatures where the answer is concentrated.
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
np.seterr(invalid="ignore", divide="ignore")

ap = argparse.ArgumentParser()
ap.add_argument("--corpus", default="imdb")
ap.add_argument("--n-queries", type=int, default=200)
ap.add_argument("--budgets", type=int, nargs="+", default=[100, 1000, 10000])
ap.add_argument("--eps", type=float, nargs="+", default=[0.02, 0.5])
ap.add_argument("--alpha", type=float, default=0.05)
ap.add_argument("--strata-cap", type=int, default=2048)
ap.add_argument("--kmeans-iter", type=int, default=20)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
Z = norm.ppf(1 - a.alpha / 2)

CORP = ROOT / "corpora" / a.corpus
df = pd.read_parquet(CORP / "corpus.parquet")
emb = np.load(CORP / "emb.npy").astype(np.float32)
thr = {"amazon": 2015.0, "imdb": 2000.0}.get(a.corpus, 3.0)
adm = df.filter_num.values >= thr
codes, facets = pd.factorize(df.facet.values[adm])
vals = df.value.values[adm].astype(np.float64)
E = emb[adm]
G, N = len(facets), int(adm.sum())
print(f"{a.corpus}: {N:,} admitted, {G} facets", flush=True)

# ---- OLLA's stratification: K-means over the embedding space --------
H = int(min(a.strata_cap, max(G, round(G * np.log(N)))))
import faiss
t0 = time.perf_counter()
km = faiss.Kmeans(E.shape[1], H, niter=a.kmeans_iter, seed=a.seed,
                  verbose=False, spherical=True)
km.train(np.ascontiguousarray(E))
_, strat = km.index.search(np.ascontiguousarray(E), 1)
strat = strat.ravel().astype(np.int64)
t_strat = time.perf_counter() - t0
print(f"  H = {H} strata (rule H0 = K log N), K-means in {t_strat:.0f}s",
      flush=True)

# per (facet, stratum) row lists, built once
key = codes.astype(np.int64) * H + strat
order = np.argsort(key, kind="stable")
key_s = key[order]
bounds = np.searchsorted(key_s, np.arange(G * H + 1))
cells = {}
for g in range(G):
    cs = []
    for h in range(H):
        lo, hi = bounds[g * H + h], bounds[g * H + h + 1]
        if hi > lo:
            cs.append(order[lo:hi])
    cells[g] = cs
sizes = np.array([sum(len(c) for c in cells[g]) for g in range(G)])

rng = np.random.RandomState(a.seed)
titles = df.text.str.split(".").str[0].str.strip()
ok = titles.str.split().str.len().between(4, 20) & ~titles.duplicated() & adm
qidx = rng.choice(np.flatnonzero(ok.values), size=a.n_queries, replace=False)

E64 = E.astype(np.float64)
recs = []
t0 = time.time()
for qn, qi in enumerate(qidx):
    s_all = E64 @ emb[qi].astype(np.float64)
    rs = np.random.RandomState(a.seed * 7919 + qn)
    for eps in a.eps:
        m = np.full(G, -np.inf); np.maximum.at(m, codes, s_all)
        w_all = np.exp((s_all - m[codes]) / eps)
        num = np.zeros(G); den = np.zeros(G)
        np.add.at(num, codes, w_all * vals); np.add.at(den, codes, w_all)
        exact = np.where(den > 0, num / den, np.nan)

        for B in a.budgets:
            t_q = time.perf_counter()
            n_scored = 0
            for g in range(G):
                if sizes[g] == 0 or not np.isfinite(exact[g]):
                    continue
                # proportional allocation over this facet's strata, with
                # a floor of 2 so a variance is estimable at all
                b_g = max(2, int(round(B * sizes[g] / N)))
                cs = cells[g]
                Ns = np.array([len(c) for c in cs], dtype=np.int64)
                # exact largest-remainder allocation of b_g over strata,
                # so the budget actually spent equals the budget granted
                share = b_g * Ns / Ns.sum()
                nh_arr = np.floor(share).astype(np.int64)
                left = b_g - int(nh_arr.sum())
                if left > 0:
                    take = np.argsort(-(share - nh_arr))[:left]
                    nh_arr[take] += 1
                nh_arr = np.minimum(np.maximum(nh_arr, 0), Ns)

                Tn = Td = 0.0
                parts, singles = [], []
                for c, Nh, nh in zip(cs, Ns, nh_arr):
                    if nh <= 0:
                        continue
                    pick = c[rs.choice(Nh, size=int(nh), replace=False)]
                    n_scored += int(nh)
                    wh = np.exp((s_all[pick] - m[g]) / eps)
                    Tn += Nh / nh * (wh * vals[pick]).sum()
                    Td += Nh / nh * wh.sum()
                    (parts if nh > 1 else singles).append(
                        (int(Nh), int(nh), wh, vals[pick]))
                if Td <= 0:
                    continue
                R = Tn / Td

                # linearised stratified variance of the ratio, with fpc.
                # Strata that received one unit cannot contribute a
                # within-stratum variance, so they are collapsed into a
                # single pooled stratum and treated as a simple random
                # sample of its union.
                var = 0.0
                for Nh, nh, wh, vh in parts:
                    u = (wh * vh - R * wh) / Td
                    var += Nh ** 2 * (1 - nh / Nh) * u.var(ddof=1) / nh
                if len(singles) > 1:
                    Np = sum(x[0] for x in singles)
                    np_ = len(singles)
                    wp = np.concatenate([x[2] for x in singles])
                    vp = np.concatenate([x[3] for x in singles])
                    up = (wp * vp - R * wp) / Td
                    var += Np ** 2 * (1 - np_ / Np) * up.var(ddof=1) / np_
                half = Z * np.sqrt(max(var, 0.0))
                recs.append(dict(
                    corpus=a.corpus, query_i=qn, eps=eps, budget=B, facet=g,
                    n_sampled=int(sum(x[1] for x in parts)
                                  + sum(x[1] for x in singles)),
                    point=R, half_width=half,
                    covers=bool(abs(R - exact[g]) <= half),
                    abs_err=abs(R - exact[g])))
            ms = (time.perf_counter() - t_q) * 1e3
            for r in recs[-1:]:
                r["query_ms"] = ms
                r["query_rows_scored"] = n_scored
    if (qn + 1) % 25 == 0:
        print(f"  {qn+1}/{len(qidx)}  ({time.time()-t0:.0f}s)", flush=True)

R = pd.DataFrame(recs)
agg = (R.groupby(["eps", "budget"])
        .agg(facets_answered=("facet", "nunique"),
             rows_scored_per_query=("n_sampled", "sum"),
             ci_coverage=("covers", "mean"),
             half_width_median=("half_width", "median"),
             abs_err_median=("abs_err", "median"),
             abs_err_p95=("abs_err", lambda x: float(x.quantile(0.95))))
        .reset_index())
agg["rows_scored_per_query"] = (agg.rows_scored_per_query
                                / max(len(qidx), 1)).round(0)
agg["facet_coverage"] = agg.facets_answered / G
summary = dict(
    corpus=a.corpus, n_admitted=N, n_facets=G, n_queries=int(len(qidx)),
    strata=H, strata_rule="H0 = K log N, capped",
    kmeans_seconds=round(t_strat, 1), nominal_coverage=1 - a.alpha,
    estimator="stratified ratio estimator with linearised variance and "
              "finite-population correction (stronger than the paper's "
              "Equation 1 applied to a ratio)",
    handicaps_removed=["true facet labels given for free (no LLM "
                       "inference needed)",
                       "ratio estimator instead of a sample-mean interval"],
    results=agg.to_dict(orient="records"))
(OUT / f"olla_{a.corpus}.json").write_text(json.dumps(summary, indent=2))
R.to_parquet(OUT / f"olla_per_facet_{a.corpus}.parquet", index=False)
pd.set_option("display.width", 220)
print("\n" + agg.to_string(index=False))
