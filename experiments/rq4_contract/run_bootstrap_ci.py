"""The obvious rebuttal: why not per-facet retrieval with a bootstrap CI?

Section 7 shows that spending the retrieval budget within each facet
fixes coverage and, at k >= 1000, is more accurate than a single global
ranked list. A reviewer will reasonably ask why that is not the whole
answer: attach a bootstrap confidence interval to each facet's mean and
the number has an error bar, without any of the machinery in Section 5.

This measures whether that error bar is trustworthy. Two estimators of
the same target, at the same per-facet budget b:

  perfacet_topk   the b BEST-SCORING rows of the facet, weighted mean
                  over them, percentile bootstrap over those b rows.
  perfacet_bayes  the same retained rows under a Bayesian (Dirichlet)
                  bootstrap, which reweights rather than resamples and
                  so cannot drop the dominant row entirely. Included so
                  the comparison is not won by a naive resampler.
  uniform_sample  b rows drawn uniformly from the facet, ratio
                  estimator (sum w v)/(sum w), percentile bootstrap.

All three produce a nominal 95% interval. A resampling interval
describes the variability of the set it was handed; it cannot describe
mass that was never in that set. So the top-b intervals are blind to
truncation bias by construction, and the uniform interval is blind to
whatever the sample missed -- which at a sharp temperature is the
handful of rows carrying nearly all the weight.

Reported per facet, pooled over queries: empirical coverage of the
nominal 95% interval, its half-width, the realised absolute error, the
deterministic bound of Section 5 at the same cut, and the weights'
effective sample size 1/sum(p^2), which explains the widths. The
comparison that matters is calibration against width: a narrow interval
that is wrong is worse than no interval, and a wide interval that
happens to cover is not a guarantee either.
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
np.seterr(invalid="ignore", divide="ignore")

ap = argparse.ArgumentParser()
ap.add_argument("--corpus", default="amazon")
ap.add_argument("--n-queries", type=int, default=200)
ap.add_argument("--k", type=int, nargs="+", default=[100, 1000])
ap.add_argument("--eps", type=float, nargs="+", default=[0.02, 0.5])
ap.add_argument("--boot", type=int, default=500)
ap.add_argument("--alpha", type=float, default=0.05)
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
print(f"{a.corpus}: {N:,} admitted, {G} facets", flush=True)

rng = np.random.RandomState(a.seed)
titles = df.text.str.split(".").str[0].str.strip()
ok = titles.str.split().str.len().between(4, 20) & ~titles.duplicated() & adm
qidx = rng.choice(np.flatnonzero(ok.values), size=a.n_queries, replace=False)

LO, HI = a.alpha / 2 * 100, (1 - a.alpha / 2) * 100


def _ci(point, est):
    if est.size == 0:
        return point, point, point
    return point, np.percentile(est, LO), np.percentile(est, HI)


def boot_ci(w, v, B, rs):
    """Percentile bootstrap of the weighted mean (sum w v)/(sum w)."""
    b = len(w)
    point = (w * v).sum() / w.sum()
    if b < 2:
        return point, point, point
    cnt = rs.multinomial(b, np.full(b, 1.0 / b), size=B)   # (B, b)
    num, den = cnt @ (w * v), cnt @ w
    good = den > 0
    return _ci(point, num[good] / den[good])


def bayes_ci(w, v, B, rs):
    """Bayesian bootstrap: Dirichlet(1,...,1) weights over the rows.

    Smoother than resampling, and it never removes a row outright, so a
    facet whose weight sits on one item is not handed an interval that
    is wide merely because the resampler kept dropping that item.
    """
    b = len(w)
    point = (w * v).sum() / w.sum()
    if b < 2:
        return point, point, point
    d = rs.dirichlet(np.ones(b), size=B)                   # (B, b)
    num, den = d @ (w * v), d @ w
    good = den > 0
    return _ci(point, num[good] / den[good])


def ess(w):
    """Effective sample size of the weights, 1 / sum(p^2)."""
    p = w / w.sum()
    return float(1.0 / np.square(p).sum())


recs = []
t0 = time.time()
for qn, qi in enumerate(qidx):
    s = E @ emb[qi]
    rs = np.random.RandomState(a.seed * 100003 + qn)
    for eps in a.eps:
        # exact per-facet answer
        m = np.full(G, -np.inf); np.maximum.at(m, codes, s)
        w_all = np.exp((s - m[codes]) / eps)
        num = np.zeros(G); den = np.zeros(G)
        np.add.at(num, codes, w_all * vals); np.add.at(den, codes, w_all)
        exact = np.where(den > 0, num / den, np.nan)

        for K in a.k:
            b_alloc = max(1, K // G)
            for g in range(G):
                idx = rows_of[g]
                if idx.size == 0 or not np.isfinite(exact[g]):
                    continue
                b = min(b_alloc, idx.size)
                sg = s[idx]

                # (1) per-facet top-b, bootstrapped over the retained rows
                top = idx[np.argpartition(-sg, b - 1)[:b]] if b < idx.size else idx
                wt = np.exp((s[top] - m[g]) / eps)
                # the deterministic bound of Section 5 at this same cut
                delta = 1.0 - wt.sum() / den[g]
                bound = delta * (g_hi[g] - g_lo[g])
                n_eff = ess(wt)
                for name, fn in (("perfacet_topk", boot_ci),
                                 ("perfacet_bayes", bayes_ci)):
                    pt, lo, hi = fn(wt, vals[top], a.boot, rs)
                    recs.append(dict(
                        corpus=a.corpus, query_i=qn, eps=eps, k=K, facet=g,
                        method=name, b=b, n_g=int(idx.size),
                        point=pt, lo=lo, hi=hi,
                        covers=bool(lo <= exact[g] <= hi),
                        half_width=(hi - lo) / 2,
                        abs_err=abs(pt - exact[g]),
                        certified_bound=bound, ess=n_eff))

                # (2) uniform sample of the same size, ratio estimator
                samp = idx[rs.choice(idx.size, size=b, replace=False)]
                ws = np.exp((s[samp] - m[g]) / eps)
                pt2, lo2, hi2 = boot_ci(ws, vals[samp], a.boot, rs)
                recs.append(dict(
                    corpus=a.corpus, query_i=qn, eps=eps, k=K, facet=g,
                    method="uniform_sample", b=b, n_g=int(idx.size),
                    point=pt2, lo=lo2, hi=hi2,
                    covers=bool(lo2 <= exact[g] <= hi2),
                    half_width=(hi2 - lo2) / 2,
                    abs_err=abs(pt2 - exact[g]),
                    certified_bound=np.nan, ess=ess(ws)))
    if (qn + 1) % 25 == 0:
        print(f"  {qn+1}/{len(qidx)}  ({time.time()-t0:.0f}s)", flush=True)

R = pd.DataFrame(recs)
agg = (R.groupby(["eps", "k", "method"])
        .agg(n_facet_queries=("covers", "size"),
             ci_coverage=("covers", "mean"),
             half_width_median=("half_width", "median"),
             abs_err_median=("abs_err", "median"),
             abs_err_p95=("abs_err", lambda x: x.quantile(0.95)),
             certified_bound_median=("certified_bound", "median"),
             ess_median=("ess", "median"))
        .reset_index())
summary = dict(
    corpus=a.corpus, n_admitted=N, n_facets=G, n_queries=int(len(qidx)),
    bootstrap_replicates=a.boot, nominal_coverage=1 - a.alpha,
    note="ci_coverage is the fraction of (query, facet) pairs whose "
         "nominal 95% bootstrap interval contains the exact answer. "
         "A calibrated interval sits at 0.95; below that the interval "
         "is overconfident. certified_bound is the deterministic "
         "per-facet bound of Section 5 evaluated at the same cut, for "
         "width comparison only.",
    results=agg.to_dict(orient="records"))
(OUT / f"bootstrap_{a.corpus}.json").write_text(json.dumps(summary, indent=2))
R.to_parquet(OUT / f"bootstrap_per_facet_{a.corpus}.parquet", index=False)
pd.set_option("display.width", 220)
print("\n" + agg.to_string(index=False))
