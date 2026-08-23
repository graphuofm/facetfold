"""RQ1 extended: the baselines an IR reviewer will ask for.

Three additions to the coverage study.

1. UNIFORM SAMPLING AS A METHOD, not just as a reference model. If
   similarity retrieval loses facets because it concentrates, the
   obvious question is why not sample uniformly instead. Sampling does
   cover more facets, so it must be judged on the accuracy of the
   values it returns, not only on coverage. Both are reported.

2. STRATIFIED SAMPLING, the strongest cheap baseline that ignores
   relevance: draw k/G items per facet, which cannot lose a non-empty
   facet at all.

2b. PER-FACET TOP-b, the strongest opponent full stop: take the b = k/G
   BEST-SCORING items of each facet, at the same total budget. This is
   grouped retrieval as deployed vector stores expose it. It attains
   full coverage by construction AND keeps the relevance ordering
   within each facet, so it attacks our coverage claim directly. It is
   included so the claim is not won by default. What it does not
   supply is a bound on the weight it omitted, which is the property
   the certified plan adds.

3. SIGNIFICANCE. Paired Wilcoxon signed-rank tests over the query set
   for every method against the exact answer, with Holm correction
   across the methods compared, so differences are not read off means
   alone.

A temperature sweep is run as well, because the temperature is the
operator's central parameter and its effect on all of the above should
be visible rather than fixed at one value.
"""
from pathlib import Path as _P
_ROOT = _P(__file__).resolve().parent.parent.parent
import argparse, json, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon

ROOT = Path(str(_ROOT / "experiments") + "")
OUT = ROOT / "rq1_coverage"

ap = argparse.ArgumentParser()
ap.add_argument("--emb", default="emb.npy",
                help="embedding file under the corpus dir; a second "
                     "encoder tests whether the findings are a property "
                     "of retrieval or of one representation")
ap.add_argument("--corpus", default="amazon")
ap.add_argument("--n-queries", type=int, default=200)
ap.add_argument("--k", type=int, default=1000)
ap.add_argument("--eps", type=float, nargs="+", default=[0.02, 0.05, 0.1, 0.5])
ap.add_argument("--min-num", type=float, default=None)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

CORP = ROOT / "corpora" / a.corpus
df = pd.read_parquet(CORP / "corpus.parquet")
emb = np.load(CORP / a.emb)
thr = a.min_num if a.min_num is not None else (
    2015.0 if a.corpus == "amazon" else 2000.0 if a.corpus == "imdb" else 3.0)
adm = df.filter_num.values >= thr
codes, facets = pd.factorize(df.facet.values[adm])
vals = df.value.values[adm].astype(np.float64)
G, N = len(facets), int(adm.sum())
print(f"{a.corpus}: {N:,} admitted, {G} facets, k={a.k}", flush=True)

rng = np.random.RandomState(a.seed)
titles = df.text.str.split(".").str[0].str.strip()
ok = titles.str.split().str.len().between(4, 20) & ~titles.duplicated()
qidx = rng.choice(np.flatnonzero(ok.values), size=a.n_queries, replace=False)

dev = "cuda" if torch.cuda.is_available() else "cpu"
E = torch.from_numpy(emb[adm]).to(dev)
Q = torch.from_numpy(emb[qidx]).to(dev)
tc = torch.from_numpy(codes.astype(np.int64)).to(dev)
tv = torch.from_numpy(vals).to(dev)

# per-facet row lists for stratified sampling
facet_rows = [np.flatnonzero(codes == g) for g in range(G)]


def agg(idx, sims, eps):
    c = tc[idx] if idx is not None else tc
    v = tv[idx] if idx is not None else tv
    s = (sims if idx is None else sims[idx]).double() / eps
    m = torch.full((G,), -np.inf, dtype=torch.float64, device=dev)
    m = m.scatter_reduce(0, c, s, reduce="amax", include_self=True)
    w = torch.exp(s - m[c])
    num = torch.zeros(G, dtype=torch.float64, device=dev).scatter_add_(0, c, w * v)
    den = torch.zeros(G, dtype=torch.float64, device=dev).scatter_add_(0, c, w)
    return torch.where(den > 0, num / den, torch.full_like(num, np.nan)).cpu().numpy()


per_q = []
t0 = time.time()
for n, qi in enumerate(qidx):
    sims = (E @ Q[n]).double()
    # methods, all at the SAME retrieval budget k
    top = torch.topk(sims, min(a.k, N)).indices
    uni = torch.from_numpy(rng.choice(N, size=min(a.k, N), replace=False)).to(dev)
    per = max(1, a.k // G)
    strat = np.concatenate([r[rng.choice(len(r), size=min(per, len(r)),
                                         replace=False)] for r in facet_rows])
    strat_t = torch.from_numpy(strat).to(dev)
    # per-facet top-b: the b best-scoring rows WITHIN each facet
    pf = []
    for r in facet_rows:
        if len(r) == 0:
            continue
        rt = torch.from_numpy(r).to(dev)
        b = min(per, len(r))
        pf.append(rt[torch.topk(sims[rt], b).indices])
    perfacet_t = torch.cat(pf) if pf else strat_t
    for eps in a.eps:
        ref = agg(None, sims, eps)
        okr = np.isfinite(ref)
        for name, idx in (("topk", top), ("uniform", uni),
                          ("stratified", strat_t), ("perfacet", perfacet_t)):
            est = agg(idx, sims, eps)
            oke = np.isfinite(est)
            both = okr & oke
            per_q.append(dict(
                query_i=n, eps=eps, method=name,
                budget=int(len(idx)),
                coverage=float(oke.sum() / max(okr.sum(), 1)),
                mae=float(np.mean(np.abs(est[both] - ref[both]))) if both.any() else np.nan,
                max_abs=float(np.max(np.abs(est[both] - ref[both]))) if both.any() else np.nan))
    if (n + 1) % 50 == 0:
        print(f"  {n+1}/{len(qidx)} ({time.time()-t0:.0f}s)", flush=True)

R = pd.DataFrame(per_q)
agg_tbl = (R.groupby(["eps", "method"])
             .agg(budget=("budget", "median"), coverage=("coverage", "mean"),
                  coverage_sd=("coverage", "std"), mae=("mae", "mean"),
                  mae_med=("mae", "median"), max_abs=("max_abs", "mean"))
             .reset_index())

# paired significance, per eps: every method against topk, Holm-corrected
tests = []
for eps in a.eps:
    sub = R[R.eps == eps]
    base = sub[sub.method == "topk"].set_index("query_i")
    raw = []
    for m in ("uniform", "stratified", "perfacet"):
        oth = sub[sub.method == m].set_index("query_i")
        for metric in ("coverage", "mae"):
            x, y = base[metric], oth[metric]
            j = x.notna() & y.notna()
            if j.sum() > 10 and (x[j] - y[j]).abs().sum() > 0:
                st = wilcoxon(x[j], y[j])
                raw.append(dict(eps=eps, metric=metric, method=m,
                                p=float(st.pvalue), n=int(j.sum()),
                                mean_topk=float(x[j].mean()),
                                mean_other=float(y[j].mean())))
    # Holm across the comparisons made at this temperature
    for rank, t in enumerate(sorted(raw, key=lambda r: r["p"])):
        t["p_holm"] = min(1.0, t["p"] * (len(raw) - rank))
        t["significant_05"] = bool(t["p_holm"] < 0.05)
    tests += raw

summary = dict(
    corpus=a.corpus, encoder=a.emb, n_admitted=N, n_facets=G, n_queries=int(len(qidx)),
    retrieval_budget_k=a.k, eps_values=a.eps,
    methods=dict(
        topk="the k highest-scoring admitted items",
        uniform="k items drawn uniformly at random from the admitted set",
        stratified="k/G items per facet: cannot lose a non-empty facet, "
                   "and is therefore the strongest cheap opponent for the "
                   "coverage claim"),
    results=agg_tbl.to_dict(orient="records"),
    significance=dict(
        test="paired Wilcoxon signed-rank against top-k over the query set",
        correction="Holm, within each temperature",
        tests=tests),
)
(OUT / f"baselines_{a.corpus}{'' if a.emb=='emb.npy' else '_'+a.emb.replace('emb_','').replace('.npy','')}.json").write_text(json.dumps(summary, indent=2))
R.to_parquet(OUT / f"baselines_per_query_{a.corpus}.parquet", index=False)
pd.set_option("display.width", 200)
print("\n" + agg_tbl.to_string(index=False))
