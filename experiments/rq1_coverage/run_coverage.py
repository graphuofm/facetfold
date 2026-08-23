"""RQ1: does top-k retrieval + aggregation answer the aggregation query?

The query a web application asks:

    for each product category, the average star rating of reviews
    similar to <q>, over verified purchases since <year>

Methods
  exhaustive   soft aggregation over every admitted item (ground truth,
               max-anchored float64)
  top-k        what practitioners do today: retrieve the k globally
               most similar admitted items from a vector index, then
               group and aggregate what came back

Metrics (per query, reported as distributions over the query set)
  coverage      answered facets / facets in the ground truth
  mae_rel       mean relative error, over facets the method DID answer
                (so a method is not penalised twice for dropping them)
  spearman_ans  facet-ranking correlation over answered facets
  spearman_all  facet-ranking correlation over ALL facets, missing
                facets ranked last -- the honest downstream view
  top1_agree    does the method pick the same best facet?

Query set: review titles sampled deterministically from the corpus
(genuine user-written phrases), >=4 words, deduped.
"""
from pathlib import Path as _P
_ROOT = _P(__file__).resolve().parents[2]
import argparse, json, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

ROOT = Path(str(_ROOT / "experiments"))
OUT = ROOT / "rq1_coverage"

ap = argparse.ArgumentParser()
ap.add_argument("--emb", default="emb.npy",
                help="embedding file under the corpus dir; a second "
                     "encoder tests whether the findings are a property "
                     "of retrieval or of one representation")
ap.add_argument("--corpus", default="amazon",
                help="corpus dir name under experiments/corpora/")
ap.add_argument("--n-queries", type=int, default=200)
ap.add_argument("--query-set", default=None,
                help="dir under corpora/ holding query_emb.npy + queries.json; "
                     "default: synthesise queries from the corpus itself")
ap.add_argument("--min-num", type=float, default=None,
                help="predicate: keep rows with filter_num >= this "
                     "(default: the corpus meta's own predicate)")
ap.add_argument("--use-bool", action="store_true", default=True,
                help="predicate: also require filter_bool")
ap.add_argument("--year", type=int, default=2015)
ap.add_argument("--eps", type=float, nargs="+", default=[0.05, 0.1])
ap.add_argument("--ks", type=int, nargs="+", default=[10, 100, 1000, 10000])
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

CORP = ROOT / "corpora" / a.corpus
cmeta = json.loads((CORP / "meta.json").read_text())
df = pd.read_parquet(CORP / "corpus.parquet")
emb = np.load(CORP / a.emb)
print(f"corpus {a.corpus}: {len(df):,} rows, {df.facet.nunique()} facets",
      flush=True)

# ---- the exact predicate ----
# Each corpus carries its own natural predicate; the shape is always
# "a boolean quality flag AND a numeric threshold", so the query is the
# same across corpora even though the columns mean different things.
thr = a.min_num if a.min_num is not None else (
    a.year if a.corpus in ("amazon", "imdb") else 3.0)
# One numeric comparison, identical to what the engine's SQL layer can
# express, so RQ1 and RQ2 measure the same query on the same rows.
admit = df.filter_num.values >= thr
pred_desc = f"{cmeta.get('filter_num_col','filter_num')} >= {thr:g}"
n_adm = int(admit.sum())
codes, facets = pd.factorize(df.facet.values[admit])
vals = df.value.values[admit].astype(np.float64)
G = len(facets)
print(f"admitted {n_adm:,} ({n_adm/len(df):.1%}), {G} facets survive", flush=True)

# ---- query set ----
rng = np.random.RandomState(a.seed)
if a.query_set:
    # an external, independently-collected workload (e.g. real user
    # search queries), so the finding cannot be an artifact of queries
    # synthesised from the corpus under test
    QD = ROOT / "corpora" / a.query_set
    qmeta = json.loads((QD / "queries.json").read_text())
    qemb_all = np.load(QD / "query_emb.npy")
    take = min(a.n_queries, len(qemb_all))
    queries = qmeta["queries"][:take]
    qemb = qemb_all[:take]
    qsrc = f"{a.query_set}: {qmeta.get('source','external')}"
else:
    # query = the first sentence of an item's text
    titles = df.text.str.split(".").str[0].str.strip()
    ok = titles.str.split().str.len().between(4, 20) & ~titles.duplicated()
    pool = np.flatnonzero(ok.values)
    qidx = rng.choice(pool, size=a.n_queries, replace=False)
    queries = list(titles.values[qidx])
    qemb = emb[qidx]
    qsrc = "first sentence of corpus items, 4-20 words, deduped"
print(f"query set: {len(queries)} ({qsrc}), e.g. {queries[0]!r}", flush=True)

dev = "cuda" if torch.cuda.is_available() else "cpu"
E = torch.from_numpy(emb[admit]).to(dev)                    # (n_adm, 384)
Q = torch.from_numpy(np.ascontiguousarray(qemb)).to(dev)    # (nq, 384)
tcodes = torch.from_numpy(codes.astype(np.int64)).to(dev)
tvals = torch.from_numpy(vals).to(dev)


def soft_agg(sims, idx, eps, G):
    """Max-anchored per-facet softmax-weighted mean over `idx` rows."""
    c = tcodes[idx] if idx is not None else tcodes
    v = tvals[idx] if idx is not None else tvals
    s = sims.double()
    m = torch.full((G,), -np.inf, dtype=torch.float64, device=dev)
    m = m.scatter_reduce(0, c, s, reduce="amax", include_self=True)
    w = torch.exp(s - m[c])
    num = torch.zeros(G, dtype=torch.float64, device=dev).scatter_add_(0, c, w * v)
    den = torch.zeros(G, dtype=torch.float64, device=dev).scatter_add_(0, c, w)
    out = torch.where(den > 0, num / den, torch.full_like(num, np.nan))
    return out.cpu().numpy()


rows = []
t0 = time.time()
for qi in range(len(queries)):
    sims_raw = (E @ Q[qi]).double()
    for eps in a.eps:
        gt = soft_agg(sims_raw / eps, None, eps, G)
        gt_ok = ~np.isnan(gt)
        gt_sorted = np.sort(gt[gt_ok])[::-1]
        gt_gap = float(gt_sorted[0] - gt_sorted[1]) if gt_ok.sum() > 1 else np.nan
        gt_spread = float(gt_sorted[0] - gt_sorted[-1]) if gt_ok.sum() > 1 else np.nan
        order_gt = np.argsort(-np.where(gt_ok, gt, -np.inf))
        for k in a.ks:
            kk = min(k, n_adm)
            top = torch.topk(sims_raw, kk).indices
            est = soft_agg(sims_raw[top] / eps, top, eps, G)
            est_ok = ~np.isnan(est)
            both = gt_ok & est_ok
            cov = est_ok.sum() / max(gt_ok.sum(), 1)
            mae = (np.abs(est[both] - gt[both]) / np.abs(gt[both])).mean() if both.any() else np.nan
            sp_ans = spearmanr(gt[both], est[both]).statistic if both.sum() > 2 else np.nan
            # honest view: unanswered facets ranked last
            est_all = np.where(est_ok, est, -np.inf)
            sp_all = spearmanr(gt[gt_ok], est_all[gt_ok]).statistic if gt_ok.sum() > 2 else np.nan
            rows.append(dict(query_i=int(qi), eps=eps, k=int(k),
                             coverage=float(cov), mae_rel=float(mae),
                             spearman_ans=float(sp_ans), spearman_all=float(sp_all),
                             top1_agree=bool(order_gt[0] == int(np.argmax(est_all))),
                             n_facets_gt=int(gt_ok.sum()),
                             n_facets_est=int(est_ok.sum()),
                             gt_top1_gap=gt_gap, gt_spread=gt_spread))
    if (qi + 1) % 25 == 0:
        print(f"  {qi+1}/{len(queries)} queries ({time.time()-t0:.0f}s)", flush=True)

res = pd.DataFrame(rows)
tag = a.corpus + (f"_{a.query_set}" if a.query_set else "") \
      + ("" if a.emb == "emb.npy" else "_" + a.emb.replace("emb_","").replace(".npy",""))
res.to_parquet(OUT / f"per_query_{tag}.parquet", index=False)

agg = (res.groupby(["eps", "k"])
          .agg(coverage_mean=("coverage", "mean"), coverage_std=("coverage", "std"),
               mae_rel_mean=("mae_rel", "mean"), mae_rel_med=("mae_rel", "median"),
               spearman_ans=("spearman_ans", "mean"), spearman_all=("spearman_all", "mean"),
               top1_agree=("top1_agree", "mean"),
               facets_answered=("n_facets_est", "mean"))
          .reset_index())
SEP = 0.05  # stars: the winner must beat the runner-up by this much
sep = res[res.gt_top1_gap > SEP]
agg2 = (sep.groupby(["eps", "k"]).agg(top1_agree_separable=("top1_agree", "mean"),
                                      n_separable=("top1_agree", "size")).reset_index())
agg = agg.merge(agg2, on=["eps", "k"], how="left")
summary = dict(
    corpus=cmeta.get("corpus", a.corpus), corpus_dir=a.corpus,
    n_rows=int(len(df)), n_admitted=n_adm,
    predicate=pred_desc,
    n_facets_total=int(df.facet.nunique()), n_facets_admitted=G,
    n_queries=len(queries), query_source="first sentence of corpus items, 4-20 words, deduped",
    seed=a.seed, eps_values=a.eps, k_values=a.ks,
    encoder=a.emb,
    ground_truth="max-anchored float64 soft aggregation over all admitted rows",
    results=agg.to_dict(orient="records"),
    separability=dict(
        threshold_stars=SEP,
        note="top-1 agreement is also reported restricted to queries whose "
             "ground-truth winner beats the runner-up by more than the "
             "threshold, so ties are not counted as failures",
        gt_top1_gap_median=float(res.gt_top1_gap.median()),
        gt_spread_median=float(res.gt_spread.median()),
        frac_queries_separable=float((res.gt_top1_gap > SEP).mean())),
    wall_seconds=round(time.time() - t0, 1),
)
(OUT / f"results_{tag}.json").write_text(json.dumps(summary, indent=2))
pd.set_option("display.width", 200)
print("\n" + agg.to_string(index=False))
print(f"\nwrote {OUT}/results_{tag}.json  ({time.time()-t0:.0f}s)")
