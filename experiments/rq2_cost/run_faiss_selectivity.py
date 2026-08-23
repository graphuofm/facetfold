"""A second ANN engine and a second index family.

Section 7's filtered-index result is measured on one engine (PostgreSQL
with pgvector) and one index family (HNSW). Recent measurement work
reports that engine, index family and selectivity all move filtered-ANN
outcomes materially, so a conclusion drawn from one configuration is
not safe. This repeats the selectivity sweep on an independent engine,
FAISS, across two index families, and adds the strategy pgvector does
not expose.

Three ways to serve a filtered top-k, all at the same requested k:

  post_filter   query the index over the whole corpus for k, then drop
                the rows the predicate rejects. This is the behaviour
                that makes the deployed route incomplete: at a
                selective predicate almost nothing survives.
  post_filter_x the same, over-fetching by a factor so that roughly k
                rows survive. The practical workaround, and it pays for
                the predicate in index work.
  pre_filter    restrict the search to admitted ids (FAISS IDSelector).
                Not what pgvector's HNSW does, and the strongest form
                of the baseline, so it is included rather than avoided.

Against one exact pass over the admitted rows. Reported per
selectivity: recall against the exact top-k, facet coverage, the
fraction of the requested k that survived, and latency.

A second phase then asks the question the paper actually cares about.
Being fast while incomplete is easy; the paper's claim is about what
completeness costs. So at a fixed selectivity we widen each index's
search parameter (efSearch for HNSW, nprobe for IVFFlat) until facet
coverage stops improving, and report the smallest setting that reaches
99% coverage together with its latency against the exact pass.
"""
import argparse, json, statistics, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import faiss

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quiet import require_quiet   # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent

ap = argparse.ArgumentParser()
ap.add_argument("--corpus", default="imdb")
ap.add_argument("--targets", type=float, nargs="+",
                default=[0.01, 0.05, 0.10, 0.25, 0.50, 1.00])
ap.add_argument("--k", type=int, default=1000)
ap.add_argument("--n-queries", type=int, default=20)
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--hnsw-m", type=int, default=32)
ap.add_argument("--ef-search", type=int, default=200)
ap.add_argument("--ivf-nlist", type=int, default=1024)
ap.add_argument("--ivf-nprobe", type=int, default=32)
ap.add_argument("--threads", type=int, default=8)
ap.add_argument("--tune-at", type=float, nargs="+", default=[0.05, 0.25],
                help="selectivities at which to widen the search until "
                     "coverage stops improving")
ap.add_argument("--ef-sweep", type=int, nargs="+",
                default=[50, 200, 800, 3200, 12800])
ap.add_argument("--nprobe-sweep", type=int, nargs="+",
                default=[8, 32, 128, 512, 1024])
ap.add_argument("--no-quiet", action="store_true")
a = ap.parse_args()
QUIET = None if a.no_quiet else require_quiet(wait_seconds=1800)
faiss.omp_set_num_threads(a.threads)

CORP = ROOT / "corpora" / a.corpus
df = pd.read_parquet(CORP / "corpus.parquet")
emb = np.load(CORP / "emb.npy")
X = np.ascontiguousarray(emb.astype(np.float32))
faiss.normalize_L2(X)                       # inner product == cosine
fnum = df.filter_num.values.astype(float)
facet = df.facet.values
n, d = X.shape
print(f"{a.corpus}: {n:,} rows, d={d}", flush=True)

rng = np.random.RandomState(0)
titles = df.text.str.split(".").str[0].str.strip()
ok = titles.str.split().str.len().between(4, 20) & ~titles.duplicated()
qidx = rng.choice(np.flatnonzero(ok.values), size=a.n_queries, replace=False)
Q = np.ascontiguousarray(X[qidx])

print("building indexes (once, over the whole corpus) ...", flush=True)
t0 = time.perf_counter()
hnsw = faiss.IndexHNSWFlat(d, a.hnsw_m, faiss.METRIC_INNER_PRODUCT)
hnsw.hnsw.efSearch = a.ef_search
hnsw.add(X)
t_hnsw = time.perf_counter() - t0
t0 = time.perf_counter()
quant = faiss.IndexFlatIP(d)
ivf = faiss.IndexIVFFlat(quant, d, a.ivf_nlist, faiss.METRIC_INNER_PRODUCT)
ivf.train(X)
ivf.add(X)
ivf.nprobe = a.ivf_nprobe
t_ivf = time.perf_counter() - t0
print(f"  HNSW {t_hnsw:.0f}s, IVFFlat {t_ivf:.0f}s", flush=True)

cuts = [(t, float(np.quantile(fnum, 1.0 - t))) for t in a.targets]
rows = []
for target, cut in cuts:
    adm = fnum >= cut
    adm_ids = np.flatnonzero(adm).astype(np.int64)
    n_adm = int(adm.sum())
    Xa = np.ascontiguousarray(X[adm])
    fa = facet[adm]
    n_fac = len(pd.unique(fa))
    kk = min(a.k, n_adm)
    sel = faiss.IDSelectorBatch(adm_ids.size, faiss.swig_ptr(adm_ids))

    for i, qv in enumerate(Q):
        qv1 = qv.reshape(1, -1)
        # exact reference over the admitted rows. The first touch of a
        # freshly sliced corpus pays page faults and a cold cache, so a
        # warm-up is discarded here exactly as in RQ2.
        _ = Xa @ qv
        ts = []
        for _ in range(a.reps):
            t0 = time.perf_counter()
            s = Xa @ qv
            top = np.argpartition(-s, kk - 1)[:kk]
            ts.append((time.perf_counter() - t0) * 1e3)
        exact_ms = statistics.median(ts)
        truth = set(map(int, adm_ids[top]))

        rec = dict(target_selectivity=target, threshold=cut, query_i=i,
                   n_admitted=n_adm, n_facets=n_fac, exact_ms=exact_ms)

        def probe(index, k_req, params=None):
            ts, I = [], None
            if params is None:
                index.search(qv1, k_req)
            else:
                index.search(qv1, k_req, params=params)
            for _ in range(a.reps):
                t0 = time.perf_counter()
                if params is None:
                    _, I = index.search(qv1, k_req)
                else:
                    _, I = index.search(qv1, k_req, params=params)
                ts.append((time.perf_counter() - t0) * 1e3)
            return statistics.median(ts), I[0][I[0] >= 0]

        for iname, index, mk_params in (
                ("hnsw", hnsw,
                 lambda s: faiss.SearchParametersHNSW(sel=s,
                                                      efSearch=a.ef_search)),
                ("ivfflat", ivf,
                 lambda s: faiss.SearchParametersIVF(sel=s,
                                                     nprobe=a.ivf_nprobe))):
            # (1) post-filter at the requested k
            ms, I = probe(index, kk)
            keep = I[adm[I]]
            rec[f"{iname}_post_ms"] = ms
            rec[f"{iname}_post_fill"] = len(keep) / kk
            rec[f"{iname}_post_recall"] = (
                len(truth & set(map(int, keep))) / max(len(truth), 1))
            rec[f"{iname}_post_coverage"] = (
                len(set(facet[keep])) / max(n_fac, 1))

            # (2) post-filter with over-fetch sized to the selectivity
            k_over = int(min(n, np.ceil(kk / max(target, 1e-3))))
            ms, I = probe(index, k_over)
            keep = I[adm[I]][:kk]
            rec[f"{iname}_over_k"] = k_over
            rec[f"{iname}_over_ms"] = ms
            rec[f"{iname}_over_fill"] = len(keep) / kk
            rec[f"{iname}_over_recall"] = (
                len(truth & set(map(int, keep))) / max(len(truth), 1))
            rec[f"{iname}_over_coverage"] = (
                len(set(facet[keep])) / max(n_fac, 1))

            # (3) pre-filter: the index only visits admitted ids
            try:
                ms, I = probe(index, kk, mk_params(sel))
                rec[f"{iname}_pre_ms"] = ms
                rec[f"{iname}_pre_fill"] = len(I) / kk
                rec[f"{iname}_pre_recall"] = (
                    len(truth & set(map(int, I))) / max(len(truth), 1))
                rec[f"{iname}_pre_coverage"] = (
                    len(set(facet[I])) / max(n_fac, 1))
            except Exception as e:                        # noqa: BLE001
                rec[f"{iname}_pre_ms"] = float("nan")
                rec[f"{iname}_pre_error"] = type(e).__name__
        rows.append(rec)
    print(f"  selectivity {target:.0%}: {n_adm:,} admitted, {n_fac} facets",
          flush=True)

# ---- phase 2: what does completeness cost on this engine? ----------
print("\ntuning each index up to completeness ...", flush=True)
tune = []
for target in a.tune_at:
    cut = float(np.quantile(fnum, 1.0 - target))
    adm = fnum >= cut
    adm_ids = np.flatnonzero(adm).astype(np.int64)
    Xa = np.ascontiguousarray(X[adm])
    fa = facet[adm]
    n_fac = len(pd.unique(fa))
    kk = min(a.k, int(adm.sum()))
    sel = faiss.IDSelectorBatch(adm_ids.size, faiss.swig_ptr(adm_ids))

    # The reference the tuning is actually chasing. An ANN index
    # converges to the EXACT top-k, so exact top-k's facet coverage is a
    # ceiling no amount of widening can pass. Without this line, an
    # index plateauing below 100% reads as an index defect; it is not.
    ex, ex_cov = [], []
    _ = Xa @ Q[0]
    for qv in Q:
        t0 = time.perf_counter()
        s_ = Xa @ qv
        top = np.argpartition(-s_, kk - 1)[:kk]
        ex.append((time.perf_counter() - t0) * 1e3)
        ex_cov.append(len(set(fa[top])) / max(n_fac, 1))
    exact_ms = statistics.median(ex)
    exact_topk_coverage = float(np.mean(ex_cov))
    print(f"  exact global top-{kk} covers {exact_topk_coverage:.1%} of "
          f"{n_fac} facets at {target:.0%} selectivity", flush=True)

    for iname, index, sweep in (("hnsw", hnsw, a.ef_sweep),
                                ("ivfflat", ivf, a.nprobe_sweep)):
        for setting in sweep:
            covs, mss = [], []
            for qv in Q:
                qv1 = qv.reshape(1, -1)
                if iname == "hnsw":
                    par = faiss.SearchParametersHNSW(sel=sel, efSearch=setting)
                else:
                    par = faiss.SearchParametersIVF(sel=sel, nprobe=setting)
                ts = []
                for _ in range(a.reps):
                    t0 = time.perf_counter()
                    _, I = index.search(qv1, kk, params=par)
                    ts.append((time.perf_counter() - t0) * 1e3)
                I = I[0][I[0] >= 0]
                mss.append(statistics.median(ts))
                covs.append(len(set(facet[I])) / max(n_fac, 1))
            tune.append(dict(target_selectivity=target, index=iname,
                             setting=setting, n_facets=n_fac,
                             coverage=float(np.mean(covs)),
                             ms=float(statistics.median(mss)),
                             exact_ms=exact_ms,
                             exact_topk_coverage=exact_topk_coverage,
                             ratio_to_exact=statistics.median(mss) / exact_ms))
        print(f"  {iname} at {target:.0%} done", flush=True)
T = pd.DataFrame(tune)

R = pd.DataFrame(rows)
num = [c for c in R.columns if c not in ("query_i",)]
agg = R[num].groupby("target_selectivity").median(numeric_only=True).reset_index()
summary = dict(machine_conditions=QUIET, corpus=a.corpus, engine="FAISS",
               faiss_version=faiss.__version__, k=a.k,
               n_queries=int(len(Q)), reps=a.reps, threads=a.threads,
               hnsw=dict(M=a.hnsw_m, efSearch=a.ef_search, build_s=t_hnsw),
               ivfflat=dict(nlist=a.ivf_nlist, nprobe=a.ivf_nprobe,
                            build_s=t_ivf),
               note="post = filter after retrieving k; over = over-fetch "
                    "k/selectivity then filter; pre = FAISS IDSelector so "
                    "the index only visits admitted ids. recall is against "
                    "the exact top-k over the admitted rows.",
               tune_note="smallest search-parameter setting reaching 99% "
                         "facet coverage, and what it costs against one "
                         "exact pass over the same admitted rows",
               results=agg.to_dict(orient="records"),
               tuning=T.to_dict(orient="records"),
               completeness=[
                   dict(target_selectivity=float(t), index=str(i),
                        setting=(int(sub[sub.coverage >= 0.99].setting.min())
                                 if (sub.coverage >= 0.99).any() else None),
                        best_coverage=float(sub.coverage.max()),
                        exact_topk_coverage=float(sub.exact_topk_coverage.iloc[0]),
                        ms=(float(sub[sub.coverage >= 0.99].ms.min())
                            if (sub.coverage >= 0.99).any() else None),
                        exact_ms=float(sub.exact_ms.iloc[0]),
                        ratio_to_exact=(
                            float(sub[sub.coverage >= 0.99].ms.min()
                                  / sub.exact_ms.iloc[0])
                            if (sub.coverage >= 0.99).any() else None))
                   for (t, i), sub in T.groupby(["target_selectivity", "index"])])
(OUT / f"faiss_selectivity_{a.corpus}.json").write_text(json.dumps(summary, indent=2))
R.to_parquet(OUT / f"faiss_per_query_{a.corpus}.parquet", index=False)
pd.set_option("display.width", 250)
cols = ["target_selectivity", "n_admitted", "exact_ms"] + [
    c for c in agg.columns if c.endswith(("_ms", "_fill", "_coverage")) and c != "exact_ms"]
print("\n" + agg[cols].to_string(index=False))
print("\n--- cost of completeness ---")
print(T.to_string(index=False))
print()
for c in summary["completeness"]:
    if c["setting"] is None:
        print(f"  {c['index']:8s} at {c['target_selectivity']:.0%}: never "
              f"reached 99% coverage (best {c['best_coverage']:.1%}, "
              f"against a ceiling of {c['exact_topk_coverage']:.1%} set by "
              f"exact top-k itself); exact pass {c['exact_ms']:.1f} ms")
    else:
        print(f"  {c['index']:8s} at {c['target_selectivity']:.0%}: 99% "
              f"coverage at setting {c['setting']}, {c['ms']:.1f} ms = "
              f"{c['ratio_to_exact']:.2f}x the {c['exact_ms']:.1f} ms exact pass")
