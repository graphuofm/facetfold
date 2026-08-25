"""What does fixing coverage with retrieval alone actually cost?

Per-facet top-b is compared in Section 7 under a matched candidate
budget, which is the right first comparison but says nothing about
execution cost. Grouped retrieval is not free: it needs each facet
ordered by score, which is either one index per facet or the same full
pass the exact answer needs.

Three ways to produce the same per-facet top-b, timed against one
exact pass over the admitted rows:

  per_facet_index   one FAISS index per facet, probed for b = k/G.
                    G probes per query, plus a build cost paid once.
  scan_partition    one pass over the admitted rows, then a per-facet
                    partial sort. No index, and the pass is the same
                    one the exact plan performs.
  global_overfetch  a single global ANN probe over-fetched by 1/target
                    and then grouped, which is what a deployed stack
                    does today. Included because it is cheap and,
                    Section 7 shows, incomplete.

Latencies are reported as ratios to the exact pass measured in the same
loop, so the three share whatever conditions the machine is in; the
absolute exact-pass cost under the quiet guard is in RQ2. Coverage is
reported alongside, because a fast method that loses facets is not
solving the problem this baseline exists to solve.
"""
import argparse, json, statistics, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import faiss

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
np.seterr(invalid="ignore", divide="ignore")

ap = argparse.ArgumentParser()
ap.add_argument("--corpus", default="imdb")
ap.add_argument("--k", type=int, default=1000)
ap.add_argument("--n-queries", type=int, default=30)
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--threads", type=int, default=8)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
faiss.omp_set_num_threads(a.threads)

CORP = ROOT / "corpora" / a.corpus
df = pd.read_parquet(CORP / "corpus.parquet")
emb = np.load(CORP / "emb.npy").astype(np.float32)
thr = {"amazon": 2015.0, "imdb": 2000.0}.get(a.corpus, 3.0)
adm = df.filter_num.values >= thr
codes, facets = pd.factorize(df.facet.values[adm])
X = np.ascontiguousarray(emb[adm])
faiss.normalize_L2(X)
G, N = len(facets), int(adm.sum())
rows_of = [np.flatnonzero(codes == g) for g in range(G)]
b = max(1, a.k // G)
print(f"{a.corpus}: {N:,} admitted, {G} facets, b={b} per facet", flush=True)

# one flat index per facet, built once
t0 = time.perf_counter()
per_facet = []
for g in range(G):
    idx = faiss.IndexFlatIP(X.shape[1])
    if rows_of[g].size:
        idx.add(np.ascontiguousarray(X[rows_of[g]]))
    per_facet.append(idx)
t_build_pf = time.perf_counter() - t0
# one global index, for the over-fetch route
t0 = time.perf_counter()
glob = faiss.IndexFlatIP(X.shape[1])
glob.add(X)
t_build_gl = time.perf_counter() - t0
print(f"  build: {G} per-facet indexes {t_build_pf:.1f}s, global {t_build_gl:.1f}s",
      flush=True)

rng = np.random.RandomState(a.seed)
titles = df.text.str.split(".").str[0].str.strip()
ok = titles.str.split().str.len().between(4, 20) & ~titles.duplicated() & adm
qidx = rng.choice(np.flatnonzero(ok.values), size=a.n_queries, replace=False)

def med(fn, reps):
    fn()                                   # discard a warm-up
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); out = fn(); ts.append(time.perf_counter() - t)
    return statistics.median(ts) * 1e3, out

recs = []
for qn, qi in enumerate(qidx):
    q = np.ascontiguousarray(emb[qi:qi+1].astype(np.float32)); faiss.normalize_L2(q)
    qv = q[0]

    def exact():
        s = X @ qv
        return np.argpartition(-s, min(a.k, N) - 1)[:min(a.k, N)]

    def per_facet_probe():
        got = []
        for g in range(G):
            if rows_of[g].size == 0:
                continue
            _, I = per_facet[g].search(q, min(b, rows_of[g].size))
            got.append(rows_of[g][I[0][I[0] >= 0]])
        return np.concatenate(got) if got else np.array([], dtype=np.int64)

    def scan_partition():
        s = X @ qv
        got = []
        for g in range(G):
            r = rows_of[g]
            if r.size == 0:
                continue
            bb = min(b, r.size)
            got.append(r[np.argpartition(-s[r], bb - 1)[:bb]] if bb < r.size else r)
        return np.concatenate(got)

    def global_overfetch():
        _, I = glob.search(q, min(a.k, N))
        return I[0][I[0] >= 0]

    ms_ex, top_ex = med(exact, a.reps)
    rec = dict(query_i=qn, exact_ms=ms_ex,
               exact_coverage=len(set(codes[top_ex])) / G)
    for name, fn in (("per_facet_index", per_facet_probe),
                     ("scan_partition", scan_partition),
                     ("global_overfetch", global_overfetch)):
        ms, got = med(fn, a.reps)
        rec[f"{name}_ms"] = ms
        rec[f"{name}_ratio"] = ms / ms_ex
        rec[f"{name}_coverage"] = len(set(codes[got])) / G if len(got) else 0.0
    recs.append(rec)
    if (qn + 1) % 10 == 0:
        print(f"  {qn+1}/{len(qidx)}", flush=True)

R = pd.DataFrame(recs)
M = ["per_facet_index", "scan_partition", "global_overfetch"]
summary = dict(corpus=a.corpus, n_admitted=N, n_facets=G, k=a.k, b=b,
               n_queries=int(len(qidx)), reps=a.reps, threads=a.threads,
               build_seconds=dict(per_facet_indexes=round(t_build_pf, 1),
                                  global_index=round(t_build_gl, 1)),
               note="ratios are to the exact pass timed in the same loop; "
                    "absolute exact-pass latency under the quiet guard is "
                    "in RQ2. Coverage is the share of facets represented.",
               exact_coverage=float(R.exact_coverage.mean()),
               results=[dict(method=m,
                             ratio_to_exact=float(R[f"{m}_ratio"].median()),
                             ms_median=float(R[f"{m}_ms"].median()),
                             coverage=float(R[f"{m}_coverage"].mean()))
                        for m in M])
(OUT / f"grouped_latency_{a.corpus}.json").write_text(json.dumps(summary, indent=2))
R.to_parquet(OUT / f"grouped_latency_per_query_{a.corpus}.parquet", index=False)
print(f"\n  {'method':18s} {'x exact':>9s} {'ms':>9s} {'coverage':>9s}")
print(f"  {'exact pass':18s} {1.0:9.2f} {float(R.exact_ms.median()):9.2f} "
      f"{float(R.exact_coverage.mean()):9.1%}")
for m in M:
    print(f"  {m:18s} {float(R[f'{m}_ratio'].median()):9.2f} "
          f"{float(R[f'{m}_ms'].median()):9.2f} {float(R[f'{m}_coverage'].mean()):9.1%}")
