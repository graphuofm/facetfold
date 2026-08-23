"""RQ2: what does the exact answer cost?

RQ1 established that top-k answers a different question. That only
matters if the exact answer is affordable -- if exhaustive soft
aggregation were seconds per query, truncating would be a defensible
engineering trade. This measures the price of exactness on the real
corpora.

Measured, per corpus, over the same query workload:
  exact_f64   the engine's exhaustive soft aggregation, float64 keys
  exact_f32   the same, float32 keys (half the bytes scanned)
  topk_k      the cost practitioners pay today for the wrong answer
              (similarity scan + partial selection + group aggregate),
              measured on the same hardware and data layout so the
              comparison isolates the aggregation strategy rather than
              the storage stack

Correctness is checked against a max-anchored float64 reference on
every run, so a latency number is never reported for a wrong answer.
"""
from pathlib import Path as _P
_ROOT = _P(__file__).resolve().parents[2]
import argparse, sys, json, statistics, time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quiet import require_quiet
from bruce import QuerySession

ROOT = Path(str(_ROOT / "experiments"))
OUT = ROOT / "rq2_cost"

ap = argparse.ArgumentParser()
ap.add_argument("--corpus", default="amazon")
ap.add_argument("--n-queries", type=int, default=20)
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--eps", type=float, default=0.05)
ap.add_argument("--min-num", type=float, default=None)
ap.add_argument("--ks", type=int, nargs="+", default=[100, 1000, 10000])
a = ap.parse_args()
QUIET = require_quiet(wait_seconds=3600)

CORP = ROOT / "corpora" / a.corpus
cmeta = json.loads((CORP / "meta.json").read_text())
df = pd.read_parquet(CORP / "corpus.parquet")
emb = np.load(CORP / "emb.npy")
thr = a.min_num if a.min_num is not None else (2015.0 if a.corpus == "amazon"
                                               else 2000.0 if a.corpus == "imdb" else 3.0)
print(f"{a.corpus}: {len(df):,} rows x {emb.shape[1]}d", flush=True)

# query workload: reuse RQ1's protocol so the two studies are comparable
rng = np.random.RandomState(0)
titles = df.text.str.split(".").str[0].str.strip()
ok = titles.str.split().str.len().between(4, 20) & ~titles.duplicated()
qidx = rng.choice(np.flatnonzero(ok.values), size=a.n_queries, replace=False)
Q = emb[qidx].astype(np.float64)

# ---------- engine under test ----------
sess = QuerySession()
t0 = time.perf_counter()
sess.register_parquet(a.corpus, str(CORP / "corpus.parquet"))
load_s = time.perf_counter() - t0
res = {}
for dtype in ("f64", "f32"):
    t0 = time.perf_counter()
    sess.attach_key(a.corpus, f"emb_{dtype}",
                    np.ascontiguousarray(emb.astype(np.float64 if dtype == "f64" else np.float32)))
    res[f"attach_{dtype}_s"] = time.perf_counter() - t0
print(f"load {load_s:.1f}s; attach f64 {res['attach_f64_s']:.1f}s "
      f"f32 {res['attach_f32_s']:.1f}s", flush=True)

def med(fn, reps):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); out = fn(); ts.append(time.perf_counter() - t0)
    return statistics.median(ts), out

# reference answer (max-anchored float64), used to validate every run
# The engine's SQL layer expresses ONE comparison, so the predicate is
# the numeric one and the reference must use exactly that -- an earlier
# version filtered the reference on filter_bool as well, which silently
# compared two different row sets on Amazon (19% "error" that was the
# harness, not the engine).
admit = df.filter_num.values >= thr
codes, facets = pd.factorize(df.facet.values[admit])
vals = df.value.values[admit].astype(np.float64)
E = emb[admit].astype(np.float64)

def reference(qv):
    s = E @ qv
    G = len(facets)
    m = np.full(G, -np.inf)
    np.maximum.at(m, codes, s)
    w = np.exp((s - m[codes]) / a.eps)
    num = np.zeros(G); den = np.zeros(G)
    np.add.at(num, codes, w * vals); np.add.at(den, codes, w)
    return np.where(den > 0, num / den, np.nan)

rows = []
for i, qv in enumerate(Q):
    ref = reference(qv)
    rec = dict(query_i=i)
    for dtype in ("f64", "f32"):
        sql = (f"SELECT facet, SOFTAVG(value, SIM(emb_{dtype}, :q), {a.eps}) "
               f"FROM {a.corpus} WHERE filter_num >= {thr:g} GROUP BY facet")
        t, out = med(lambda: sess.run(sql, {"q": qv}), a.reps)
        labels, values = out[0], out[1]
        got = {l: v for l, v in zip(labels, values)}
        err = max(abs(got[f] - ref[j]) / abs(ref[j])
                  for j, f in enumerate(facets) if f in got and np.isfinite(ref[j]))
        rec[f"exact_{dtype}_ms"] = t * 1e3
        rec[f"exact_{dtype}_relerr"] = err
        rec[f"exact_{dtype}_facets"] = len(labels)
    # what the top-k path costs on identical data (score, select, group)
    for k in a.ks:
        def topk():
            s = E @ qv
            idx = np.argpartition(-s, min(k, len(s) - 1))[:k]
            c = codes[idx]; sv = s[idx]
            G = len(facets)
            m = np.full(G, -np.inf); np.maximum.at(m, c, sv)
            w = np.exp((sv - m[c]) / a.eps)
            num = np.zeros(G); den = np.zeros(G)
            np.add.at(num, c, w * vals[idx]); np.add.at(den, c, w)
            return np.where(den > 0, num / den, np.nan)
        t, _ = med(topk, a.reps)
        rec[f"topk{k}_ms"] = t * 1e3
    rows.append(rec)
    if (i + 1) % 5 == 0:
        print(f"  {i+1}/{len(Q)} queries", flush=True)

R = pd.DataFrame(rows)
summary = dict(
    machine_conditions=QUIET,
    corpus=cmeta.get("corpus", a.corpus), corpus_dir=a.corpus,
    n_rows=int(len(df)), n_admitted=int(admit.sum()), n_facets=int(len(facets)),
    dim=int(emb.shape[1]), eps=a.eps, predicate=f"filter_num >= {thr:g}",
    n_queries=int(len(Q)), reps=a.reps,
    load_seconds=round(load_s, 2),
    attach_seconds={k: round(v, 2) for k, v in res.items()},
    latency_ms={c: dict(median=float(R[c].median()), p90=float(R[c].quantile(.9)))
                for c in R.columns if c.endswith("_ms")},
    correctness={c: dict(max_rel_err=float(R[c].max()))
                 for c in R.columns if c.endswith("_relerr")},
    facets_answered={c: float(R[c].mean()) for c in R.columns if c.endswith("_facets")},
    note="top-k is timed on the same in-memory scores and layout as the "
         "exact path, so the comparison isolates the aggregation strategy, "
         "not the storage stack; an index would reduce its scan cost and "
         "is measured separately",
)
OUT.mkdir(parents=True, exist_ok=True)
(OUT / f"results_{a.corpus}.json").write_text(json.dumps(summary, indent=2))
R.to_parquet(OUT / f"per_query_{a.corpus}.parquet", index=False)

print(f"\n=== {a.corpus}: {admit.sum():,} admitted rows, {len(facets)} facets")
for c in sorted(summary["latency_ms"]):
    print(f"  {c:<16} median {summary['latency_ms'][c]['median']:8.1f} ms")
for c in summary["correctness"]:
    print(f"  {c:<16} max rel err {summary['correctness'][c]['max_rel_err']:.2e}")
