"""The indexed top-k baseline: what practitioners actually pay.

RQ2's first pass timed top-k without an index, where it must scan every
item anyway and is therefore no cheaper than exact aggregation. That is
not the configuration anyone deploys. This measures top-k served by a
real ANN index (pgvector HNSW), which is why the pattern is popular,
so the paper reports the price of exactness against the strongest
version of the alternative rather than the weakest.

Reported per query: index-served top-k latency at several k, and the
facet coverage of what the index returned -- the same coverage metric
as RQ1, so cost and correctness are read off the same run.
"""
from pathlib import Path as _P
_ROOT = _P(__file__).resolve().parent.parent.parent
import argparse, sys, json, statistics, time
import os
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quiet import require_quiet
import psycopg2

ROOT = Path(str(_ROOT / "experiments") + "")
OUT = ROOT / "rq2_cost"

ap = argparse.ArgumentParser()
ap.add_argument("--table", default="movies")
ap.add_argument("--corpus", default="imdb")
ap.add_argument("--group-col", default="genre")
ap.add_argument("--val-col", default="rating")
ap.add_argument("--filter-col", default="year")
ap.add_argument("--min-num", type=float, default=2000)
ap.add_argument("--n-queries", type=int, default=20)
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--ks", type=int, nargs="+", default=[100, 1000, 10000])
ap.add_argument("--ef-search", type=int, default=200)
ap.add_argument("--iterative", default="off",
                choices=["off", "relaxed_order", "strict_order"],
                help="pgvector 0.8 iterative index scan: the vendor's own "
                     "fix for overfiltering, and the strongest form of "
                     "this baseline")
ap.add_argument("--max-scan-tuples", type=int, default=20000)
ap.add_argument("--tag", default="")
a = ap.parse_args()
QUIET = require_quiet(wait_seconds=3600)

CORP = ROOT / "corpora" / a.corpus
df = pd.read_parquet(CORP / "corpus.parquet")
emb = np.load(CORP / "emb.npy")
rng = np.random.RandomState(0)
titles = df.text.str.split(".").str[0].str.strip()
ok = titles.str.split().str.len().between(4, 20) & ~titles.duplicated()
qidx = rng.choice(np.flatnonzero(ok.values), size=a.n_queries, replace=False)
Q = emb[qidx].astype(np.float64)

admit = df.filter_num.values >= a.min_num
n_facets_gt = df.facet.values[admit]
G = len(pd.unique(n_facets_gt))
print(f"{a.table}: reference has {int(admit.sum()):,} admitted rows, {G} facets",
      flush=True)

conn = psycopg2.connect(host=os.environ.get("PGHOST", "/tmp"),
                        port=int(os.environ.get("PGPORT", 54329)),
                        user=os.environ.get("PGUSER", os.environ.get("USER", "postgres")),
                        dbname=os.environ.get("PGDATABASE", "postgres"))
conn.autocommit = True
cur = conn.cursor()
cur.execute(f"SET hnsw.ef_search = {a.ef_search}")
cur.execute(f"SET hnsw.iterative_scan = {a.iterative}")
if a.iterative != "off":
    cur.execute(f"SET hnsw.max_scan_tuples = {a.max_scan_tuples}")
print(f"pgvector: ef_search={a.ef_search} iterative_scan={a.iterative}"
      + (f" max_scan_tuples={a.max_scan_tuples}" if a.iterative != "off" else ""),
      flush=True)

rows = []
for i, qv in enumerate(Q):
    qs = "[" + ",".join(f"{v:.6f}" for v in qv) + "]"
    rec = dict(query_i=i)
    for k in a.ks:
        # the deployed shape: index-served nearest neighbours, filtered,
        # then grouped and aggregated by the application
        sql = (f"SELECT {a.group_col}, count(*), avg({a.val_col}) FROM ("
               f"  SELECT {a.group_col}, {a.val_col} FROM {a.table}"
               f"  WHERE {a.filter_col} >= {a.min_num:g}"
               f"  ORDER BY emb <#> %s::vector LIMIT {k}) t "
               f"GROUP BY {a.group_col}")
        ts, out = [], None
        for _ in range(a.reps):
            t0 = time.perf_counter()
            cur.execute(sql, (qs,))
            out = cur.fetchall()
            ts.append(time.perf_counter() - t0)
        rec[f"pg_hnsw_topk{k}_ms"] = statistics.median(ts) * 1e3
        rec[f"pg_hnsw_topk{k}_facets"] = len(out)
        rec[f"pg_hnsw_topk{k}_coverage"] = len(out) / G
    rows.append(rec)
    if (i + 1) % 5 == 0:
        print(f"  {i+1}/{len(Q)}", flush=True)

R = pd.DataFrame(rows)
summary = dict(
    machine_conditions=QUIET,
    baseline="PostgreSQL 18 + pgvector 0.8.3, HNSW (vector_ip_ops)",
    ef_search=a.ef_search, iterative_scan=a.iterative,
    max_scan_tuples=(a.max_scan_tuples if a.iterative != "off" else None),
    table=a.table, corpus=a.corpus,
    predicate=f"{a.filter_col} >= {a.min_num:g}",
    n_queries=int(len(Q)), reps=a.reps, n_facets_reference=int(G),
    latency_ms={c: dict(median=float(R[c].median()), p90=float(R[c].quantile(.9)))
                for c in R.columns if c.endswith("_ms")},
    coverage={c: float(R[c].mean()) for c in R.columns if c.endswith("_coverage")},
    note="this is the configuration the top-k pattern exists for: the "
         "index avoids scanning the corpus. Latency here is therefore the "
         "fair opponent for the exact path, and the coverage columns show "
         "what that speed costs in answer completeness.",
)
(OUT / f"results_indexed_{a.corpus}{a.tag}.json").write_text(json.dumps(summary, indent=2))
R.to_parquet(OUT / f"per_query_indexed_{a.corpus}{a.tag}.parquet", index=False)
print(f"\n=== indexed top-k on {a.table} (ef_search={a.ef_search})")
for k in a.ks:
    print(f"  k={k:<6} {summary['latency_ms'][f'pg_hnsw_topk{k}_ms']['median']:7.1f} ms   "
          f"coverage {summary['coverage'][f'pg_hnsw_topk{k}_coverage']:6.1%}")
