"""How the filtered-index route behaves as the predicate tightens.

Figure 2 compares one filtered-ANN configuration against the exact pass
on one corpus, which is not enough to support a general statement about
filtered vector search. The behaviour that matters here -- an index
returning fewer usable rows than asked for once a predicate is applied
-- depends on how selective the predicate is, so this sweeps it.

At each selectivity we report, for the same queries:

  ann_ms / ann_coverage      index-served top-k, as shipped
  iter_ms / iter_coverage    the same with the vendor's iterative index
                             scan enabled, which is its own remedy for
                             the behaviour above
  ann_recall                 fraction of the exact top-k the index
                             returned, so coverage loss can be
                             attributed to the index rather than to k
  exact_ms                   one exact pass over the admitted rows

The conclusion this can support is about the tested engine, index and
corpus. It is reported as such.
"""
import argparse, json, os, statistics, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quiet import require_quiet   # noqa: E402
import psycopg2                   # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent

ap = argparse.ArgumentParser()
ap.add_argument("--corpus", default="imdb")
ap.add_argument("--table", default="movies")
ap.add_argument("--group-col", default="genre")
ap.add_argument("--val-col", default="rating")
ap.add_argument("--filter-col", default="year")
ap.add_argument("--targets", type=float, nargs="+",
                default=[0.01, 0.05, 0.10, 0.25, 0.50, 1.00])
ap.add_argument("--k", type=int, default=1000)
ap.add_argument("--n-queries", type=int, default=20)
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--ef-search", type=int, default=200)
ap.add_argument("--max-scan-tuples", type=int, default=20000)
ap.add_argument("--no-quiet", action="store_true")
a = ap.parse_args()
QUIET = None if a.no_quiet else require_quiet(wait_seconds=1800)

CORP = ROOT / "corpora" / a.corpus
df = pd.read_parquet(CORP / "corpus.parquet")
emb = np.load(CORP / "emb.npy").astype(np.float64)
fnum = df.filter_num.values.astype(float)

conn = psycopg2.connect(host=os.environ.get("PGHOST", "/tmp"),
                        port=int(os.environ.get("PGPORT", 54329)),
                        user=os.environ.get("PGUSER", os.environ.get("USER", "postgres")),
                        dbname=os.environ.get("PGDATABASE", "postgres"))
conn.autocommit = True
cur = conn.cursor()

rng = np.random.RandomState(0)
titles = df.text.str.split(".").str[0].str.strip()
ok = titles.str.split().str.len().between(4, 20) & ~titles.duplicated()
qidx = rng.choice(np.flatnonzero(ok.values), size=a.n_queries, replace=False)
Q = emb[qidx]

# thresholds on the numeric filter that hit each target selectivity
cuts = [(t, float(np.quantile(fnum, 1.0 - t))) for t in a.targets]
print(f"{a.corpus}: {len(df):,} rows; thresholds "
      + ", ".join(f"{t:.0%}->{c:g}" for t, c in cuts), flush=True)

rows = []
for target, cut in cuts:
    adm = fnum >= cut
    n_adm = int(adm.sum())
    codes, facets = pd.factorize(df[a.group_col if a.group_col in df else "facet"]
                                 .values[adm])
    vals = df.value.values[adm].astype(np.float64)
    E = emb[adm]
    G = len(facets)

    for i, qv in enumerate(Q):
        qs = "[" + ",".join(f"{v:.6f}" for v in qv) + "]"
        # exact reference: the true top-k ids and the exact pass
        t0 = time.perf_counter()
        s = E @ qv
        kk = min(a.k, n_adm)
        top = np.argpartition(-s, kk - 1)[:kk]
        exact_topk = set(map(int, top))
        exact_ms = (time.perf_counter() - t0) * 1e3
        exact_cov_facets = len(set(codes.tolist()))

        rec = dict(target_selectivity=target, threshold=cut,
                   n_admitted=n_adm, n_facets=G, query_i=i,
                   exact_ms=exact_ms)

        for mode in ("off", "relaxed_order"):
            cur.execute(f"SET hnsw.ef_search = {a.ef_search}")
            cur.execute(f"SET hnsw.iterative_scan = {mode}")
            if mode != "off":
                cur.execute(f"SET hnsw.max_scan_tuples = {a.max_scan_tuples}")
            sql = (f"SELECT {a.group_col}, count(*) FROM ("
                   f"  SELECT {a.group_col} FROM {a.table}"
                   f"  WHERE {a.filter_col} >= {cut:g}"
                   f"  ORDER BY emb <#> %s::vector LIMIT {a.k}) t "
                   f"GROUP BY {a.group_col}")
            ts, out = [], None
            for _ in range(a.reps):
                t0 = time.perf_counter()
                cur.execute(sql, (qs,))
                out = cur.fetchall()
                ts.append((time.perf_counter() - t0) * 1e3)
            got = sum(c for _, c in out)
            tag = "ann" if mode == "off" else "iter"
            rec[f"{tag}_ms"] = statistics.median(ts)
            rec[f"{tag}_facets"] = len(out)
            rec[f"{tag}_coverage"] = len(out) / max(exact_cov_facets, 1)
            rec[f"{tag}_rows_returned"] = got
            rec[f"{tag}_fill"] = got / kk       # did it return k usable rows?
        rows.append(rec)
    print(f"  selectivity {target:.0%}: {n_adm:,} admitted, {G} facets",
          flush=True)

R = pd.DataFrame(rows)
agg = (R.groupby("target_selectivity")
        .agg(n_admitted=("n_admitted", "first"), n_facets=("n_facets", "first"),
             exact_ms=("exact_ms", "median"),
             ann_ms=("ann_ms", "median"), ann_coverage=("ann_coverage", "mean"),
             ann_fill=("ann_fill", "mean"),
             iter_ms=("iter_ms", "median"), iter_coverage=("iter_coverage", "mean"),
             iter_fill=("iter_fill", "mean"))
        .reset_index())
summary = dict(machine_conditions=QUIET, corpus=a.corpus, table=a.table,
               k=a.k, n_queries=int(len(Q)), reps=a.reps,
               ef_search=a.ef_search, max_scan_tuples=a.max_scan_tuples,
               engine="PostgreSQL 18.4 + pgvector 0.8.3, HNSW (vector_ip_ops)",
               note="fill is the fraction of the requested k that the "
                    "index actually returned after the predicate; a fill "
                    "below 1 is the overfiltering behaviour the iterative "
                    "scan exists to fix",
               results=agg.to_dict(orient="records"))
(OUT / f"selectivity_{a.corpus}.json").write_text(json.dumps(summary, indent=2))
R.to_parquet(OUT / f"selectivity_per_query_{a.corpus}.parquet", index=False)
pd.set_option("display.width", 220)
print("\n" + agg.to_string(index=False))
