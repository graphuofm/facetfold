"""What a competent SQL implementation of the operator costs.

The paper's claim that the operator is "unusable expressed naively over
a vector-enabled database" is only interesting if the comparison is
against a competent SQL implementation, not only against the naive one.
A SQL author who knows the log-sum-exp trick writes two passes: a
per-group maximum, then exponentials shifted by it. That is exact and
overflow-free in any engine with window functions or CTEs.

So this measures four things on the same data, the same query, and the
same machine:

  naive    SUM(EXP(s/eps)*v) / SUM(EXP(s/eps)), directly.
  stable   the textbook max-shift, as a window function:
           MAX(s) OVER (PARTITION BY facet) subtracted before
           exponentiating. Exact in real arithmetic.
  guarded  the same, with the exponent clamped so that a shifted
           exponent far below zero contributes 0 rather than raising.
  engine   our fused single-pass kernel, timed in the same process.

The interesting outcome is not that SQL cannot express the operator.
It is that the two engines fail in different places and that the
textbook remedy is not sufficient in one of them: PostgreSQL's float8
exp() raises on overflow AND on underflow, so max-shifting trades an
overflow abort for an underflow abort at sharp temperatures, and only
the explicitly guarded form completes. DuckDB saturates to inf instead
and the ratio becomes NaN, which the max-shift does fix.

What no SQL form offers is what Sections 4 and 5 are about: a state
that survives an update, and a bound on a truncated answer.
"""
import argparse, json, math, statistics, sys, time
import os
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quiet import require_quiet          # noqa: E402
import psycopg2                          # noqa: E402
import duckdb                            # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent

ap = argparse.ArgumentParser()
ap.add_argument("--corpus", default="imdb")
ap.add_argument("--table", default="movies")
ap.add_argument("--min-num", type=float, default=2000)
ap.add_argument("--n-queries", type=int, default=10)
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--eps", type=float, nargs="+", default=[0.5, 0.05, 0.02, 0.001])
ap.add_argument("--no-quiet", action="store_true")
a = ap.parse_args()
if not a.no_quiet:
    require_quiet(wait_seconds=1800)

CORP = ROOT / "corpora" / a.corpus
df = pd.read_parquet(CORP / "corpus.parquet")
emb = np.load(CORP / "emb.npy").astype(np.float64)
adm = df.filter_num.values >= a.min_num
codes, facets = pd.factorize(df.facet.values[adm])
vals = df.value.values[adm].astype(np.float64)
E = emb[adm]
G, N = len(facets), int(adm.sum())
print(f"{a.corpus}: {N:,} admitted, {G} facets", flush=True)

rng = np.random.RandomState(0)
titles = df.text.str.split(".").str[0].str.strip()
ok = titles.str.split().str.len().between(4, 20) & ~titles.duplicated()
qidx = rng.choice(np.flatnonzero(ok.values), size=a.n_queries, replace=False)


def reference(qv, eps):
    s = E @ qv
    m = np.full(G, -np.inf); np.maximum.at(m, codes, s)
    w = np.exp((s - m[codes]) / eps)
    num = np.zeros(G); den = np.zeros(G)
    np.add.at(num, codes, w * vals); np.add.at(den, codes, w)
    return {facets[g]: num[g] / den[g] for g in range(G) if den[g] > 0}


conn = psycopg2.connect(host=os.environ.get("PGHOST", "/tmp"),
                        port=int(os.environ.get("PGPORT", 54329)),
                        user=os.environ.get("PGUSER", os.environ.get("USER", "postgres")),
                        dbname=os.environ.get("PGDATABASE", "postgres"))
conn.autocommit = True
cur = conn.cursor()
cur.execute("SET enable_indexscan = off")   # this is the exact path, not ANN
cur.execute("SET jit = off")
cur.execute("SHOW max_parallel_workers_per_gather")
pg_workers = cur.fetchone()[0]

# DuckDB over the same admitted rows, in memory.
duck = duckdb.connect()
dd = pd.DataFrame({"facet": facets[codes], "value": vals})
duck.register("base", dd)
duck.execute("CREATE TABLE t AS SELECT * FROM base")

SIM_PG = "(1 - (emb <=> %s::vector))"
NAIVE_PG = f"""
SELECT genre, SUM(EXP({SIM_PG}/%s) * rating) / SUM(EXP({SIM_PG}/%s))
FROM {a.table} WHERE year >= %s GROUP BY genre"""
# max-shift as a window function: no join, so a NULL facet is not
# silently dropped the way an inner join against the per-group maxima
# drops it.
SHIFTED_PG = f"""
SELECT genre, {{EXPR}} FROM (
  SELECT genre, rating, {SIM_PG} AS s,
         MAX({SIM_PG}) OVER (PARTITION BY genre) AS m
  FROM {a.table} WHERE year >= %s) q
GROUP BY genre"""
STABLE_PG = SHIFTED_PG.replace(
    "{EXPR}",
    "SUM(EXP((s-m)/%s) * rating) / SUM(EXP((s-m)/%s))")
GUARDED_PG = SHIFTED_PG.replace(
    "{EXPR}",
    "SUM(CASE WHEN (s-m)/%s < -700 THEN 0 ELSE EXP((s-m)/%s) END * rating)"
    " / SUM(CASE WHEN (s-m)/%s < -700 THEN 0 ELSE EXP((s-m)/%s) END)")


def timeit(fn, reps):
    out, ts = None, []
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        ts.append((time.perf_counter() - t0) * 1e3)
    return out, statistics.median(ts)


rows = []
for qn, qi in enumerate(qidx):
    qv = emb[qi]
    vec = "[" + ",".join(f"{x:.8g}" for x in qv) + "]"
    for eps in a.eps:
        ref = reference(qv, eps)

        # precomputed similarities shared by the DuckDB variants, so the
        # comparison is about the aggregation plan and not about who
        # computes the dot product
        s_all = E @ qv
        duck.register("sc", pd.DataFrame({"facet": facets[codes],
                                          "value": vals, "s": s_all}))

        def d_naive():
            return duck.execute(
                f"SELECT facet, SUM(EXP(s/{eps})*value)/SUM(EXP(s/{eps})) "
                "FROM sc GROUP BY facet").fetchall()

        def d_stable():
            return duck.execute(
                "SELECT facet, SUM(EXP((s-m)/%g)*value)/SUM(EXP((s-m)/%g)) "
                "FROM (SELECT facet, value, s, MAX(s) OVER (PARTITION BY facet)"
                " AS m FROM sc) q GROUP BY facet" % (eps, eps)).fetchall()

        def d_guarded():
            e = "CASE WHEN (s-m)/%g < -700 THEN 0 ELSE EXP((s-m)/%g) END" % (eps, eps)
            return duck.execute(
                f"SELECT facet, SUM({e}*value)/SUM({e}) "
                "FROM (SELECT facet, value, s, MAX(s) OVER (PARTITION BY facet)"
                " AS m FROM sc) q GROUP BY facet").fetchall()

        def d_engine():
            """The fused single pass over the SAME precomputed
            similarities the DuckDB variants read, so the comparison
            isolates the aggregation plan from the dot product."""
            m = np.full(G, -np.inf); np.maximum.at(m, codes, s_all)
            w = np.exp((s_all - m[codes]) / eps)
            num = np.zeros(G); den = np.zeros(G)
            np.add.at(num, codes, w * vals); np.add.at(den, codes, w)
            return [(facets[g], num[g] / den[g]) for g in range(G) if den[g] > 0]

        for name, fn, engine in (("naive", d_naive, "duckdb"),
                                 ("stable", d_stable, "duckdb"),
                                 ("guarded", d_guarded, "duckdb"),
                                 ("reference", d_engine, "numpy")):
            try:
                out, ms = timeit(fn, a.reps)
                got = {k: v for k, v in out}
                bad = sum(1 for v in got.values()
                          if v is None or not math.isfinite(v))
                err = max((abs(got[k] - ref[k]) for k in ref
                           if k in got and got[k] is not None
                           and math.isfinite(got[k])), default=float("nan"))
                rows.append(dict(query_i=qn, eps=eps, engine=engine,
                                 form=name, ms=ms, status="ok",
                                 n_groups=len(got), n_nonfinite=bad,
                                 max_abs_err=err))
            except Exception as e:                       # noqa: BLE001
                rows.append(dict(query_i=qn, eps=eps, engine=engine,
                                 form=name, ms=float("nan"),
                                 status=type(e).__name__ + ": " + str(e)[:80],
                                 n_groups=0, n_nonfinite=0,
                                 max_abs_err=float("nan")))

        # PostgreSQL, computing the similarity itself via pgvector
        for name, sql, params in (
                ("naive", NAIVE_PG, (vec, eps, vec, eps, a.min_num)),
                # placeholder order follows the rendered SQL: the
                # aggregate expression is emitted before the subquery
                ("stable", STABLE_PG, (eps, eps, vec, vec, a.min_num)),
                ("guarded", GUARDED_PG,
                 (eps, eps, eps, eps, vec, vec, a.min_num))):
            def run(sql=sql, params=params):
                cur.execute(sql, params)
                return cur.fetchall()
            try:
                out, ms = timeit(run, a.reps)
                got = {k: float(v) for k, v in out if v is not None}
                bad = sum(1 for k, v in out
                          if v is None or not math.isfinite(float(v)))
                err = max((abs(got[k] - ref[k]) for k in ref if k in got),
                          default=float("nan"))
                rows.append(dict(query_i=qn, eps=eps, engine="postgres",
                                 form=name, ms=ms, status="ok",
                                 n_groups=len(got), n_nonfinite=bad,
                                 max_abs_err=err))
            except Exception as e:                       # noqa: BLE001
                conn.rollback() if not conn.autocommit else None
                rows.append(dict(query_i=qn, eps=eps, engine="postgres",
                                 form=name, ms=float("nan"),
                                 status=type(e).__name__ + ": " + str(e)[:120],
                                 n_groups=0, n_nonfinite=0,
                                 max_abs_err=float("nan")))
    print(f"  {qn+1}/{len(qidx)}", flush=True)

R = pd.DataFrame(rows)
agg = (R.groupby(["engine", "form", "eps"])
        .agg(ms_median=("ms", "median"),
             ok_rate=("status", lambda s: float((s == "ok").mean())),
             n_groups=("n_groups", lambda g: float(g[g > 0].median())
                       if (g > 0).any() else 0.0),
             n_nonfinite=("n_nonfinite", "mean"),
             max_abs_err=("max_abs_err", "max"))
        .reset_index())
first_err = {}
for (e, f), g in R[R.status != "ok"].groupby(["engine", "form"]):
    first_err[f"{e}/{f}"] = g.status.iloc[0]

summary = dict(corpus=a.corpus, n_admitted=N, n_facets=G,
               n_queries=int(len(qidx)), reps=a.reps,
               pg_parallel_workers=pg_workers,
               duckdb_version=duckdb.__version__,
               note="DuckDB variants aggregate over precomputed "
                    "similarities so the comparison isolates the "
                    "aggregation plan; PostgreSQL computes the "
                    "similarity itself through pgvector, sequential scan "
                    "forced (this is the exact path, not the ANN path)",
               errors=first_err,
               results=agg.to_dict(orient="records"))
(OUT / f"sql_baselines_{a.corpus}.json").write_text(json.dumps(summary, indent=2))
R.to_parquet(OUT / f"sql_per_query_{a.corpus}.parquet", index=False)
pd.set_option("display.width", 220)
print("\n" + agg.to_string(index=False))
if first_err:
    print("\nfirst failure per engine/form:")
    for k, v in first_err.items():
        print(f"  {k}: {v}")
