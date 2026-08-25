"""How maintenance scales with the number of standing queries.

A maintained state answers one query embedding, so the per-edit cost
reported in RQ3 is the cost at Q=1. What decides whether materialising
these answers is affordable is how many standing queries a deployment
keeps, because an inserted or deleted row must be folded into the state
of every one of them.

An edit belongs to exactly one facet, so it touches Q states, not Q x G.
This harness builds Q maintained states over one facet, through the
same engine path RQ3 uses, and times an insert and an ordinary delete
as Q grows. Reported alongside: the accumulator payload for the whole
answer (Q x G x (3 doubles + an anchor-multiplicity counter)) and the
cost of recomputing all Q answers with one batched pass, which is what
a system without maintained state pays per refresh.

The anchor-deletion case is not swept here: its cost is set by the size
of the facet rather than by Q, and RQ3 measures it against facet size.
"""
import argparse, json, statistics, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import bruce

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quiet import require_quiet  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent

ap = argparse.ArgumentParser()
ap.add_argument("--corpus", default="amazon")
ap.add_argument("--qs", type=int, nargs="+", default=[1, 10, 100, 1000])
ap.add_argument("--eps", type=float, default=0.05)
ap.add_argument("--facet-rows", type=int, default=2000)
ap.add_argument("--reps", type=int, default=200)
ap.add_argument("--no-quiet", action="store_true")
a = ap.parse_args()
QUIET = None if a.no_quiet else require_quiet(wait_seconds=1800)

CORP = ROOT / "corpora" / a.corpus
df = pd.read_parquet(CORP / "corpus.parquet")
emb = np.load(CORP / "emb.npy").astype(np.float64)
thr = {"amazon": 2015.0, "imdb": 2000.0}.get(a.corpus, 3.0)
adm = df.filter_num.values >= thr
codes, facets = pd.factorize(df.facet.values[adm])
vals = df.value.values[adm].astype(np.float64)
E = emb[adm]
G, N, d = len(facets), int(adm.sum()), emb.shape[1]
print(f"{a.corpus}: {N:,} admitted, {G} facets, d={d}", flush=True)

rng = np.random.RandomState(0)
QPOOL = emb[rng.choice(len(emb), size=max(a.qs), replace=False)].astype(np.float64)
g0 = int(pd.Series(codes).value_counts().idxmax())
pool = np.flatnonzero(codes == g0)
pool = rng.choice(pool, size=min(a.facet_rows, len(pool)), replace=False)
KEYS = np.ascontiguousarray(E[pool])
VALS = np.ascontiguousarray(vals[pool].reshape(-1, 1))
IDS = [str(i) for i in pool]
print(f"  facet {facets[g0]!r}, {len(pool)} rows held in each state", flush=True)


def med(fn, reps):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1e6


rows = []
for Q in a.qs:
    t0 = time.perf_counter()
    mems = []
    for qi in range(Q):
        m = bruce.IncrementalMemory(query=QPOOL[qi], eps=a.eps, d_v=1, sim="dot")
        m.insert_many(IDS, KEYS, VALS)
        mems.append(m)
    build_s = time.perf_counter() - t0

    probe = int(pool[len(pool) // 2])
    pk, pv = E[probe], np.array([vals[probe]])

    def do_insert():
        for m in mems:
            m.insert("probe", pk, pv)

    def do_delete():
        for m in mems:
            m.delete("probe")

    pair_us = med(lambda: (do_insert(), do_delete()), a.reps)
    do_insert()
    del_us = med(lambda: (do_delete(), do_insert()), a.reps) / 2
    do_delete()
    ins_us = max(pair_us - del_us, 0.0)

    read_us = med(lambda: [m.output()[0] for m in mems], max(20, a.reps // 10))

    # what a system without maintained state pays per refresh: one
    # batched pass over every admitted row for all Q queries. The
    # (N x Q) score matrix is materialised, which is what makes the
    # single pass a single pass; past a few hundred standing queries it
    # no longer fits and a tiled rebuild must score every row twice, so
    # we report the cost only where the honest single-pass form runs
    # and record where it stopped fitting.
    QE = QPOOL[:Q]
    fits = len(E) * Q * 8 < 4e9
    if fits:
        t0 = time.perf_counter()
        S = E @ QE.T
        mx = np.full((G, Q), -np.inf); np.maximum.at(mx, codes, S)
        W = np.exp((S - mx[codes]) / a.eps)
        num = np.zeros((G, Q)); den = np.zeros((G, Q))
        np.add.at(num, codes, W * vals[:, None]); np.add.at(den, codes, W)
        rebuild_ms = (time.perf_counter() - t0) * 1e3
        del S, W, num, den
    else:
        rebuild_ms = float("nan")

    # 40 bytes per (query, facet): three f64 accumulators plus the
    # live-row and anchor-multiplicity counters, which is
    # size_of::<GroupState>() in the engine.
    payload = G * Q * 40
    rec = dict(corpus=a.corpus, Q=Q, n_facets=G, n_admitted=N,
               facet_rows=int(len(pool)), build_seconds=round(build_s, 2),
               insert_us=ins_us, delete_us=del_us,
               insert_us_per_query=ins_us / Q, read_us=read_us,
               rebuild_all_ms=rebuild_ms,
               rebuild_fits_in_memory=bool(fits),
               payload_bytes=int(payload),
               breakeven_edits_per_rebuild=(rebuild_ms * 1e3 / max(ins_us, 1e-9)
                                            if fits else float("nan")))
    rows.append(rec)
    print(f"  Q={Q:>5} ins={ins_us:9.2f}us ({ins_us/Q:6.3f}/query) "
          f"del={del_us:9.2f}us read={read_us:9.1f}us "
          f"rebuild={rebuild_ms:8.1f}ms payload={payload/1024:.1f}KiB",
          flush=True)
    del mems

summary = dict(machine_conditions=QUIET, corpus=a.corpus, n_admitted=N,
               n_facets=G, dim=d, eps=a.eps, reps=a.reps,
               note="per-edit cost of maintaining Q standing queries, "
                    "measured through the same engine path as RQ3. "
                    "Payload is 40 B per (query, facet), the engine's "
                    "GroupState. Aggregate-state only: excludes encoding the text, "
                    "writing the base relation, and vector-index "
                    "maintenance.",
               results=rows)
(OUT / f"qscaling_{a.corpus}.json").write_text(json.dumps(summary, indent=2))
print("\n" + pd.DataFrame(rows)[
    ["Q", "insert_us", "insert_us_per_query", "delete_us", "read_us",
     "rebuild_all_ms", "payload_bytes"]].to_string(index=False))
