"""Where does the time in one certified answer actually go?

Section 7 reports the pieces separately: planning latency, the scan,
the candidate fraction, the repair rate. What it does not show is the
sum, and the sum is what explains why a plan that accumulates 2-4% of
the rows is not 25-50x faster.

This times the four stages of one certified answer on the same query:

  planning        score a 1024-row sketch, walk its mass curve, pick k
  scoring         compute the similarity of every admitted row, which
                  the plan cannot avoid without a certified index
  accumulation    fold the retained prefix of each facet
  repair          re-fold, exactly, each facet whose verified bound
                  missed the budget

Reported as a SHARE of the certified total. Shares are what this
harness can measure reliably: repeated independent runs agree on them
to within a percentage point, while the absolute millisecond figures
move by up to 1.6x when another tenant's job is on the machine. The
absolute cost of one pass is measured separately, under the quiet
guard, in RQ2.

The point of the decomposition is the share of scoring: it is the term
a certified index-backed candidate generator would remove, and until
something removes it the contract cannot pay for itself in time.
"""
import argparse, json, statistics, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from quiet import require_quiet   # noqa: E402
np.seterr(invalid="ignore", divide="ignore")

ap = argparse.ArgumentParser()
ap.add_argument("--corpus", default="imdb")
ap.add_argument("--n-queries", type=int, default=50)
ap.add_argument("--eps", type=float, default=0.02)
ap.add_argument("--budget-frac", type=float, default=0.01)
ap.add_argument("--sketch", type=int, default=1024)
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--no-quiet", action="store_true")
a = ap.parse_args()
QUIET = None if a.no_quiet else require_quiet(wait_seconds=1800)
if a.no_quiet:
    # record the conditions rather than assert they were clean
    from quiet import cpu_busy_fraction, gpu_utilisation  # noqa: E402
    try:
        QUIET = dict(guard="bypassed", cpu_busy_fraction=cpu_busy_fraction(),
                     gpu_utilisation_pct=gpu_utilisation(),
                     note="shares are reported; absolute milliseconds "
                          "from this run are not quoted in the paper")
    except Exception:
        QUIET = dict(guard="bypassed")

CORP = ROOT / "corpora" / a.corpus
df = pd.read_parquet(CORP / "corpus.parquet")
emb = np.load(CORP / "emb.npy").astype(np.float64)
thr = {"amazon": 2015.0, "imdb": 2000.0}.get(a.corpus, 3.0)
adm = df.filter_num.values >= thr
codes, facets = pd.factorize(df.facet.values[adm])
vals = df.value.values[adm].astype(np.float64)
E = np.ascontiguousarray(emb[adm])
G, N = len(facets), int(adm.sum())
rows_of = [np.flatnonzero(codes == g) for g in range(G)]
g_lo = np.array([vals[r].min() if len(r) else np.nan for r in rows_of])
g_hi = np.array([vals[r].max() if len(r) else np.nan for r in rows_of])
col_range = float(vals.max() - vals.min())
beta = a.budget_frac * col_range
srows = np.array([int((i + 0.5) * max(N / a.sketch, 1.0))
                  for i in range(min(a.sketch, N))])
print(f"{a.corpus}: {N:,} admitted, {G} facets, beta={beta:g}", flush=True)

rng = np.random.RandomState(0)
titles = df.text.str.split(".").str[0].str.strip()
ok = titles.str.split().str.len().between(4, 20) & ~titles.duplicated() & adm
qidx = rng.choice(np.flatnonzero(ok.values), size=a.n_queries, replace=False)

recs = []
for qn, qi in enumerate(qidx):
    qv = emb[qi]
    per = {"planning": [], "scoring": [], "accumulation": [], "repair": [],
           "exact": []}
    for _ in range(a.reps):
        # --- planning: the sketch only
        t = time.perf_counter()
        ss = np.sort(E[srows] @ qv)[::-1]
        w = np.exp((ss - ss[0]) / a.eps)
        tail = 1.0 - np.cumsum(w) / w.sum()
        hit = np.flatnonzero(tail * col_range <= beta)
        kstar = (int(hit[0]) + 1) * N // len(srows) if hit.size else N
        per["planning"].append(time.perf_counter() - t)

        # --- scoring: every admitted row, unavoidable without an index
        t = time.perf_counter()
        s = E @ qv
        per["scoring"].append(time.perf_counter() - t)

        # --- accumulation: fold the retained prefix of each facet
        t = time.perf_counter()
        m = np.full(G, -np.inf); np.maximum.at(m, codes, s)
        need = []
        num = np.zeros(G); den = np.zeros(G); full = np.zeros(G)
        for g in range(G):
            idx = rows_of[g]
            if idx.size == 0:
                continue
            k_g = min(max(1, int(np.ceil(kstar * idx.size / N))), idx.size)
            part = idx[np.argpartition(-s[idx], k_g - 1)[:k_g]] if k_g < idx.size else idx
            wg = np.exp((s[part] - m[g]) / a.eps)
            num[g] = (wg * vals[part]).sum(); den[g] = wg.sum()
            wf = np.exp((s[idx] - m[g]) / a.eps)
            full[g] = wf.sum()
            if (1.0 - den[g] / full[g]) * (g_hi[g] - g_lo[g]) > beta:
                need.append(g)
        per["accumulation"].append(time.perf_counter() - t)

        # --- repair: exact re-fold of the facets that missed
        t = time.perf_counter()
        for g in need:
            idx = rows_of[g]
            wf = np.exp((s[idx] - m[g]) / a.eps)
            num[g] = (wf * vals[idx]).sum(); den[g] = wf.sum()
        per["repair"].append(time.perf_counter() - t)

        # --- the exact plan, for reference
        t = time.perf_counter()
        se = E @ qv
        me = np.full(G, -np.inf); np.maximum.at(me, codes, se)
        we = np.exp((se - me[codes]) / a.eps)
        n2 = np.zeros(G); d2 = np.zeros(G)
        np.add.at(n2, codes, we * vals); np.add.at(d2, codes, we)
        per["exact"].append(time.perf_counter() - t)

    rec = {k: statistics.median(v) * 1e3 for k, v in per.items()}
    rec.update(query_i=qn, n_repaired=len(need))
    recs.append(rec)
    if (qn + 1) % 10 == 0:
        print(f"  {qn+1}/{len(qidx)}", flush=True)

R = pd.DataFrame(recs)
stages = ["planning", "scoring", "accumulation", "repair"]
med = {s: float(R[s].median()) for s in stages}
tot = sum(med.values())
summary = dict(machine_conditions=QUIET, corpus=a.corpus, n_admitted=N,
               n_facets=G, eps=a.eps, budget_frac=a.budget_frac,
               budget=beta, n_queries=int(len(qidx)), reps=a.reps,
               sketch=a.sketch,
               note="median milliseconds per stage of one certified "
                    "answer; exact_ms is the single-pass plan for "
                    "reference. Shares are of the certified total.",
               stages_ms=med, certified_total_ms=tot,
               shares={s: med[s] / tot for s in stages},
               exact_ms=float(R["exact"].median()),
               facets_repaired_median=float(R.n_repaired.median()))
(OUT / f"cost_breakdown_{a.corpus}.json").write_text(json.dumps(summary, indent=2))
print(f"\n  {'stage':14s} {'ms':>9s} {'share':>8s}")
for s in stages:
    print(f"  {s:14s} {med[s]:9.3f} {med[s]/tot*100:7.1f}%")
print(f"  {'certified':14s} {tot:9.3f}")
print(f"  {'exact plan':14s} {summary['exact_ms']:9.3f}")
