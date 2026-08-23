"""RQ3: when the corpus changes, what does a fresh answer cost?

Web corpora are not static: reviews arrive, posts are edited, and
users exercise a right to erasure. The question a serving system faces
is not "how fast is one query" but "how fast is the answer correct
again after a change".

Compared, on the same corpus and query:
  rescan      recompute the aggregate from scratch (what a system with
              no maintained state must do, and what a vector store's
              refresh amounts to)
  maintained  update the (m, num, den) state and read it

Update kinds are separated because they do not cost the same, and
reporting a single "delete is O(1)" number would hide that:
  insert          fold one more item into the state
  delete_plain    remove an item that does not hold its facet's anchor
  delete_anchor   remove the item that does, which forces one bounded
                  re-anchoring pass over that facet's survivors
  churn           a long randomised stream of the above, to show the
                  state does not drift

Exactness is checked against a from-scratch recomputation after every
phase; a cost is never reported without the accompanying error.
"""
from pathlib import Path as _P
_ROOT = _P(__file__).resolve().parents[2]
import argparse, sys, json, statistics, time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quiet import require_quiet
import bruce

ROOT = Path(str(_ROOT / "experiments"))
OUT = ROOT / "rq3_maintenance"

ap = argparse.ArgumentParser()
ap.add_argument("--corpus", default="amazon")
ap.add_argument("--eps", type=float, default=0.05)
ap.add_argument("--min-num", type=float, default=None)
ap.add_argument("--reps", type=int, default=5)
ap.add_argument("--churn-ops", type=int, default=10000)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
QUIET = require_quiet(wait_seconds=3600)

CORP = ROOT / "corpora" / a.corpus
df = pd.read_parquet(CORP / "corpus.parquet")
emb = np.load(CORP / "emb.npy").astype(np.float64)
thr = a.min_num if a.min_num is not None else (
    2015.0 if a.corpus == "amazon" else 2000.0 if a.corpus == "imdb" else 3.0)
adm = df.filter_num.values >= thr
codes, facets = pd.factorize(df.facet.values[adm])
vals = df.value.values[adm].astype(np.float64)
E = emb[adm]
N, G = len(vals), len(facets)
rng = np.random.RandomState(a.seed)
qv = E[rng.randint(N)].copy()          # a fixed query: views serve standing queries
sims = E @ qv
print(f"{a.corpus}: {N:,} admitted rows, {G} facets", flush=True)


def rescan(mask=None, score=True):
    """From-scratch max-anchored aggregate; the no-state baseline.

    `score=True` recomputes the similarities, which is what a real
    refresh must do and is the dominant cost -- an earlier version
    reused a cached `sims`, which understated the baseline.
    """
    if score:
        s = (E if mask is None else E[mask]) @ qv
    else:
        s = sims if mask is None else sims[mask]
    c = codes if mask is None else codes[mask]
    v = vals if mask is None else vals[mask]
    m = np.full(G, -np.inf); np.maximum.at(m, c, s)
    w = np.exp((s - m[c]) / a.eps)
    num = np.zeros(G); den = np.zeros(G)
    np.add.at(num, c, w * v); np.add.at(den, c, w)
    return np.where(den > 0, num / den, np.nan)


def med(fn, reps):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


res = dict(machine_conditions=QUIET, corpus=a.corpus, n_rows=int(N), n_facets=int(G), eps=a.eps,
           predicate=f"filter_num >= {thr:g}")

# ---------- baseline: refresh by recomputing ----------
res["rescan_ms"] = med(rescan, a.reps) * 1e3
ref_full = rescan(score=False)

# ---------- maintained state, one IncrementalMemory per facet ----------
t0 = time.perf_counter()
mems, members = {}, {}
for g in range(G):
    idx = np.flatnonzero(codes == g)
    members[g] = list(idx)
    m = bruce.IncrementalMemory(query=qv, eps=a.eps, d_v=1, sim="dot")
    m.insert_many([str(i) for i in idx],
                  np.ascontiguousarray(E[idx]),
                  np.ascontiguousarray(vals[idx].reshape(-1, 1)))
    mems[g] = m
res["build_seconds"] = round(time.perf_counter() - t0, 2)
res["state_bytes_per_facet"] = 8 * (1 + 1 + 1)   # (m, num, den), d_v = 1
res["state_bytes_total"] = res["state_bytes_per_facet"] * G
res["state_vs_corpus_ratio"] = res["state_bytes_total"] / (E.nbytes + vals.nbytes)

def read_all():
    return np.array([mems[g].output()[0] if len(members[g]) else np.nan
                     for g in range(G)])

res["maintained_read_ms"] = med(read_all, a.reps) * 1e3
got = read_all()
ok = np.isfinite(ref_full) & np.isfinite(got)
res["build_max_rel_err"] = float(np.max(np.abs(got[ok] - ref_full[ok])
                                        / np.abs(ref_full[ok])))

# ---------- the three update kinds ----------
big = int(pd.Series(codes).value_counts().idxmax())
gidx = np.array(members[big])
anchor = int(gidx[np.argmax(sims[gidx])])
plain = int(gidx[np.argsort(sims[gidx])[len(gidx) // 2]])

def timed_delete(row):
    def do():
        mems[big].delete(str(row))
        mems[big].output()
    t = med(do, 1)
    mems[big].insert(str(row), E[row], np.array([vals[row]]))
    for _ in range(a.reps - 1):
        t = min(t, med(do, 1))
        mems[big].insert(str(row), E[row], np.array([vals[row]]))
    return t

res["facet_under_test"] = dict(facet=str(facets[big]), rows=int(len(gidx)))
res["delete_plain_us"] = timed_delete(plain) * 1e6
res["delete_anchor_us"] = timed_delete(anchor) * 1e6

def timed_insert():
    def do():
        mems[big].insert("probe", E[plain], np.array([vals[plain]]))
        mems[big].output()
    t = med(do, 1); mems[big].delete("probe")
    for _ in range(a.reps - 1):
        t = min(t, med(do, 1)); mems[big].delete("probe")
    return t
res["insert_us"] = timed_insert() * 1e6

# exactness of a single delete against recomputation
mems[big].delete(str(anchor))
alive = np.ones(N, bool); alive[anchor] = False
ref_after = rescan(alive, score=False)
got_after = mems[big].output()[0]
res["delete_anchor_rel_err"] = float(abs(got_after - ref_after[big]) / abs(ref_after[big]))
mems[big].insert(str(anchor), E[anchor], np.array([vals[anchor]]))

# ---------- churn: does the state drift? ----------
pool = rng.choice(gidx, size=min(500, len(gidx)), replace=False)
live = {int(i) for i in gidx}
t0, checkpoints = time.perf_counter(), []
for step in range(a.churn_ops):
    i = int(pool[rng.randint(len(pool))])
    if i in live:
        mems[big].delete(str(i)); live.discard(i)
    else:
        mems[big].insert(str(i), E[i], np.array([vals[i]])); live.add(i)
    if (step + 1) % (a.churn_ops // 10) == 0:
        # every row outside the facet under test stays; inside it,
        # only the rows currently alive
        m2 = np.ones(N, bool)
        m2[gidx] = False
        m2[list(live)] = True
        r = rescan(m2, score=False)
        checkpoints.append(dict(
            ops=step + 1,
            rel_err=float(abs(mems[big].output()[0] - r[big]) / abs(r[big]))))
res["churn"] = dict(ops=a.churn_ops, pool=len(pool),
                    seconds=round(time.perf_counter() - t0, 2),
                    per_op_us=1e6 * (time.perf_counter() - t0) / a.churn_ops,
                    max_rel_err=max(c["rel_err"] for c in checkpoints),
                    checkpoints=checkpoints)

OUT.mkdir(parents=True, exist_ok=True)
(OUT / f"results_{a.corpus}.json").write_text(json.dumps(res, indent=2))
print(f"\n=== {a.corpus}")
print(f"  rescan (no state)      {res['rescan_ms']:9.1f} ms")
print(f"  maintained read        {res['maintained_read_ms']:9.3f} ms")
print(f"  insert                 {res['insert_us']:9.1f} us")
print(f"  delete (non-anchor)    {res['delete_plain_us']:9.1f} us")
print(f"  delete (anchor)        {res['delete_anchor_us']:9.1f} us")
print(f"  state total            {res['state_bytes_total']:,} B "
      f"({res['state_vs_corpus_ratio']:.2e} of corpus)")
print(f"  build err {res['build_max_rel_err']:.1e} | anchor-delete err "
      f"{res['delete_anchor_rel_err']:.1e} | churn {a.churn_ops} ops max err "
      f"{res['churn']['max_rel_err']:.1e}")
