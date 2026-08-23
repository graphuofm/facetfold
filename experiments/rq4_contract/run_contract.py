"""RQ4: approximation that keeps a promise.

RQ1 showed that truncating retrieval silently drops whole facets. The
operator's answer is not "never approximate" -- it is "approximate only
under a contract the system can check". The query declares an absolute
error budget; the planner certifies a truncation that meets it, or
refuses; and execution re-folds any facet whose realised bound misses,
so no facet is ever dropped.

The comparison that matters is at EQUAL retrieval budget: given the
same k, does the plan answer every facet within the promise, or does it
quietly answer some and lose the rest?

Per query and budget we record: the k the planner certified, the
realised maximum error, whether the promise held, facet coverage, and
-- for the same k -- what plain top-k would have returned.
"""
from pathlib import Path as _P
_ROOT = _P(__file__).resolve().parents[2]
import argparse, json, statistics, time
from pathlib import Path

import numpy as np
import pandas as pd
from bruce import QuerySession

ROOT = Path(str(_ROOT / "experiments"))
OUT = ROOT / "rq4_contract"

ap = argparse.ArgumentParser()
ap.add_argument("--corpus", default="imdb")
ap.add_argument("--n-queries", type=int, default=50)
ap.add_argument("--eps", type=float, default=0.02)
ap.add_argument("--budgets", type=float, nargs="+", default=[0.5, 0.2, 0.05, 0.01])
ap.add_argument("--min-num", type=float, default=None)
a = ap.parse_args()

CORP = ROOT / "corpora" / a.corpus
df = pd.read_parquet(CORP / "corpus.parquet")
emb = np.load(CORP / "emb.npy").astype(np.float64)
thr = a.min_num if a.min_num is not None else (
    2015.0 if a.corpus == "amazon" else 2000.0 if a.corpus == "imdb" else 3.0)
adm = df.filter_num.values >= thr
codes, facets = pd.factorize(df.facet.values[adm])
vals = df.value.values[adm].astype(np.float64)
E = emb[adm]
G = len(facets)
print(f"{a.corpus}: {int(adm.sum()):,} admitted, {G} facets", flush=True)

rng = np.random.RandomState(0)
titles = df.text.str.split(".").str[0].str.strip()
ok = titles.str.split().str.len().between(4, 20) & ~titles.duplicated()
qidx = rng.choice(np.flatnonzero(ok.values), size=a.n_queries, replace=False)

sess = QuerySession()
sess.register_parquet(a.corpus, str(CORP / "corpus.parquet"))
sess.attach_key(a.corpus, "emb", emb)
WHERE = f"WHERE filter_num >= {thr:g}"


def exact_ref(qv):
    s = E @ qv
    m = np.full(G, -np.inf); np.maximum.at(m, codes, s)
    w = np.exp((s - m[codes]) / a.eps)
    num = np.zeros(G); den = np.zeros(G)
    np.add.at(num, codes, w * vals); np.add.at(den, codes, w)
    return np.where(den > 0, num / den, np.nan)


def plain_topk(qv, k):
    """What truncation alone returns at the same budget."""
    s = E @ qv
    idx = np.argpartition(-s, min(k, len(s) - 1))[:k]
    c, sv = codes[idx], s[idx]
    m = np.full(G, -np.inf); np.maximum.at(m, c, sv)
    w = np.exp((sv - m[c]) / a.eps)
    num = np.zeros(G); den = np.zeros(G)
    np.add.at(num, c, w * vals[idx]); np.add.at(den, c, w)
    return np.where(den > 0, num / den, np.nan)


rows = []
for i, qi in enumerate(qidx):
    qv = emb[qi]
    ref = exact_ref(qv)
    ok_ref = np.isfinite(ref)
    for b in a.budgets:
        sql = (f"SELECT facet, SOFTAVG(value, SIM(emb, :q), {a.eps}, {b}) "
               f"FROM {a.corpus} {WHERE} GROUP BY facet")
        t0 = time.perf_counter()
        out = sess.run(sql, {"q": qv})
        ms = (time.perf_counter() - t0) * 1e3
        labels, values, explain = out[0], out[1], str(out[2])
        got = {l: v for l, v in zip(labels, values)}
        errs = [abs(got[f] - ref[j]) for j, f in enumerate(facets)
                if f in got and ok_ref[j]]
        chosen = ("contract" if "TopKContractScan[" in explain.split("== candidates ==")[0]
                  else "exact")
        # EXPLAIN prints k* twice: in the chosen-plan block and again
        # inside the candidate line's parenthesised note, so strip any
        # trailing punctuation and take the first occurrence.
        kstar = None
        import re as _re
        mk = _re.search(r"k\*=(\d+)", explain)
        if mk:
            kstar = int(mk.group(1))
        rec = dict(query_i=i, budget=b, plan=chosen, kstar=kstar, ms=ms,
                   max_abs_err=float(max(errs)) if errs else np.nan,
                   promise_held=bool(max(errs) <= b) if errs else True,
                   coverage=len(got) / int(ok_ref.sum()))
        if kstar:   # same retrieval budget, truncation only
            tk = plain_topk(qv, kstar)
            ok_tk = np.isfinite(tk)
            both = ok_ref & ok_tk
            rec["topk_same_k_coverage"] = float(ok_tk.sum() / ok_ref.sum())
            rec["topk_same_k_max_abs_err"] = (
                float(np.max(np.abs(tk[both] - ref[both]))) if both.any() else np.nan)
            rec["topk_same_k_promise_held"] = bool(
                ok_tk.sum() == ok_ref.sum()
                and np.max(np.abs(tk[both] - ref[both])) <= b)
        rows.append(rec)
    if (i + 1) % 10 == 0:
        print(f"  {i+1}/{len(qidx)}", flush=True)

R = pd.DataFrame(rows)
agg = (R.groupby("budget")
        .agg(plan_contract=("plan", lambda s: float((s == "contract").mean())),
             kstar_median=("kstar", "median"),
             ms_median=("ms", "median"),
             max_abs_err_p100=("max_abs_err", "max"),
             promise_held=("promise_held", "mean"),
             coverage=("coverage", "mean"),
             topk_coverage=("topk_same_k_coverage", "mean"),
             topk_err_p100=("topk_same_k_max_abs_err", "max"),
             topk_promise_held=("topk_same_k_promise_held", "mean"))
        .reset_index())
summary = dict(corpus=a.corpus, n_admitted=int(adm.sum()), n_facets=int(G),
               eps=a.eps, n_queries=int(len(qidx)),
               predicate=f"filter_num >= {thr:g}",
               comparison="at the k the planner certified, plain truncation "
                          "is evaluated on the same data and query",
               results=agg.to_dict(orient="records"))
OUT.mkdir(parents=True, exist_ok=True)
(OUT / f"results_{a.corpus}.json").write_text(json.dumps(summary, indent=2))
R.to_parquet(OUT / f"per_query_{a.corpus}.parquet", index=False)
pd.set_option("display.width", 220)
print("\n" + agg.to_string(index=False))
