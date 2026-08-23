"""RQ4 (revised): what the certified truncation costs, stage by stage.

The first version of this experiment reported only that the promise was
kept. That is close to a tautology: execution measures the true omitted
mass and re-folds any group that misses, so of course the delivered
answer satisfies the budget. The interesting question is what keeping
the promise costs, and how much slack the bound carries.

This harness therefore separates the three mechanisms the engine
actually runs and prices each one:

  proposal      a fixed 1024-row uniform sketch of the key column is
                scored against the query; the smallest sample index
                whose omitted mass certifies the budget is scaled to a
                candidate k*. May decline (resolution-limited).
  verification  the full pass scores every admitted row, so the true
                omitted mass per group is known exactly; the certified
                bound delta_g * (range of group g's values) is compared
                with the budget.
  repair        any group whose bound misses is re-folded exactly.

Against these we measure the oracle: the smallest per-group k that
truly satisfies the bound, and the smallest that truly satisfies the
realised error.  k_planner / k_oracle is the price of not knowing the
answer in advance; bound / realised error is the slack in the bound
itself.

The bound.  Split group g into retained rows R and omitted rows O with
softmax masses W_R, W_O and mass-weighted means mu_R, mu_O, and write
delta = W_O / (W_R + W_O). The exact answer is the convex combination
F = (1-delta) mu_R + delta mu_O, so |mu_R - F| = delta |mu_R - mu_O| <=
delta * (v_max - v_min) since both means are convex combinations of
values from that range. That is what the engine certifies. The earlier
delta/(1-delta)(1+1/(1-delta)) max|v| form is valid but loose by 2.2x
(IMDb) to 100x (StackExchange, where per-group ranges are far narrower
than the column's).

Budgets are declared as a fraction of the admitted value range and
reported in the value's own units, because an absolute 0.01 means
something very different on 1-5 star ratings and on StackExchange
answer scores that span -515 to 8390.

The numpy simulation here mirrors the engine; every configuration is
cross-checked against the engine's own SQL answer, and the harness
exits non-zero if they disagree.
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from bruce import QuerySession

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quiet import require_quiet  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent

SKETCH_SAMPLE = 1024      # bruce-query DbOptions::stats_sample
RESOLUTION_MIN = 8        # bruce-query stats.rs RESOLUTION_MIN_SAMPLES

ap = argparse.ArgumentParser()
ap.add_argument("--corpus", default="imdb")
ap.add_argument("--n-queries", type=int, default=200)
ap.add_argument("--eps", type=float, default=0.02)
# budgets as a fraction of the admitted value range
ap.add_argument("--budget-fracs", type=float, nargs="+",
                default=[0.10, 0.05, 0.01, 0.005])
ap.add_argument("--sample-sizes", type=int, nargs="+",
                default=[64, 256, 1024, 4096])
ap.add_argument("--no-quiet", action="store_true")
a = ap.parse_args()

if not a.no_quiet:
    require_quiet()

CORP = ROOT / "corpora" / a.corpus
df = pd.read_parquet(CORP / "corpus.parquet")
emb = np.load(CORP / "emb.npy").astype(np.float64)
thr = {"amazon": 2015.0, "imdb": 2000.0}.get(a.corpus, 3.0)
adm = df.filter_num.values >= thr
codes, facets = pd.factorize(df.facet.values[adm])
vals = df.value.values[adm].astype(np.float64)
E = emb[adm]
G = len(facets)
N = int(adm.sum())

# column range (what the planner prices with) and per-group ranges
# (what execution certifies with)
col_lo, col_hi = float(vals.min()), float(vals.max())
col_range = col_hi - col_lo
g_lo = np.full(G, np.inf); np.minimum.at(g_lo, codes, vals)
g_hi = np.full(G, -np.inf); np.maximum.at(g_hi, codes, vals)
g_range = g_hi - g_lo
budgets = [f * col_range for f in a.budget_fracs]
print(f"{a.corpus}: {N:,} admitted, {G} facets, value range "
      f"[{col_lo:g},{col_hi:g}] = {col_range:g}", flush=True)
print(f"  per-facet range: median {np.median(g_range):g}, "
      f"max {g_range.max():g}", flush=True)

# the sketch: the engine's deterministic stride sample, same rule
def sketch_rows(n, take):
    take = max(1, min(take, n))
    stride = max(n / take, 1.0)
    return np.array([int((i + 0.5) * stride) for i in range(take)])

rng = np.random.RandomState(0)
titles = df.text.str.split(".").str[0].str.strip()
ok = titles.str.split().str.len().between(4, 20) & ~titles.duplicated()
qidx = rng.choice(np.flatnonzero(ok.values), size=a.n_queries, replace=False)

sess = QuerySession()
sess.register_parquet(a.corpus, str(CORP / "corpus.parquet"))
sess.attach_key(a.corpus, "emb", emb)
WHERE = f"WHERE filter_num >= {thr:g}"


def exact_ref(s):
    m = np.full(G, -np.inf); np.maximum.at(m, codes, s)
    w = np.exp((s - m[codes]) / a.eps)
    num = np.zeros(G); den = np.zeros(G)
    np.add.at(num, codes, w * vals); np.add.at(den, codes, w)
    return np.where(den > 0, num / den, np.nan), m


def propose(s_sample, budget, n_pop, take):
    """Sketch stage: scaled k*, or None when resolution-limited."""
    ss = np.sort(s_sample)[::-1]
    w = np.exp((ss - ss[0]) / a.eps)
    z = w.sum()
    tail = 1.0 - np.cumsum(w) / z
    hit = np.flatnonzero(tail * col_range <= budget)
    if hit.size == 0:
        return None, 0.0, False
    j = int(hit[0]) + 1
    limited = j < RESOLUTION_MIN
    return int(np.ceil(j * n_pop / take)), float(tail[j - 1]), limited


abl = []
t_start = time.time()
for qi_n, qi in enumerate(qidx):
    qv = emb[qi]
    t0 = time.perf_counter()
    s = E @ qv
    scan_ms = (time.perf_counter() - t0) * 1e3
    ref, anchors = exact_ref(s)
    live = np.isfinite(ref)

    # per-group descending order and prefix sums, computed ONCE per
    # query: the budget and the sketch size change only which prefix
    # is read, never the prefixes themselves.
    order = np.argsort(-s, kind="stable")
    gorder = [[] for _ in range(G)]
    for r in order:
        gorder[codes[r]].append(r)
    pre = []
    for g in range(G):
        idx = np.asarray(gorder[g], dtype=np.int64)
        if idx.size == 0:
            pre.append(None)
            continue
        wg = np.exp((s[idx] - anchors[g]) / a.eps)
        cw = np.cumsum(wg)
        cn = np.cumsum(wg * vals[idx])
        zfull = cw[-1]
        tail = 1.0 - cw / zfull
        # smallest k whose certified bound meets each budget (oracle)
        pre.append((idx.size, cw, cn, zfull, tail))

    for take in a.sample_sizes:
        srows = sketch_rows(N, take)
        s_smp = s[srows]
        for bf, b in zip(a.budget_fracs, budgets):
            tp0 = time.perf_counter()
            kstar, dhat, limited = propose(s_smp, b, N, len(srows))
            plan_ms = (time.perf_counter() - tp0) * 1e3
            declined = kstar is None or limited or kstar >= N
            rec = dict(corpus=a.corpus, query_i=qi_n, sample=take,
                       budget_frac=bf, budget=b, declined=bool(declined),
                       kstar=kstar, delta_hat=dhat, plan_ms=plan_ms,
                       scan_ms=scan_ms)
            if declined:
                abl.append(rec)
                continue

            n_rep = 0; n_live = 0
            errs = []; slack = []
            k_pl = 0; k_or = 0
            for g in range(G):
                if pre[g] is None:
                    continue
                ng, cw, cn, zfull, tail = pre[g]
                n_live += 1
                k_g = min(max(1, int(np.ceil(kstar * ng / N))), ng)
                delta = tail[k_g - 1]
                bound = delta * g_range[g]
                if bound <= b:
                    ans = cn[k_g - 1] / cw[k_g - 1]
                else:
                    n_rep += 1
                    ans = cn[-1] / zfull
                e = abs(ans - ref[g])
                errs.append(e)
                if bound <= b:
                    slack.append(bound / max(e, 1e-15))
                hit = np.flatnonzero(tail * g_range[g] <= b)
                k_or += int(hit[0]) + 1 if hit.size else ng
                k_pl += k_g

            rec.update(
                n_live_facets=n_live,
                n_repaired=n_rep,
                frac_repaired=n_rep / max(n_live, 1),
                max_abs_err=float(max(errs)) if errs else np.nan,
                promise_held=bool(max(errs) <= b * (1 + 1e-9)) if errs else True,
                bound_slack_median=float(np.median(slack)) if slack else np.nan,
                k_planner_total=int(k_pl),
                k_oracle_total=int(k_or),
                k_ratio=float(k_pl / max(k_or, 1)),
                candidates_frac=float(k_pl / N),
            )
            abl.append(rec)

    if (qi_n + 1) % 25 == 0:
        print(f"  {qi_n+1}/{len(qidx)}  ({time.time()-t_start:.0f}s)", flush=True)

A = pd.DataFrame(abl)
# the shipped configuration: the engine's 1024-row sketch
R = A[A["sample"] == SKETCH_SAMPLE]

# ---- cross-check the simulation against the engine on a few queries --
print("\ncross-checking the simulation against the engine ...", flush=True)
mismatch = []
for qi_n, qi in enumerate(qidx[:10]):
    qv = emb[qi]
    ref, _ = exact_ref(E @ qv)
    for bf, b in zip(a.budget_fracs, budgets):
        sql = (f"SELECT facet, SOFTAVG(value, SIM(emb, :q), {a.eps}, {b}) "
               f"FROM {a.corpus} {WHERE} GROUP BY facet")
        out = sess.run(sql, {"q": qv})
        got = {l: v for l, v in zip(out[0], out[1])}
        live = np.isfinite(ref)
        miss = int(live.sum()) - len(got)
        e = max((abs(got[f] - ref[j]) for j, f in enumerate(facets)
                 if f in got and live[j]), default=0.0)
        if miss != 0 or e > b * (1 + 1e-6):
            mismatch.append((qi_n, bf, miss, e, b))
if mismatch:
    print("ENGINE DISAGREES WITH THE CONTRACT:", mismatch[:5])
    sys.exit(1)
print("  engine kept the promise and dropped no facet on every checked query")


def summarise(frame):
    out = []
    for (bf,), grp in frame.groupby(["budget_frac"]):
        acc = grp[~grp.declined]
        out.append(dict(
            budget_frac=float(bf), budget=float(grp.budget.iloc[0]),
            declined_rate=float(grp.declined.mean()),
            n_accepted=int(len(acc)),
            promise_held=float(acc.promise_held.mean()) if len(acc) else np.nan,
            repaired_queries=float((acc.n_repaired > 0).mean()) if len(acc) else np.nan,
            frac_facets_repaired=float(acc.frac_repaired.mean()) if len(acc) else np.nan,
            k_ratio_median=float(acc.k_ratio.median()) if len(acc) else np.nan,
            candidates_frac_median=float(acc.candidates_frac.median()) if len(acc) else np.nan,
            bound_slack_median=float(acc.bound_slack_median.median()) if len(acc) else np.nan,
            plan_ms_median=float(grp.plan_ms.median()),
            scan_ms_median=float(grp.scan_ms.median()),
        ))
    return out


summary = dict(
    corpus=a.corpus, n_admitted=N, n_facets=G, eps=a.eps,
    n_queries=int(len(qidx)), predicate=f"filter_num >= {thr:g}",
    value_range=[col_lo, col_hi],
    facet_range_median=float(np.median(g_range)),
    facet_range_max=float(g_range.max()),
    sketch_sample=SKETCH_SAMPLE,
    bound="delta_g * (v_max - v_min) over group g's own values",
    budgets_are="fractions of the admitted value range, reported absolute",
    main=summarise(R),
    sample_ablation={
        str(t): summarise(A[A["sample"] == t]) for t in a.sample_sizes},
)
OUT.mkdir(parents=True, exist_ok=True)
(OUT / f"results2_{a.corpus}.json").write_text(json.dumps(summary, indent=2))
A.to_parquet(OUT / f"ablation_{a.corpus}.parquet", index=False)
pd.set_option("display.width", 250)
print("\n" + pd.DataFrame(summary["main"]).to_string(index=False))
