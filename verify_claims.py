"""Check every numeric claim in the paper against the result files.

A paper's numbers drift from its data during editing, and a reviewer
who spots one stops trusting the rest. Each claim below names the file
and field it must match; a mismatch is printed rather than tolerated.
Run before every submission build.

The ABSTRACT is checked here too. An earlier draft carried a stale
coverage figure in its first paragraph for weeks, because this script
only read main.tex and the abstract lives in its own file. Both are
loaded now.
"""
import json, re, sys

import numpy as np
import pandas as pd
from pathlib import Path

E = Path("experiments")
# The paper source is not part of this artifact, so the text checks
# (claims that must appear verbatim, stale claims that must not) are
# skipped when it is absent. Every NUMERIC check below runs either way,
# against the result files in experiments/.
_SRC = [Path(f) for f in ("main.tex", "abstract_body.tex")]
HAVE_TEXT = all(f.exists() for f in _SRC)
TXT = "".join(f.read_text() for f in _SRC) if HAVE_TEXT else ""
fails, checks = [], 0


def load(p):
    return json.load(open(E / p))


def check(desc, claimed, actual, tol=0.05, unit=""):
    """Claimed value must be within `tol` relative of the measurement.

    Latency tolerances are deliberately wide. Repeated guarded runs on
    this machine still move a latency by 20-30% between sessions, so a
    tighter check would fail on measurement noise rather than on a
    drifted claim. Coverage and error figures are deterministic and are
    checked tightly.
    """
    global checks
    checks += 1
    ok = actual != 0 and abs(claimed - actual) / abs(actual) <= tol
    if not ok:
        fails.append(f"{desc}: paper says {claimed}{unit}, data says {actual:.4g}{unit}")
    return ok


def in_text(s):
    """Only meaningful when the paper source is present."""
    global checks
    if not HAVE_TEXT:
        return
    checks += 1
    if s not in TXT:
        fails.append(f"missing from paper text: {s!r}")


# ---- corpora ----
for d, n_rows, n_facets in [("amazon", 1524113, 33), ("stackexchange", 1200000, 152),
                            ("imdb", 459865, 29)]:
    m = json.load(open(E / "corpora" / d / "meta.json"))
    check(f"{d} rows", n_rows, m["n_rows"], 0.001)
    check(f"{d} facets", n_facets, m["n_facets"], 0.001)

# ---- RQ0 retriever ----
r0 = load("rq0_retriever/results.json")
nd = {k.split("/")[-1]: v for k, v in r0["results"].items()}
check("MiniLM nDCG@10", 0.739, nd["all-MiniLM-L6-v2"]["ndcg@10"], 0.01)
check("MPNet nDCG@10", 0.733, nd["all-mpnet-base-v2"]["ndcg@10"], 0.01)

# ---- RQ1 coverage at eps=0.05 ----
cov = {}
for c in ("amazon", "stackexchange", "imdb"):
    r = load(f"rq1_coverage/results_{c}.json")
    cov[c] = {x["k"]: x["coverage_mean"] for x in r["results"] if x["eps"] == 0.05}
check("amazon coverage k=100", 0.30, cov["amazon"][100], 0.08)
check("imdb coverage k=100", 0.49, cov["imdb"][100], 0.08)
check("stackexchange coverage k=100", 0.08, cov["stackexchange"][100], 0.10)
check("amazon coverage k=10000", 0.945, cov["amazon"][10000], 0.03)
# the abstract's headline: how many of Amazon's categories a top-100
# retrieval answers, stated as a count rather than a percentage
n_fac_am = load("rq1_coverage/results_amazon.json")["n_facets_admitted"]
check("amazon facets answered at k=100", 10,
      cov["amazon"][100] * n_fac_am, 0.06)

# ---- top-facet agreement, and the separable subset it is over ----
# A sentence here once mixed two result files with different query sets
# and temperature grids, and quoted a range that matched neither. Both
# the range and the denominators are pinned to one source now.
agree, nsep = [], {}
for c in ("amazon", "stackexchange", "imdb"):
    r = load(f"rq1_coverage/results_{c}.json")
    v = [x for x in r["results"] if x["eps"] == 0.05]
    agree += [x["top1_agree_separable"] for x in v]
    nsep[c] = {x["n_separable"] for x in v}
    checks += 1
    if len(nsep[c]) != 1:
        fails.append(f"{c}: n_separable varies within one temperature {nsep[c]}")
check("lowest top-facet agreement (Amazon)", 0.058,
      min(x["top1_agree_separable"] for x in
          load("rq1_coverage/results_amazon.json")["results"] if x["eps"] == 0.05), 0.05)
check("lowest top-facet agreement (StackExchange)", 0.10,
      min(x["top1_agree_separable"] for x in
          load("rq1_coverage/results_stackexchange.json")["results"] if x["eps"] == 0.05), 0.05)
for c, want in (("amazon", 121), ("stackexchange", 200), ("imdb", 182)):
    check(f"{c} separable queries", want, nsep[c].pop(), 0.001)

# ---- concentration ----
sc = load("rq1_coverage/scaling.json")
am = {p["k"]: p for p in sc["corpora"]["amazon"]["points"]}
check("amazon uniform ref k=100", 0.91, am[100]["uniform_reference"], 0.03)
check("amazon concentration gap k=100 (pp)", 62, am[100]["concentration_gap_pp"], 0.05)

# ---- RQ2 cost (repeated, trustworthy) ----
rep = load("rq2_cost/results_repeated.json")["results"]
check("amazon exact f32 ms", 40.8, rep["amazon"]["exact_f32_ms"]["median_of_runs"], 0.12)
check("stackexchange exact f32 ms", 20.5, rep["stackexchange"]["exact_f32_ms"]["median_of_runs"], 0.12)
check("imdb exact f32 ms", 9.6, rep["imdb"]["exact_f32_ms"]["median_of_runs"], 0.12)

idx = load("rq2_cost/results_indexed_imdb.json")
itr = load("rq2_cost/results_indexed_imdb_iter.json")
check("ANN as-shipped coverage", 0.475, idx["coverage"]["pg_hnsw_topk10000_coverage"], 0.05)
check("ANN iterative k=10000 ms", 29.4, itr["latency_ms"]["pg_hnsw_topk10000_ms"]["median"], 0.35)
check("ANN iterative k=10000 coverage", 0.99, itr["coverage"]["pg_hnsw_topk10000_coverage"], 0.02)

# ---- RQ3 maintenance ----
m3 = {c: load(f"rq3_maintenance/results_{c}.json") for c in ("amazon", "stackexchange", "imdb")}
check("amazon rescan ms", 99.4, m3["amazon"]["rescan_ms"], 0.35)
check("amazon maintained read us", 6.1, m3["amazon"]["maintained_read_ms"] * 1e3, 0.50)
check("amazon state bytes", 792, m3["amazon"]["state_bytes_total"], 0.001)
check("insert us", 0.7, m3["amazon"]["insert_us"], 0.50)
lo = min(m3[c]["delete_anchor_us"] for c in m3)
hi = max(m3[c]["delete_anchor_us"] for c in m3)
check("anchor delete low us", 282, lo, 0.40)
check("anchor delete high us", 654, hi, 0.40)

# ---- RQ4 contract (v2: propose / verify / repair) ----
declined, repaired, slack, cand, kr = [], [], [], [], []
for c in ("amazon", "stackexchange", "imdb"):
    r4 = load(f"rq4_contract/results2_{c}.json")
    for row in r4["main"]:
        checks += 1
        if row["promise_held"] != 1.0:
            fails.append(f"{c} budget {row['budget']:.3g}: promise held "
                         f"{row['promise_held']:.2f}, paper claims always")
        declined.append(row["declined_rate"])
        repaired.append(row["frac_facets_repaired"])
        slack.append(row["bound_slack_median"])
        cand.append(row["candidates_frac_median"])
    # planning must stay far below the pass it plans for
    checks += 1
    if max(r["plan_ms_median"] for r in r4["main"]) > 0.07:
        fails.append(f"{c}: planning exceeded the 0.07 ms the paper claims")
    # the sample ablation the paper reports and Figure 4(b) draws.
    # Read from the per-query frame with the SAME aggregation the
    # figure uses, so the text and the plot cannot disagree.
    d = pd.read_parquet(E / "rq4_contract" / f"ablation_{c}.parquet")
    d = d[~d.declined]
    kr.append((c, d.groupby("sample").k_ratio.median().to_dict()))

npairs = sum(load(f"rq4_contract/results2_{c}.json")["n_queries"]
             * len(load(f"rq4_contract/results2_{c}.json")["main"])
             for c in ("imdb", "amazon", "stackexchange"))
check("(query, budget) pairs the contract held on", 2400, npairs, 0.001)
check("declined rate low", 0.26, min(declined), 0.05)
check("declined rate high", 0.84, max(declined), 0.05)
check("facets repaired low", 0.001, min(repaired), 0.60)
check("facets repaired high", 0.36, max(repaired), 0.08)
check("candidate fraction low (%)", 1.0, min(cand) * 100, 0.10)
check("candidate fraction high (%)", 3.8, max(cand) * 100, 0.10)
check("bound slack low", 11, min(slack), 0.15)
check("bound slack high", 47, max(slack), 0.15)

# the k_plan/k_oracle trend quoted in RQ4 and drawn in Figure 4
KR = {c: m for c, m in kr}
for c, first, last in (("imdb", 35.9, 1.81), ("stackexchange", 27.2, 1.33),
                       ("amazon", 12.6, 0.74)):
    ks = sorted(KR[c])
    check(f"{c} k_plan/k_oracle at {ks[0]} samples", first, KR[c][ks[0]], 0.10)
    check(f"{c} k_plan/k_oracle at {ks[-1]} samples", last, KR[c][ks[-1]], 0.10)

# ---- per-facet top-b, the counter-design ----
pf = load("rq1_coverage/perfacet_amazon.json")
P = {(r["eps"], r["k"], r["method"]): r for r in pf["results"]}
check("perfacet amazon cov k=1000", 1.0, P[(0.02, 1000, "perfacet_equal")]["coverage"], 0.001)
check("perfacet amazon mae k=1000", 0.030, P[(0.02, 1000, "perfacet_equal")]["mae"], 0.10)
check("global topk amazon mae k=1000", 0.092, P[(0.02, 1000, "global_topk")]["mae"], 0.10)
check("perfacet amazon top1 k=1000", 0.73, P[(0.02, 1000, "perfacet_equal")]["top1_agree"], 0.06)
check("global topk amazon top1 k=1000", 0.085, P[(0.02, 1000, "global_topk")]["top1_agree"], 0.10)
check("perfacet amazon worst facet mean", 0.17,
      P[(0.02, 1000, "perfacet_equal")]["max_abs_err_mean"], 0.10)
check("perfacet amazon worst facet max", 0.62,
      P[(0.02, 1000, "perfacet_equal")]["max_abs_err_p100"], 0.10)
check("perfacet amazon mae k=100", 0.154, P[(0.02, 100, "perfacet_equal")]["mae"], 0.08)
check("global topk amazon mae k=100", 0.101, P[(0.02, 100, "global_topk")]["mae"], 0.08)

bl = load("rq1_coverage/baselines_stackexchange.json")
B = {(r["eps"], r["method"]): r for r in bl["results"]}
check("stackexchange perfacet mae", 1.02, B[(0.02, "perfacet")]["mae"], 0.08)
check("stackexchange topk mae", 1.24, B[(0.02, "topk")]["mae"], 0.08)
check("stackexchange perfacet worst", 20.1, B[(0.02, "perfacet")]["max_abs"], 0.08)
check("stackexchange topk worst", 16.8, B[(0.02, "topk")]["max_abs"], 0.08)
for c in ("amazon", "stackexchange", "imdb"):
    b = load(f"rq1_coverage/baselines_{c}.json")
    check(f"{c} baselines n_queries", 200, b["n_queries"], 0.001)
    # Who is most accurate at each temperature. The paper and the table
    # caption both used to assert that per-facet top-b wins everywhere;
    # it wins at the sharp setting and LOSES to stratified sampling at
    # the diffuse one on every corpus, which is the interesting half.
    t = {(x["eps"], x["method"]): x["mae"] for x in b["results"]}
    for eps, want in ((0.02, "perfacet"), (0.5, "stratified")):
        cand = [(v, m) for (e, m), v in t.items() if e == eps]
        checks += 1
        got = min(cand)[1]
        if got != want:
            fails.append(f"{c} at eps={eps}: lowest MAE is {got!r}, "
                         f"paper says {want!r}")

# ---- SQL forms ----
sq = load("rq2_cost/sql_baselines_imdb.json")
S = {(r["engine"], r["form"], r["eps"]): r for r in sq["results"]}
checks += 1
if "overflow" not in sq["errors"].get("postgres/naive", ""):
    fails.append("postgres naive did not fail with an overflow")
checks += 1
if "underflow" not in sq["errors"].get("postgres/stable", ""):
    fails.append("postgres max-shifted form did not fail with an underflow")
checks += 1
if S[("postgres", "guarded", 0.001)]["ok_rate"] != 1.0:
    fails.append("the guarded PostgreSQL form did not complete at every temperature")
ratio = (S[("duckdb", "stable", 0.02)]["ms_median"]
         / S[("numpy", "reference", 0.02)]["ms_median"])
check("fused pass vs stable DuckDB", 10, ratio, 0.30)
pg_naive = S[("postgres", "naive", 0.5)]["ms_median"]
pg_guard = S[("postgres", "guarded", 0.5)]["ms_median"]
check("guarded PG vs naive PG", 10, pg_guard / pg_naive, 0.20)

# ---- selectivity sweep ----
sv = load("rq2_cost/selectivity_imdb.json")
V = {r["target_selectivity"]: r for r in sv["results"]}
check("index fill at 5% selectivity (%)", 1.0, V[0.05]["ann_fill"] * 100, 0.15)
check("coverage as shipped at 5% (%)", 18, V[0.05]["ann_coverage"] * 100, 0.10)
check("iterative/exact at 5%", 2.6, V[0.05]["iter_ms"] / V[0.05]["exact_ms"], 0.15)
check("iterative/exact unfiltered", 0.34, V[1.0]["iter_ms"] / V[1.0]["exact_ms"], 0.20)
lo_cov = min(r["iter_coverage"] for r in sv["results"])
hi_cov = max(r["iter_coverage"] for r in sv["results"])
check("iterative coverage low (%)", 81, lo_cov * 100, 0.03)
check("iterative coverage high (%)", 89, hi_cov * 100, 0.03)

# ---- Q standing queries ----
qs = load("rq3_maintenance/qscaling_amazon.json")
Q = {r["Q"]: r for r in qs["results"]}
QCLAIMS = {1: 0.22, 10: 3.2, 100: 15.7, 1000: 167.0}
for q, us in QCLAIMS.items():
    checks += 1
    if q not in Q:
        fails.append(f"Q-scaling has no measurement at Q={q}")
    else:
        check(f"insert us at Q={q}", us, Q[q]["insert_us"], 0.40)
check("payload MiB at Q=1000", 1.3, Q[1000]["payload_bytes"] / 1024**2, 0.05)
# the state size the paper reports must be the engine's GroupState:
# three f64 accumulators plus the live-row and multiplicity counters
check("bytes per (query, facet)", 40,
      Q[1000]["payload_bytes"] / (Q[1000]["n_facets"] * 1000), 0.001)
check("Amazon maintained state, bytes", 1320,
      Q[1][ "payload_bytes"], 0.001)
check("rebuild ms at Q=1", 125, Q[1]["rebuild_all_ms"], 0.30)
check("rebuild ms at Q=10", 635, Q[10]["rebuild_all_ms"], 0.30)
check("rebuild ms at Q=100", 3373, Q[100]["rebuild_all_ms"], 0.30)
checks += 1
if Q[1000]["rebuild_fits_in_memory"]:
    fails.append("the paper says the batched rebuild stops fitting at Q=1000")

# ---- source-item exclusion control ----
worst = 0.0
for c in ("amazon", "stackexchange", "imdb"):
    se = load(f"rq1_coverage/selfexclude_{c}.json")
    worst = max(worst, se["max_abs_delta_pp"])
    checks += 1
    if se["self_similarity_median"] < 0.99:
        fails.append(f"{c}: query-to-source similarity {se['self_similarity_median']:.3f}, "
                     "the paper says 1.000")
check("largest coverage change when the source item is removed (pp)",
      0.23, worst, 0.15)

# ---- error bars that cannot be believed (Section 7.8) ----
# Every number the new subsection quotes, pinned to its result file.
bt, ol, cert = {}, {}, []
for c in ("imdb", "amazon", "stackexchange"):
    b = load(f"rq4_contract/bootstrap_{c}.json")
    for r in b["results"]:
        bt[(c, r["method"], r["eps"], r["k"])] = r
    o = load(f"rq1_coverage/olla_{c}.json")
    for r in o["results"]:
        ol[(c, r["eps"], r["budget"])] = r
    # the contract, evaluated on the same pairs the bootstrap used
    d = pd.read_parquet(E / "rq4_contract" / f"bootstrap_per_facet_{c}.parquet")
    d = d[(d.method == "perfacet_topk") & (d.k == 1000)]
    for eps in (0.02, 0.5):
        sub = d[d.eps == eps]
        cert.append(float((sub.abs_err <= sub.certified_bound * (1 + 1e-9)).mean()))

CORPORA = ("imdb", "amazon", "stackexchange")
boot_sharp = [bt[(c, "perfacet_topk", 0.02, 1000)]["ci_coverage"] for c in CORPORA]
boot_diff = [bt[(c, "perfacet_topk", 0.5, 1000)]["ci_coverage"] for c in CORPORA]
check("bootstrap coverage, sharp, low", 0.96, min(boot_sharp), 0.02)
check("bootstrap coverage, sharp, high", 1.00, max(boot_sharp), 0.02)
check("bootstrap coverage, diffuse, low", 0.47, min(boot_diff), 0.05)
check("bootstrap coverage, diffuse, high", 0.86, max(boot_diff), 0.03)

ess = [bt[(c, "perfacet_topk", 0.02, k)]["ess_median"]
       for c in CORPORA for k in (100, 1000)]
check("effective sample size, low", 1.0, min(ess), 0.02)
check("effective sample size, high", 3.6, max(ess), 0.05)

# the Bayesian bootstrap must not rescue the coverage
for c in CORPORA:
    for eps in (0.02, 0.5):
        a_, b_ = (bt[(c, "perfacet_topk", eps, 1000)]["ci_coverage"],
                  bt[(c, "perfacet_bayes", eps, 1000)]["ci_coverage"])
        checks += 1
        if abs(a_ - b_) > 0.05:
            fails.append(f"{c} eps={eps}: Bayesian bootstrap coverage {b_:.2f} "
                         f"differs from the percentile bootstrap {a_:.2f}; "
                         "the paper says the resampler does not matter")

olla_sharp = [ol[(c, 0.02, 1000)]["ci_coverage"] for c in CORPORA]
olla_diff = [ol[(c, 0.5, 1000)]["ci_coverage"] for c in CORPORA]
check("stratified sampling coverage, sharp, low", 0.29, min(olla_sharp), 0.05)
check("stratified sampling coverage, sharp, high", 0.39, max(olla_sharp), 0.05)
check("stratified sampling coverage, diffuse, low", 0.68, min(olla_diff), 0.05)
check("stratified sampling coverage, diffuse, high", 0.83, max(olla_diff), 0.03)
check("stratified sampling, Amazon at 10x the budget", 0.94,
      ol[("amazon", 0.5, 10000)]["ci_coverage"], 0.03)
for c in CORPORA:
    checks += 1
    if ol[(c, 0.02, 1000)]["facet_coverage"] != 1.0:
        fails.append(f"{c}: stratified sampling lost a facet; the paper "
                     "says it never does")

checks += 1
if any(x != 1.0 for x in cert):
    fails.append(f"the deterministic bound did not hold on every pair: {cert}")
# and it is the narrower of the two on exactly two corpora at the sharp setting
narrower = sum(1 for c in CORPORA
               if bt[(c, "perfacet_topk", 0.02, 1000)]["certified_bound_median"]
               < bt[(c, "perfacet_topk", 0.02, 1000)]["half_width_median"])
check("corpora where the bound is narrower at the sharp setting", 2, narrower, 0.01)
ratio = max(bt[(c, "perfacet_topk", 0.02, 1000)]["half_width_median"]
            / bt[(c, "perfacet_topk", 0.02, 1000)]["certified_bound_median"]
            for c in CORPORA
            if bt[(c, "perfacet_topk", 0.02, 1000)]["certified_bound_median"]
            < bt[(c, "perfacet_topk", 0.02, 1000)]["half_width_median"])
check("largest width advantage of the bound", 33, ratio, 0.05)

# ---- cost decomposition and the two P2 experiments ----
sh = {}
for c in ("imdb", "stackexchange", "amazon"):
    d = load(f"rq4_contract/cost_breakdown_{c}.json")
    sh[c] = d["shares"]
check("scoring share, low", 0.797, min(x["scoring"] for x in sh.values()), 0.03)
check("scoring share, high", 0.817, max(x["scoring"] for x in sh.values()), 0.03)
check("accumulation share, low", 0.155, min(x["accumulation"] for x in sh.values()), 0.06)
check("accumulation share, high", 0.181, max(x["accumulation"] for x in sh.values()), 0.06)
checks += 1
if max(x["planning"] for x in sh.values()) >= 0.02:
    fails.append("planning exceeded the 2% share the paper claims")
checks += 1
if max(x["repair"] for x in sh.values()) >= 0.03:
    fails.append("repair exceeded the 3% share the paper claims")

bt = load("rq4_contract/bound_tightness.json")
check("bound attained under the adversarial instance", 1.0,
      bt["adversarial_max_ratio"], 0.001)
checks += 1
if bt["natural_min_ratio"] < 10:
    fails.append("the natural regime was not far from the bound after all")

lx = load("rq1_coverage/lexical_amazon.json")
L = {(r["k"], r["retriever"]): r for r in lx["results"]}
check("dense concentration gap at k=100 (pp)", 58, L[(100, "dense")]["gap_pp"], 0.05)
check("hybrid concentration gap at k=100 (pp)", 45, L[(100, "hybrid")]["gap_pp"], 0.05)
check("bm25 concentration gap at k=100 (pp)", 33, L[(100, "bm25")]["gap_pp"], 0.06)
checks += 1
if not (L[(100, "bm25")]["gap_pp"] < L[(100, "hybrid")]["gap_pp"]
        < L[(100, "dense")]["gap_pp"]):
    fails.append("the paper's ordering bm25 < hybrid < dense does not hold")

# ---- what coverage costs to buy with retrieval (P2.14) ----
gl = {}
for c in ("imdb", "stackexchange", "amazon"):
    d = load(f"rq2_cost/grouped_latency_{c}.json")
    gl[c] = {r["method"]: r for r in d["results"]}
    checks += 1
    if gl[c]["per_facet_index"]["coverage"] != 1.0:
        fails.append(f"{c}: per-facet index did not reach full coverage")
    checks += 1
    if gl[c]["scan_partition"]["coverage"] != 1.0:
        fails.append(f"{c}: scan+partition did not reach full coverage")
pf = [gl[c]["per_facet_index"]["ratio_to_exact"] for c in gl]
sp = [gl[c]["scan_partition"]["ratio_to_exact"] for c in gl]
check("per-facet index, cheapest", 1.07, min(pf), 0.05)
check("per-facet index, dearest", 1.39, max(pf), 0.05)
check("scan+partition, cheapest", 1.00, min(sp), 0.05)
check("scan+partition, dearest", 1.12, max(sp), 0.05)

# ---- the pre-execution bound's headroom, now claimed in Sections 5 and 7 ----
cp_ = {}
for c in ("imdb", "stackexchange", "amazon"):
    d = load(f"rq4_contract/certified_perfacet_{c}.json")
    sharp = [r for r in d["results"] if r["eps"] == 0.02]
    cp_[c] = sharp
    checks += 1
    if any(r["declined_pre"] != 0.0 for r in sharp):
        fails.append(f"{c}: the pre-execution plan declined a facet; "
                     "the paper says it declines none")
fr = [r["frac_pre_median"] for v in cp_.values() for r in v]
check("pre-execution plan, rows read, low (%)", 0.6, min(fr) * 100, 0.15)
check("pre-execution plan, rows read, high (%)", 9.3, max(fr) * 100, 0.10)
sp = [r["speedup_if_indexed"] for v in cp_.values() for r in v]
check("scan reduction, low", 11, min(sp), 0.10)
check("scan reduction, high", 170, max(sp), 0.10)

# ---- a second engine and index family (W2) ----
fs = load("rq2_cost/faiss_selectivity_imdb.json")
comp = fs["completeness"]
checks += 1
if any(c["setting"] is not None for c in comp):
    fails.append("some FAISS configuration did reach 99% coverage; the "
                 "paper says none did")
best = [c["best_coverage"] for c in comp]
ceil = [c["exact_topk_coverage"] for c in comp]
check("FAISS plateau, low", 0.862, min(best), 0.02)
check("FAISS plateau, high", 0.891, max(best), 0.02)
check("exact top-k ceiling at 5%", 0.884,
      [c["exact_topk_coverage"] for c in comp
       if c["target_selectivity"] == 0.05][0], 0.02)
check("exact top-k ceiling at 25%", 0.861,
      [c["exact_topk_coverage"] for c in comp
       if c["target_selectivity"] == 0.25][0], 0.02)
# the plateau must actually BE the ceiling, which is the paper's point
checks += 1
if max(b - c for b, c in zip(best, ceil)) > 0.02:
    fails.append("the FAISS plateau is not the exact top-k ceiling after all")
T = fs["tuning"]
w = max((r for r in T if r["index"] == "hnsw" and r["target_selectivity"] == 0.05),
        key=lambda r: r["setting"])
check("HNSW at the widest setting, x exact", 631, w["ratio_to_exact"], 0.10)
check("HNSW gain over the ceiling (pp)", 0.36,
      (w["coverage"] - w["exact_topk_coverage"]) * 100, 0.40)

# ---- claims that must be present verbatim ----
in_text("nDCG@10 of 0.739")
in_text("40 bytes per facet")
in_text("1320 bytes")
in_text("80--82\\%")
in_text("Generative-AI tools were used for")
in_text("0.74\\% of Amazon's 1.35M admitted rows")
in_text("709.78")
in_text("24 cores")
# LaTeX line breaks split phrases, so this one is normalised first
checks += 1
if HAVE_TEXT and "rather than running its system" not in re.sub(r"\s+", " ", TXT):
    fails.append("the paper no longer says it did not run the rival system")
in_text("$1/\\sum_i p_i^2$")
in_text("121, 200 and 182 of 200")
for stale in ("8--24", "121 to\n197"):
    checks += 1
    if HAVE_TEXT and stale in TXT:
        fails.append(f"stale top-facet claim still present: {stale!r}")
# the stale figures this script was extended to catch
for stale in ("7 of 28", "1.2M-review", "17--26", "32-core",
              "exact at every temperature", "beyond $10^{300}$",
              "as few as eleven"):
    if not HAVE_TEXT:
        break
    checks += 1
    if stale in TXT:
        fails.append(f"stale claim still in the paper: {stale!r}")

if not HAVE_TEXT:
    print("note: the paper source is not in this artifact, so the "
          "text-presence checks are skipped; every numeric check ran.")
print(f"checked {checks} claims")
if fails:
    print(f"\n{len(fails)} MISMATCH(ES):")
    for f in fails:
        print("  !", f)
    sys.exit(1)
print("all claims match the result files")
