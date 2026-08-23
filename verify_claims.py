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
QCLAIMS = {1: 0.22, 10: 3.1, 100: 15.4, 1000: 170.0}
for q, us in QCLAIMS.items():
    checks += 1
    if q not in Q:
        fails.append(f"Q-scaling has no measurement at Q={q}")
    else:
        check(f"insert us at Q={q}", us, Q[q]["insert_us"], 0.40)
check("payload MiB at Q=1000", 1.0, Q[1000]["payload_bytes"] / 1024**2, 0.02)
check("rebuild ms at Q=1", 118, Q[1]["rebuild_all_ms"], 0.30)
check("rebuild ms at Q=10", 572, Q[10]["rebuild_all_ms"], 0.30)
check("rebuild ms at Q=100", 3217, Q[100]["rebuild_all_ms"], 0.30)
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

# ---- claims that must be present verbatim ----
in_text("nDCG@10 of 0.739")
in_text("792 bytes")
in_text("0.74\\% of Amazon's 1.35M admitted rows")
in_text("709.78")
in_text("24 cores")
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
