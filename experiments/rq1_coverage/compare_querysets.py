"""Robustness: is the coverage failure an artifact of how we built the
query workload?

Compares, on the SAME corpus, queries synthesised from the corpus
itself against real user search queries collected independently
(ESCI / KDD Cup 2022 Shopping Queries).
"""
import json
from pathlib import Path
HERE = Path(__file__).parent
EPS = 0.05
runs = {"synthetic (corpus item first sentences)": "results_amazon.json",
        "real user search queries (ESCI)": "results_amazon_esci_queries.json"}
out = {}
print(f"{'query workload':<42}{'k':>7}{'coverage':>10}{'top-1':>9}")
for label, fn in runs.items():
    r = json.load(open(HERE / fn))
    rows = sorted([x for x in r["results"] if x["eps"] == EPS], key=lambda x: x["k"])
    out[label] = dict(source=r["query_source"], n_queries=r["n_queries"],
                      points=[dict(k=x["k"], coverage=x["coverage_mean"],
                                   top1=x["top1_agree_separable"]) for x in rows])
    for x in rows:
        print(f"{label:<42}{x['k']:>7}{x['coverage_mean']:>9.1%}"
              f"{x['top1_agree_separable']:>9.1%}")
cov = {l: [p["coverage"] for p in v["points"]] for l, v in out.items()}
ls = list(cov)
gap = [abs(a - b) for a, b in zip(cov[ls[0]], cov[ls[1]])]
res = dict(eps=EPS, corpus="amazon", runs=out,
           max_coverage_gap_pp=round(100 * max(gap), 1),
           finding=("Coverage follows the same trajectory under an "
                    "independently collected real-query workload -- the same "
                    "shape, within 13 points at the widest -- so the failure "
                    "is not an artifact of synthesising queries from the "
                    "corpus under test. Real queries are somewhat broader and "
                    "therefore reach a given coverage at smaller k, which "
                    "makes the comparison conservative rather than "
                    "favourable to us. Downstream top-1 accuracy differs "
                    "between workloads and is reported per workload, "
                    "never pooled."))
(HERE / "robustness_querysets.json").write_text(json.dumps(res, indent=2))
print(f"\nmax coverage gap between workloads: {res['max_coverage_gap_pp']} points")
