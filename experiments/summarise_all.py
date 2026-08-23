"""One table of every number the paper claims, read from the result
files rather than retyped.

Two jobs. It gives the writing a single place to quote from, and it
cross-checks measurements that were taken independently: the same
quantity measured by two harnesses must agree, and a disagreement is a
bug in one of them rather than a detail to smooth over. Run it before
quoting anything in the text.
"""
from pathlib import Path as _P
_ROOT = _P(__file__).resolve().parent.parent
import glob, json
from pathlib import Path

E = Path(str(_ROOT / "experiments"))
out = {}

def load(pat):
    return {Path(f).stem: json.load(open(f)) for f in sorted(glob.glob(str(pat)))}

# ---- corpora ----
out["corpora"] = {}
for m in sorted(glob.glob(str(E / "corpora/*/meta.json"))):
    d = json.load(open(m)); name = Path(m).parent.name
    out["corpora"][name] = dict(
        rows=d.get("n_rows"), facets=d.get("n_facets"),
        value=d.get("value_col"), source=d.get("source"),
        encoders=[d.get("embedding", {}).get("model")] +
                 [v["model"] for v in d.get("alt_embeddings", {}).values()])

# ---- RQ0 retriever control ----
p = E / "rq0_retriever/results.json"
if p.exists():
    r = json.load(open(p))
    out["rq0_retriever"] = {k.split("/")[-1]: v for k, v in r["results"].items()}

# ---- RQ1 coverage ----
out["rq1_coverage"] = {}
for k, r in load(E / "rq1_coverage/results_*.json").items():
    tag = k.replace("results_", "")
    rows = [x for x in r["results"] if x["eps"] == 0.05]
    out["rq1_coverage"][tag] = dict(
        corpus=r.get("corpus_dir", tag), encoder=r.get("encoder", "emb.npy"),
        queries=r["n_queries"], admitted=r["n_admitted"], facets=r["n_facets_admitted"],
        coverage={x["k"]: round(x["coverage_mean"], 4) for x in rows},
        top1_separable={x["k"]: round(x["top1_agree_separable"], 4) for x in rows})

sp = E / "rq1_coverage/scaling.json"
if sp.exists():
    s = json.load(open(sp))
    out["rq1_concentration"] = {
        c: {p["k"]: dict(measured=round(p["coverage"], 4),
                         uniform_ref=round(p["uniform_reference"], 4),
                         gap_pp=round(p["concentration_gap_pp"], 1))
            for p in v["points"]}
        for c, v in s["corpora"].items()}
    out["rq1_retracted"] = s.get("retracted_claim")

out["rq1_baselines"] = {}
for k, r in load(E / "rq1_coverage/baselines_*.json").items():
    if "per_query" in k:
        continue
    out["rq1_baselines"][k.replace("baselines_", "")] = dict(
        encoder=r.get("encoder"), k=r["retrieval_budget_k"],
        table={f"eps{x['eps']}_{x['method']}":
               dict(cov=round(x["coverage"], 4), mae=round(x["mae"], 4))
               for x in r["results"]},
        all_significant=all(t.get("significant_05", False)
                            for t in r["significance"]["tests"]))

# ---- RQ2 cost ----
out["rq2_cost"] = {}
for k, r in load(E / "rq2_cost/results_*.json").items():
    if "indexed" in k:
        out.setdefault("rq2_indexed", {})[k.replace("results_indexed_", "")] = dict(
            iterative=r.get("iterative_scan"),
            latency={c: round(v["median"], 2) for c, v in r["latency_ms"].items()},
            coverage={c: round(v, 4) for c, v in r["coverage"].items()})
    else:
        out["rq2_cost"][k.replace("results_", "")] = dict(
            admitted=r["n_admitted"], facets=r["n_facets"],
            latency={c: round(v["median"], 2) for c, v in r["latency_ms"].items()},
            max_rel_err={c: v["max_rel_err"] for c, v in r["correctness"].items()})

# ---- RQ3 maintenance ----
out["rq3_maintenance"] = {}
for k, r in load(E / "rq3_maintenance/results_*.json").items():
    out["rq3_maintenance"][k.replace("results_", "")] = dict(
        rescan_ms=round(r["rescan_ms"], 2),
        read_ms=round(r["maintained_read_ms"], 4),
        insert_us=round(r["insert_us"], 2),
        delete_plain_us=round(r["delete_plain_us"], 2),
        delete_anchor_us=round(r["delete_anchor_us"], 1),
        state_bytes=r["state_bytes_total"],
        churn_max_err=r["churn"]["max_rel_err"],
        speedup=round(r["rescan_ms"] / r["maintained_read_ms"]))

# ---- RQ4 contract ----
out["rq4_contract"] = {}
for k, r in load(E / "rq4_contract/results_*.json").items():
    out["rq4_contract"][k.replace("results_", "")] = dict(
        queries=r["n_queries"],
        rows=[dict(budget=x["budget"], kstar=x["kstar_median"],
                   promise=x["promise_held"], coverage=x["coverage"],
                   topk_promise=x["topk_promise_held"],
                   topk_coverage=round(x["topk_coverage"], 4))
              for x in r["results"]])

# ---- cross-checks between independent harnesses ----
checks = []
for c in ("imdb", "stackexchange", "amazon"):
    a = out["rq2_cost"].get(c, {}).get("latency", {}).get("exact_f64_ms")
    b = out["rq3_maintenance"].get(c, {}).get("rescan_ms")
    if a and b:
        checks.append(dict(
            quantity=f"{c}: full exact pass",
            rq2_exact_f64_ms=a, rq3_rescan_ms=b,
            ratio=round(max(a, b) / min(a, b), 2),
            note="two independent implementations of the same computation; "
                 "a ratio far from 1 would mean one of them is wrong"))
out["cross_checks"] = checks

(E / "SUMMARY.json").write_text(json.dumps(out, indent=2))
print(json.dumps({k: (list(v)[:4] if isinstance(v, dict) else v)
                  for k, v in out.items()}, indent=1)[:900])
print("\ncross-checks:")
for c in checks:
    print(f"  {c['quantity']:<28} RQ2 {c['rq2_exact_f64_ms']:6.1f} ms  "
          f"RQ3 {c['rq3_rescan_ms']:6.1f} ms   ratio {c['ratio']}")
print(f"\nwrote {E/'SUMMARY.json'}")
