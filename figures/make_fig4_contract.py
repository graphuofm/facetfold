"""Figure 4: what the certified truncation costs.

The earlier version of this figure asked whether the promise was kept.
That is close to a tautology: execution measures the true omitted mass
and re-folds any group whose bound misses, so the delivered answer must
satisfy the budget. The informative questions are how much slack the
bound carries and how good the proposal is, so this figure reports
those instead.

(a) declared budget against the realised worst-facet error, per query.
    Everything sits below the diagonal, which is the guarantee; the
    vertical distance from it is slack the contract does not spend.
(b) the sketch's sample size against k_planner / k_oracle. The proposal
    approaches the oracle as the sample grows and crosses BELOW one on
    two corpora, which is why the guarantee cannot rest on it and the
    verification pass is not optional.

Numbers read from experiments/rq4_contract/results2_*.json and the
per-query ablation frames beside them.
"""
from pathlib import Path as _P
_ROOT = _P(__file__).resolve().parent.parent
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EXP = Path(str(_ROOT / "experiments") + "/rq4_contract")
OUT = Path(__file__).resolve().parent / "fig4_contract.pdf"
ORDER = ["imdb", "stackexchange", "amazon"]
NICE = {"imdb": "IMDb", "stackexchange": "StackExchange", "amazon": "Amazon"}
COL = {"imdb": "#E69F00", "stackexchange": "#CC79A7", "amazon": "#0072B2"}
MK = {"imdb": "s", "stackexchange": "^", "amazon": "o"}
SKETCH = 1024

R, A = {}, {}
for c in ORDER:
    f = EXP / f"results2_{c}.json"
    if f.exists():
        R[c] = json.load(open(f))
        A[c] = pd.read_parquet(EXP / f"ablation_{c}.parquet")
cs = [c for c in ORDER if c in R]
assert cs, "no RQ4 results found"

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.4, 2.0))

# ---- (a) budget against realised worst-facet error -------------------
lo, hi = 1e-4, 2e3
ax1.plot([lo, hi], [lo, hi], ls="--", lw=1.2, color="0.45", zorder=1)
violations = 0
for c in cs:
    d = A[c]
    d = d[(d["sample"] == SKETCH) & (~d.declined)]
    violations += int((d.max_abs_err > d.budget * (1 + 1e-9)).sum())
    ax1.scatter(d.budget, np.maximum(d.max_abs_err, lo), s=5, alpha=0.35,
                marker=MK[c], color=COL[c], linewidths=0, label=NICE[c])
assert violations == 0, f"{violations} deliveries outside the declared budget"
ax1.set_xscale("log"); ax1.set_yscale("log")
ax1.set_xlim(lo, hi); ax1.set_ylim(lo, hi)
ax1.set_xlabel("declared budget $\\beta$ (value units)", fontsize=8.5)
ax1.set_ylabel("realised worst-facet error", fontsize=8.5)
ax1.set_title("(a) the guarantee, and its slack", fontsize=8.5, pad=6)
ax1.tick_params(labelsize=7.5); ax1.grid(alpha=0.25, lw=0.5)
ax1.legend(fontsize=6, loc="upper left", framealpha=0.93, markerscale=2)
ax1.text(0.97, 0.06, "below the diagonal\n= within budget", fontsize=6,
         ha="right", va="bottom", transform=ax1.transAxes, color="0.35")

# ---- (b) sample size against k_planner / k_oracle --------------------
for c in cs:
    d = A[c]
    d = d[~d.declined]
    g = d.groupby("sample").k_ratio.median()
    ax2.plot(g.index, g.values, MK[c] + "-", ms=4.5, lw=1.5,
             color=COL[c], label=NICE[c])
ax2.axhline(1.0, ls="--", lw=1.2, color="0.45")
ax2.set_xscale("log", base=2); ax2.set_yscale("log")
ax2.set_xlabel("planning sample size (rows)", fontsize=8.5)
ax2.set_ylabel("$k_\\mathrm{plan}\\,/\\,k_\\mathrm{oracle}$", fontsize=8.5)
ax2.set_title("(b) how good the proposal is", fontsize=8.5, pad=6)
ax2.tick_params(labelsize=7.5); ax2.grid(alpha=0.25, lw=0.5, which="both")
ax2.set_xticks(sorted(A[cs[0]]["sample"].unique()))
ax2.set_xticklabels([str(x) for x in sorted(A[cs[0]]["sample"].unique())],
                    fontsize=7.5)
ax2.legend(fontsize=6, loc="upper right", framealpha=0.93)
ax2.text(0.03, 0.06, "below 1 = the proposal\nunder-reads; repair\ncarries the guarantee",
         fontsize=6, ha="left", va="bottom", transform=ax2.transAxes,
         color="0.35")

fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight")
print(f"wrote {OUT}")
print(f"  deliveries outside the declared budget: {violations}")
for c in cs:
    d = A[c][~A[c].declined]
    g = d.groupby("sample").k_ratio.median()
    print(f"  {NICE[c]}: k_plan/k_oracle {g.iloc[0]:.1f} -> {g.iloc[-1]:.2f}")
