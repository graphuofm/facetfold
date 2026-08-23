"""Figure 3: keeping the answer fresh.

Left: the cost of one refresh. Recomputing from scratch is what a
system with no maintained state must do; reading a maintained state is
what the operator's mergeable form allows. Update costs are shown
separately because they genuinely differ -- removing the item that
holds a facet's anchor forces one bounded pass over that facet, and
reporting a single "constant time" number would hide it.

Right: the state stays exact under a long randomised update stream, so
the cheap path is not buying speed with drift.

Numbers read from experiments/rq3_maintenance/results_*.json.
"""
from pathlib import Path as _P
_ROOT = _P(__file__).resolve().parent.parent
import glob, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
# ACM submissions must not carry Type 3 fonts, which is what matplotlib
# emits by default; 42 selects TrueType so the glyphs embed properly.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import numpy as np

EXP = Path(str(_ROOT / "experiments") + "/rq3_maintenance")
ORDER = ["imdb", "stackexchange", "amazon"]
NICE = {"imdb": "IMDb\n233K", "stackexchange": "StackExch.\n500K",
        "amazon": "Amazon\n1.35M"}
R = {}
for f in glob.glob(str(EXP / "results_*.json")):
    r = json.load(open(f))
    R[r["corpus"]] = r
cs = [c for c in ORDER if c in R]

BLUE, ORANGE, GREEN, GREY = "#0072B2", "#E69F00", "#009E73", "0.45"
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.4, 2.4))

series = [("rescan_ms", lambda r: r["rescan_ms"], ORANGE, "recompute from scratch"),
          ("anchor", lambda r: r["delete_anchor_us"] / 1e3, BLUE, "delete: anchor item"),
          ("plain", lambda r: r["delete_plain_us"] / 1e3, "#56B4E9", "delete: ordinary item"),
          ("read", lambda r: r["maintained_read_ms"], GREEN, "read maintained answer")]
w, xs = 0.2, np.arange(len(cs))
for i, (_, get, col, lab) in enumerate(series):
    ax1.bar(xs + (i - 1.5) * w, [get(R[c]) for c in cs], w, color=col, label=lab)
ax1.set_yscale("log")
ax1.set_xticks(xs); ax1.set_xticklabels([NICE[c] for c in cs], fontsize=7)
ax1.set_ylabel("time per operation (ms)", fontsize=8.5)
ax1.set_title("(a) cost of a fresh answer", fontsize=8.5, pad=6)
ax1.tick_params(axis="y", labelsize=7.5)
ax1.legend(fontsize=5.9, ncol=2, loc="upper center", framealpha=0.93)
ax1.grid(axis="y", alpha=0.25, lw=0.5)
ax1.set_ylim(1e-4, 10 ** 3.6)
# the headline ratio belongs in the caption, not on top of the bars
sp = max(R[c]["rescan_ms"] / R[c]["maintained_read_ms"] for c in cs)

for c, col in zip(cs, [ORANGE, BLUE, GREEN]):
    ck = R[c]["churn"]["checkpoints"]
    ax2.plot([p["ops"] for p in ck], [max(p["rel_err"], 1e-17) for p in ck],
             marker="o", ms=3, lw=1.4, color=col, label=NICE[c].replace("\n", " "))
ax2.set_yscale("log")
ax2.set_xlabel("randomised insert/delete operations", fontsize=8.5)
ax2.set_ylabel("relative error vs recomputation", fontsize=8.5)
ax2.set_title("(b) the state does not drift", fontsize=8.5, pad=6)
ax2.tick_params(labelsize=7.5)
ax2.set_ylim(1e-17, 1e-9)
ax2.grid(alpha=0.25, lw=0.5)
ax2.legend(fontsize=6.0, loc="upper left", framealpha=0.93)
ax2.axhline(1e-15, color=GREY, lw=0.8, ls=":")
ax2.text(0.97, 0.06, "double precision floor", transform=ax2.transAxes,
         fontsize=6.0, color=GREY, ha="right")

plt.tight_layout(pad=0.4)
plt.savefig(Path(str(_ROOT / "figures") + "/fig3_maintenance.pdf"))
print("fig3 written; corpora:", cs, f"max speedup {sp:,.0f}x")
