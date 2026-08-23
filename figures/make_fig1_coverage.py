"""Figure 1: why top-k retrieval does not answer the aggregation query.

(a) Measured facet coverage (solid) against the coverage a UNIFORM
    RANDOM draw of the same size would achieve (dotted). The vertical
    gap between a corpus's two curves is the cost of semantic
    concentration: similarity retrieval looks in far fewer facets than
    a random sample of identical size, so the shortfall is not a
    budget problem that a larger k fixes cheaply.
(b) Whether the returned answer supports the downstream decision
    (picking the most relevant facet). Reported per corpus, never
    pooled: it depends on how separable the ground truth is, which
    differs by corpus.

All numbers read from experiments/rq1_coverage/scaling.json.
"""
from pathlib import Path as _P
_ROOT = _P(__file__).resolve().parent.parent
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
# ACM submissions must not carry Type 3 fonts, which is what matplotlib
# emits by default; 42 selects TrueType so the glyphs embed properly.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt

S = json.load(open(str(_ROOT / "experiments") + "/rq1_coverage/scaling.json"))
GREEN, GREY = "#009E73", "0.45"
STYLE = {"amazon": ("#0072B2", "o", "Amazon Reviews"),
         "imdb": ("#E69F00", "s", "IMDb"),
         "stackexchange": ("#CC79A7", "^", "StackExchange")}

fig, ax1 = plt.subplots(1, 1, figsize=(3.4, 2.02))

for name, c in S["corpora"].items():
    col, mk, lab = STYLE.get(name, ("0.3", "d", name))
    lab = f"{lab} ({c['n_facets']} facets)"
    ks = [p["k"] for p in c["points"]]
    ax1.plot(ks, [p["coverage"] for p in c["points"]], marker=mk, ms=4,
             lw=1.7, color=col, label=lab)
    ax1.plot(ks, [p["uniform_reference"] for p in c["points"]], ls=":",
             lw=1.4, color=col, alpha=0.85)

ax1.plot([], [], ls=":", lw=1.4, color=GREY, label="same $k$, drawn at random")
for ax, ylab, title in ((ax1, "facet coverage",
                         "coverage, against a random draw"),):
    ax.axhline(1.0, color=GREEN, lw=1.6, ls="--", zorder=1,
               label="soft aggregation (exact)")
    ax.set_xscale("log")
    # headroom above 1.0 is for the annotation, but coverage is a
    # fraction: ticks above 1.0 would label impossible values
    ax.set_ylim(0, 1.34)
    ax.set_yticks([0.0, 0.25, 0.50, 0.75, 1.00])
    ax.set_xlabel(r"retrieved items $k$", fontsize=8.5)
    ax.set_ylabel(ylab, fontsize=8.5)
    ax.set_title(title, fontsize=8, pad=6)
    ax.tick_params(labelsize=7.5)
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=5.2, loc="lower right", framealpha=0.93)

worst = max((p for c in S["corpora"].values() for p in c["points"]),
            key=lambda p: p["concentration_gap_pp"])
ax1.text(0.03, 0.97,
         f"similarity retrieval covers up to\n"
         f"{worst['concentration_gap_pp']:.0f} points fewer facets than a\n"
         f"random draw of the same size",
         transform=ax1.transAxes, fontsize=6.2, color=GREY,
         va="top", ha="left", linespacing=1.35)

plt.tight_layout(pad=0.4)
plt.savefig(Path(str(_ROOT / "figures") + "/fig1_coverage.pdf"))
print(f"fig1 rebuilt: {len(S['corpora'])} corpora, worst concentration gap "
      f"{worst['concentration_gap_pp']:.1f} pp")
