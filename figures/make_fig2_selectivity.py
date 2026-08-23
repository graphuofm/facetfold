"""Figure 2: how the filtered-index route behaves as the predicate tightens.

The earlier version of this figure compared one filtered-ANN
configuration against the exact pass at one selectivity, which cannot
support a general statement about filtered vector search. This sweeps
the predicate instead, and the result is a crossover rather than a
verdict: with no filter the index is several times faster than the
exact pass, and it stays faster when the predicate is very selective
because the admitted set is then small; in between, once the vendor's
iterative index scan is turned on to recover the facets the index was
dropping, the exact pass is competitive or cheaper.

(a) what the index returns: the fraction of the requested k that
    survives the predicate (fill), and the facet coverage that follows
    from it.
(b) what it costs, against one exact pass over the admitted rows.

Numbers read from experiments/rq2_cost/selectivity_imdb.json.
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

EXP = Path(str(_ROOT / "experiments") + "/rq2_cost")
OUT = Path(__file__).resolve().parent / "fig2_selectivity.pdf"

d = json.load(open(EXP / "selectivity_imdb.json"))
R = sorted(d["results"], key=lambda r: r["target_selectivity"])
sel = [r["target_selectivity"] * 100 for r in R]

fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.5))

a = ax[0]
a.plot(sel, [r["ann_fill"] * 100 for r in R], "o-", color="#c44e52",
       label="index fill, as shipped")
a.plot(sel, [r["ann_coverage"] * 100 for r in R], "o--", color="#c44e52",
       alpha=0.55, label="facet coverage, as shipped")
a.plot(sel, [r["iter_coverage"] * 100 for r in R], "s-", color="#4c72b0",
       label="facet coverage, iterative scan")
a.set_xscale("log")
a.set_xlabel("predicate selectivity (% of rows admitted)")
a.set_ylabel("percent")
a.set_ylim(-3, 105)
a.set_title("(a) what the index returns", fontsize=9)
a.legend(fontsize=6.5, frameon=False, loc="center right")
a.grid(alpha=0.25)

b = ax[1]
b.plot(sel, [r["ann_ms"] for r in R], "o-", color="#c44e52",
       label="index, as shipped")
b.plot(sel, [r["iter_ms"] for r in R], "s-", color="#4c72b0",
       label="index + iterative scan")
b.plot(sel, [r["exact_ms"] for r in R], "^-", color="#55a868",
       label="exact pass")
b.set_xscale("log")
b.set_yscale("log")
b.set_xlabel("predicate selectivity (% of rows admitted)")
b.set_ylabel("latency (ms)")
b.set_title("(b) what it costs", fontsize=9)
b.legend(fontsize=6.5, frameon=False, loc="upper left")
b.grid(alpha=0.25, which="both")

for x in ax:
    x.set_xticks(sel)
    x.set_xticklabels([f"{s:g}" for s in sel], fontsize=7)
    x.tick_params(labelsize=7)
    x.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())

fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight")
print(f"wrote {OUT}")
worst = max(R, key=lambda r: r["iter_ms"] / r["exact_ms"])
best = min(R, key=lambda r: r["iter_ms"] / r["exact_ms"])
print(f"  iterative/exact ratio: worst {worst['iter_ms']/worst['exact_ms']:.2f}x "
      f"at {worst['target_selectivity']:.0%}, best "
      f"{best['iter_ms']/best['exact_ms']:.2f}x at {best['target_selectivity']:.0%}")
print(f"  minimum index fill: {min(r['ann_fill'] for r in R):.4f} "
      f"at {min(R, key=lambda r: r['ann_fill'])['target_selectivity']:.0%}")
