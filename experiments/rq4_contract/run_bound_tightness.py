"""Is the bound tight, or merely conservative on our corpora?

Section 5 proves that |mu_R - F| <= delta_g (v_max - v_min) is attained,
and Section 7 measures a median slack of 11-47x on real workloads.
Those two statements are easy to confuse: a reader can come away
thinking the bound is loose by construction, when in fact the slack is
a property of the data.

This separates them with a synthetic instance built to sit ON the
bound. Within one facet, put the retained rows at v_min and the omitted
rows at v_max, so mu_R - mu_O is exactly the value range and the
inequality holds with equality. Then sweep delta and report the ratio
bound/error, which should stay at 1.

A second sweep interpolates towards the real thing: the omitted values
are drawn from the same distribution as the retained ones rather than
placed adversarially, which is what a natural corpus looks like, and
the ratio grows. The two curves together say what the paper means: the
bound cannot be improved without assuming something about the omitted
values, and our corpora happen not to be adversarial.
"""
import argparse, json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=10000, help="rows in the facet")
ap.add_argument("--eps", type=float, default=0.05)
ap.add_argument("--deltas", type=float, nargs="+",
                default=[0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002])
ap.add_argument("--vmin", type=float, default=1.0)
ap.add_argument("--vmax", type=float, default=5.0)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
rng = np.random.RandomState(a.seed)
RANGE = a.vmax - a.vmin


def instance(delta_target, adversarial):
    """One facet whose omitted softmax mass is delta_target.

    Scores are chosen so the retained prefix carries exactly
    1 - delta_target of the mass: give the retained rows score 0 and the
    omitted rows a score low enough that their total weight is the
    wanted share.
    """
    n_om = max(1, int(round(a.n * 0.5)))
    n_ret = a.n - n_om
    # weight per omitted row so that omitted/(retained+omitted) = delta
    w_ret_total = float(n_ret)                       # each retained row w=1
    w_om_total = w_ret_total * delta_target / (1 - delta_target)
    w_om = w_om_total / n_om
    if adversarial:
        v_ret = np.full(n_ret, a.vmin)
        v_om = np.full(n_om, a.vmax)
    else:
        v_ret = rng.uniform(a.vmin, a.vmax, n_ret)
        v_om = rng.uniform(a.vmin, a.vmax, n_om)
    mu_R = v_ret.mean()
    F = (w_ret_total * mu_R + w_om_total * v_om.mean()) / (w_ret_total + w_om_total)
    err = abs(mu_R - F)
    delta = w_om_total / (w_ret_total + w_om_total)
    return delta, err, delta * RANGE


rows = []
for adversarial in (True, False):
    for d in a.deltas:
        delta, err, bound = instance(d, adversarial)
        rows.append(dict(regime="adversarial" if adversarial else "natural",
                         delta=delta, error=err, bound=bound,
                         ratio=bound / err if err > 0 else float("inf")))

adv = [r for r in rows if r["regime"] == "adversarial"]
nat = [r for r in rows if r["regime"] == "natural"]
worst_adv = max(r["ratio"] for r in adv)
summary = dict(
    n_rows=a.n, eps=a.eps, value_range=RANGE, deltas=a.deltas,
    note="adversarial places retained values at v_min and omitted at "
         "v_max, the configuration the proof says attains the bound; "
         "natural draws both from the same uniform distribution. "
         "ratio is bound/realised error, so 1.0 means the inequality "
         "is an equality.",
    adversarial_max_ratio=worst_adv,
    natural_min_ratio=min(r["ratio"] for r in nat),
    natural_max_ratio=max(r["ratio"] for r in nat),
    results=rows)
(OUT / "bound_tightness.json").write_text(json.dumps(summary, indent=2))
print(f"adversarial: bound/error in "
      f"[{min(r['ratio'] for r in adv):.4f}, {worst_adv:.4f}]  (1.0 = attained)")
print(f"natural    : bound/error in "
      f"[{min(r['ratio'] for r in nat):.1f}, {max(r['ratio'] for r in nat):.1f}]")
for r in rows:
    print(f"  {r['regime']:12s} delta={r['delta']:.4f}  err={r['error']:.5f}  "
          f"bound={r['bound']:.5f}  ratio={r['ratio']:.3f}")
