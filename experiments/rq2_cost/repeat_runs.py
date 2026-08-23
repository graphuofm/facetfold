"""Latency is reported from repeated INDEPENDENT runs, not from one.

Two things forced this. Timings taken while a GPU job ran disagreed
with quiet ones by up to 1.7x (hence quiet.py). And running corpora
back to back in one loop left the previous corpus's arrays in page
cache, so the next process met memory pressure and measured ~1.8x
slow: a single median is therefore not a measurement, it is one draw.

This driver launches each corpus in its own process, several times,
with a settling gap between runs, and reports the median of the
per-run medians together with the spread. A wide spread is a finding
about the measurement, not something to average away, so it is
recorded rather than hidden.
"""
import argparse, json, statistics, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).parent
ap = argparse.ArgumentParser()
ap.add_argument("--corpora", nargs="+", default=["imdb", "stackexchange", "amazon"])
ap.add_argument("--runs", type=int, default=3)
ap.add_argument("--n-queries", type=int, default=20)
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--settle", type=int, default=30, help="seconds between runs")
ap.add_argument("--warmup", type=int, default=1,
                help="runs to discard first: the first touch of a corpus "
                     "pays cold page cache and library init, which showed "
                     "up as a 3.2x outlier on the first corpus measured")
a = ap.parse_args()

out = {}
for corpus in a.corpora:
    runs = []
    for i in range(a.runs + a.warmup):
        if i:
            time.sleep(a.settle)          # let page cache and clocks settle
        r = subprocess.run(
            [sys.executable, str(HERE / "run_cost.py"), "--corpus", corpus,
             "--n-queries", str(a.n_queries), "--reps", str(a.reps)],
            capture_output=True, text=True, cwd=HERE)
        if r.returncode:
            print(f"{corpus} run {i+1} FAILED:\n{r.stdout[-600:]}{r.stderr[-600:]}")
            continue
        res = json.load(open(HERE / f"results_{corpus}.json"))
        if i < a.warmup:
            print(f"  {corpus} warm-up {i+1}/{a.warmup} discarded", flush=True)
            continue
        runs.append({k: v["median"] for k, v in res["latency_ms"].items()})
        print(f"  {corpus} run {i+1-a.warmup}/{a.runs}: "
              + "  ".join(f"{k.replace('_ms','')}={v:.1f}" for k, v in runs[-1].items()),
              flush=True)
    if not runs:
        continue
    keys = runs[0].keys()
    out[corpus] = {k: dict(median_of_runs=round(statistics.median(r[k] for r in runs), 2),
                           min=round(min(r[k] for r in runs), 2),
                           max=round(max(r[k] for r in runs), 2),
                           spread_ratio=round(max(r[k] for r in runs)
                                              / max(min(r[k] for r in runs), 1e-9), 2),
                           runs=[round(r[k], 2) for r in runs])
                   for k in keys}

# Merge rather than overwrite: running one corpus must not destroy the
# other corpora's measurements, which is exactly what happened once.
prev = {}
if (HERE / "results_repeated.json").exists():
    prev = json.load(open(HERE / "results_repeated.json")).get("results", {})
prev.update(out)
out = prev

summary = dict(
    protocol=dict(
        independent_runs=a.runs, warmup_runs_discarded=a.warmup,
        queries_per_run=a.n_queries,
        reps_per_query=a.reps, settle_seconds=a.settle,
        note="each run is a separate process; the machine-quiet gate in "
             "quiet.py guards every one of them"),
    results=out)
(HERE / "results_repeated.json").write_text(json.dumps(summary, indent=2))
print("\n" + f"{'corpus':<15}{'metric':<18}{'median':>9}{'min':>9}{'max':>9}{'spread':>8}")
for c, d in out.items():
    for k, v in d.items():
        print(f"{c:<15}{k.replace('_ms',''):<18}{v['median_of_runs']:>9.1f}"
              f"{v['min']:>9.1f}{v['max']:>9.1f}{v['spread_ratio']:>8.2f}x")
