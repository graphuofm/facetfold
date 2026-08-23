# Reproducing the paper

Everything below was run on one machine: Intel Core i9-13900K
(24 cores, 8 performance and 16 efficient, 32 threads), 64 GB RAM,
NVIDIA RTX A5000 (used only for embeddings), Ubuntu 22.04.
Exact software versions are in `experiments/ENVIRONMENT.json`.

Total time from a clean checkout: roughly **4 hours**, of which about
40 minutes is downloading and embedding corpora and the rest is
experiments. Disk: about **8 GB** for the rebuilt corpora.

## 0. Prerequisites

```bash
# Rust (for the engine) and Python 3.10+
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
python3 -m pip install numpy pandas pyarrow torch scipy \
                       sentence-transformers datasets huggingface_hub \
                       matplotlib maturin psycopg2-binary duckdb
```

A GPU is optional; embeddings fall back to CPU and take longer.
PostgreSQL 18 with pgvector 0.8+ is needed only for the indexed
baseline of RQ2; every other experiment runs without it.

## 1. Build and install the engine

```bash
cd engine && make python     # builds the wheel and installs it
cargo test --workspace       # ~530 tests, all should pass
```

## 2. Rebuild the corpora  (~40 min, ~8 GB)

```bash
cd experiments/corpora
python3 build_amazon.py            # 1.5M reviews, 33 facets   (~5 min)
python3 build_stackexchange.py     # 1.2M questions, 152 facets (~7 min)
python3 build_imdb.py              # reuses the bundled IMDb subset
python3 build_esci_queries.py      # 200 real product-search queries

python3 embed.py --dir amazon      # ~7 min on a GPU
python3 embed.py --dir stackexchange
python3 embed.py --dir imdb        # IMDb ships with embeddings; skip
python3 embed2.py --dir imdb       # second encoder (robustness check)
```

Sampling is deterministic: the same commands produce the same corpora.
The parameters, and the bias they introduce, are recorded in each
corpus's `meta.json` rather than left implicit.

## 3. Run the experiments

Latency harnesses refuse to record a timing while the machine is busy
(`experiments/quiet.py`), so run them on an otherwise idle machine.
They wait rather than produce a number you should not trust.

```bash
cd experiments

# RQ0  is the retrieval signal sound? (~5 min)
python3 rq0_retriever/run_ndcg.py --n-queries 300

# RQ1  what does a global ranked list return? (~2 min per corpus)
for c in amazon stackexchange imdb; do
  python3 rq1_coverage/run_coverage.py --corpus $c
done
python3 rq1_coverage/run_coverage.py --corpus amazon --query-set esci_queries
python3 rq1_coverage/run_coverage.py --corpus imdb --emb emb_mpnet.npy
python3 rq1_coverage/analyze_scaling.py
python3 rq1_coverage/compare_querysets.py
for c in amazon stackexchange imdb; do
  python3 rq1_coverage/run_baselines.py --corpus $c --n-queries 200
done
python3 rq1_coverage/run_baselines.py --corpus imdb --n-queries 200 \
        --emb emb_mpnet.npy

# per-facet top-b, the counter-design: same total budget, spent within
# each facet instead of across the corpus  (~15 min)
for c in imdb amazon stackexchange; do
  python3 rq1_coverage/run_perfacet.py --corpus $c --n-queries 200
done

# control: does each query's own source item drive the result? (~5 min)
for c in imdb amazon stackexchange; do
  python3 rq1_coverage/run_selfexclude.py --corpus $c --n-queries 200
done

# RQ2  what does the exact answer cost? (~30 min: repeated runs)
python3 rq2_cost/repeat_runs.py --runs 3 --warmup 1 --n-queries 20
python3 rq2_cost/run_indexed_topk.py                              # needs PostgreSQL
python3 rq2_cost/run_indexed_topk.py --iterative relaxed_order --tag _iter

# the same operator written three ways in SQL, against the fused pass
python3 rq2_cost/run_sql_baselines.py --n-queries 20 --reps 3  # needs PostgreSQL

# how the filtered index behaves as the predicate tightens (~20 min)
python3 rq2_cost/run_selectivity.py --n-queries 20 --reps 3    # needs PostgreSQL

# RQ3  what does staying fresh cost? (~10 min)
for c in imdb stackexchange amazon; do
  python3 rq3_maintenance/run_maintenance.py --corpus $c --churn-ops 10000
done
# per-edit cost against the number of standing queries (~15 min)
python3 rq3_maintenance/run_qscaling.py --qs 1 10 100 1000 --facet-rows 20

# RQ4  what does the guarantee cost? (~25 min)
# proposal / verification / repair, priced separately, with a sweep of
# the planning sketch's sample size. Cross-checks its own simulation
# against the engine and exits non-zero if they disagree.
for c in imdb stackexchange amazon; do
  python3 rq4_contract/run_contract2.py --corpus $c --n-queries 200
done

python3 summarise_all.py    # one table of everything, plus cross-checks
```

## 4. Regenerate figures and tables

```bash
cd figures
for f in make_*.py; do python3 $f; done
```

Each generator reads a result file; none contains a hardcoded
measurement. Changing an experiment and re-running the generator is
the only supported way to change a number in the paper.

## 5. Check the paper against the data

```bash
python3 verify_claims.py    # 100 claims, each tied to a file and field
```

This is the check that matters. It failed three times during writing
and caught a real error each time: a claim quoted the range of
per-corpus means as if it were the range over all conditions; a table
caption asserted a significance result that some comparisons did not
support; and the text and a figure aggregated the same quantity two
different ways, which is now read from one frame by both.

The paper source is not part of this artifact, so the checks that a
phrase appears (or no longer appears) in the text are skipped here and
the script says so. Every numeric check runs against
`experiments/*/`.

## Expected variation

Latencies vary by roughly 10% between sessions on a quiet machine and
much more on a busy one; a first, cold run on a corpus can be several
times slower, which is why `repeat_runs.py` discards a warm-up.
Coverage, error, and significance figures are deterministic and should
reproduce exactly.
