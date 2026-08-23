# facetfold

Relevance-weighted aggregation over retrieved sets, as a first-class
query operator.

Given a corpus whose items carry a facet label, a numeric attribute,
and an embedding, `facetfold` answers questions of the form

> for each facet, the average of a numeric attribute over the items
> passing an exact filter, weighted by each item's relevance to a query

in one pass, with three properties that a retrieve-then-aggregate
pipeline does not have:

* **complete** — every item admitted by the filter contributes, so no
  facet silently disappears from the answer;
* **fresh** — the per-facet state is three numbers and is additive, so
  an insert or a delete updates the answer instead of triggering a
  recomputation;
* **honestly approximate** — a query may declare an absolute error
  budget; the planner proposes a truncation from a 1024-row sketch or
  declines, execution verifies it against the omitted weight actually
  measured, and any facet whose bound misses is re-folded exactly. The
  certified bound is `delta_g * (v_max - v_min)` over the group's own
  observed value range, which is tight in the worst case.

The operator's semantics are a per-group Nadaraya--Watson kernel
regression estimator with a softmax kernel, and its evaluation
specialises the max-shifted online softmax accumulator to relational
grouping under exact predicates. Neither is new; what this repository
implements is what a data system adds on top of them.

It contains the engine, the experimental harnesses, and the result
files behind every number in the accompanying paper. The strongest
alternative designs are implemented here too and are not straw men:
per-facet (grouped) retrieval at equal budget, uniform and stratified
sampling, a filtered ANN index with its vendor's iterative-scan remedy
swept across predicate selectivity, and the same aggregate written
three ways in SQL.

## Layout

```
engine/         Rust workspace: the operator, query layer, planner,
                maintained views, PostgreSQL extension, CDC client
experiments/    corpus builders, the research-question harnesses, and
                the result files they produced. Notable ones:
                  rq1_coverage/run_perfacet.py    per-facet top-b
                  rq1_coverage/run_selfexclude.py query-item control
                  rq2_cost/run_sql_baselines.py   naive/stable/guarded SQL
                  rq2_cost/run_selectivity.py     filtered-index sweep
                  rq3_maintenance/run_qscaling.py Q standing queries
                  rq4_contract/run_contract2.py   propose/verify/repair
figures/        figure and table generators; each reads a result file
                and hardcodes nothing
verify_claims.py  checks every numeric claim in the paper against the
                result files; run it before trusting the text. The
                paper source is not shipped, so the text-presence
                checks are skipped and it says so; the 100 numeric
                checks all run.
```

Corpora are **not** shipped (6.5 GB of parquet and embeddings). They
are rebuilt by scripts in `experiments/corpora/`, which stream from
public sources. See `REPRODUCE.md`.

## Anonymity

This repository is meant to be readable by reviewers while the paper is
under double-blind review, so it carries no author name, institutional
email, home directory, or cluster name, and its commit metadata is
neutral. `./check_anonymity.sh` re-checks all of that over every tracked
file and exits non-zero if anything slips back in; the `engine/` tree is
copied from a working repository that is *not* anonymous, so run it
before every push.

## Note on naming

The Rust crates and the Python package are still named after an
earlier working title for this project. They are the same code; only
the project name changed.
