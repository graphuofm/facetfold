"""StackExchange -> unified corpus schema (second real corpus).

Source: donfu/oa-stackexchange (question/answer pairs mined from the
public StackExchange dumps, one record per accepted answer).

  facet       the StackExchange site (mythology, physics, cooking, ...)
              -- ~170 natural facets, far more than Amazon's 33
  value       answer_score (community votes on the answer)
  filter_num  question_score (the exact predicate: well-received questions)
  text        the question, which is what a user would search with

Its role in the paper: show the RQ1 finding is not an artifact of one
domain. Q&A search is also squarely WSDM territory.
"""
from pathlib import Path as _P
_ROOT = _P(__file__).resolve().parents[2]
import argparse, json, time
from pathlib import Path
import pandas as pd
from datasets import load_dataset

OUT = Path("" + str(_ROOT / "experiments") + "/corpora/stackexchange")

ap = argparse.ArgumentParser()
ap.add_argument("--cap", type=int, default=1_200_000, help="max rows kept")
ap.add_argument("--cap-per-facet", type=int, default=60_000)
a = ap.parse_args()

ds = load_dataset("donfu/oa-stackexchange", split="train", streaming=True)
rows, per_facet, t0 = [], {}, time.time()
for i, d in enumerate(ds):
    md = d.get("METADATA") or {}
    site = (d.get("SOURCE") or "").replace("stackexchange-", "").strip()
    q = (d.get("INSTRUCTION") or "").strip()
    ans, qs = md.get("answer_score"), md.get("question_score")
    if not site or not q or ans is None or qs is None:
        continue
    if per_facet.get(site, 0) >= a.cap_per_facet:
        continue
    per_facet[site] = per_facet.get(site, 0) + 1
    rows.append(dict(item_id=f"{site}:{i}", facet=site, value=float(ans),
                     filter_num=float(qs), filter_bool=bool(qs >= 3),
                     text=q.replace("\n", " ")[:1000]))
    if len(rows) >= a.cap:
        break
    if len(rows) % 200_000 == 0:
        print(f"  {len(rows):,} rows, {len(per_facet)} facets "
              f"({time.time()-t0:.0f}s)", flush=True)

df = pd.DataFrame(rows)
OUT.mkdir(parents=True, exist_ok=True)
df.to_parquet(OUT / "corpus.parquet", index=False)
(OUT / "meta.json").write_text(json.dumps(dict(
    corpus="stackexchange", source="donfu/oa-stackexchange",
    role="second real corpus: shows RQ1 is not an artifact of one domain",
    sampling=dict(cap_rows=a.cap, cap_per_facet=a.cap_per_facet,
                  note="streamed in dataset order; per-facet cap keeps the "
                       "largest sites from swamping the corpus"),
    n_rows=int(len(df)), n_facets=int(df.facet.nunique()),
    value_col="answer_score", filter_num_col="question_score",
    filter_bool_col="question_score >= 3",
    seconds=round(time.time() - t0, 1),
), indent=2))
print(f"\nWROTE {len(df):,} rows, {df.facet.nunique()} facets "
      f"({time.time()-t0:.0f}s)")
print(df.groupby('facet').size().describe().to_string())
