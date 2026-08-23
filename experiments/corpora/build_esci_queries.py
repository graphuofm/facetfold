"""Real user search queries from ESCI (KDD Cup 2022 Shopping Queries).

Our RQ1 query workload so far is synthesised from the corpus itself
(the first sentence of an item). A reviewer can fairly ask whether the
coverage failure is an artifact of that construction. ESCI supplies
real Amazon search queries -- same product universe as the Amazon
Reviews corpus -- so the same experiment can be re-run against a query
workload nobody constructed for this paper.

ESCI carries no numeric column, so it is used ONLY as a query set,
never as a corpus. That limit is stated in the paper.
"""
from pathlib import Path as _P
_ROOT = _P(__file__).resolve().parents[2]
import argparse, json, time
from pathlib import Path
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

OUT = Path("" + str(_ROOT / "experiments") + "/corpora/esci_queries")
ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=200)
ap.add_argument("--pool", type=int, default=400_000, help="rows to scan")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

ds = load_dataset("tasksource/esci", split="train", streaming=True)
seen, t0 = {}, time.time()
for i, d in enumerate(ds):
    if i >= a.pool:
        break
    if d.get("product_locale") != "us":
        continue
    q = (d.get("query") or "").strip()
    if 2 <= len(q.split()) <= 20 and q.lower() not in seen:
        seen[q.lower()] = q
print(f"{len(seen):,} unique US queries from {a.pool:,} rows "
      f"({time.time()-t0:.0f}s)")

rng = np.random.RandomState(a.seed)
keys = sorted(seen)
pick = rng.choice(len(keys), size=min(a.n, len(keys)), replace=False)
queries = [seen[keys[j]] for j in pick]

m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")
emb = m.encode(queries, batch_size=256, convert_to_numpy=True,
               normalize_embeddings=True)
OUT.mkdir(parents=True, exist_ok=True)
np.save(OUT / "query_emb.npy", emb.astype(np.float32))
(OUT / "queries.json").write_text(json.dumps(dict(
    source="tasksource/esci (KDD Cup 2022 Shopping Queries), us locale",
    role="real user search queries; ESCI has no numeric column so it is "
         "never used as a corpus, only as a query workload",
    n_scanned=a.pool, n_unique=len(seen), n_sampled=len(queries),
    seed=a.seed, encoder="all-MiniLM-L6-v2",
    queries=queries), indent=2))
print(f"wrote {len(queries)} queries, emb {emb.shape}")
print("examples:", queries[:5])
