"""A control the coverage study needs: is the retriever any good?

The finding in RQ1 is that similarity retrieval concentrates into few
facets. An obvious rival explanation is that our retriever is simply
weak, and that a competent one would spread out. This measures the
retrieval signal directly against human relevance judgements, on a
standard product-search benchmark with published labels (ESCI / KDD
Cup 2022 Shopping Queries), using the ordinary IR metric.

If the encoders score in the range reported for embedding retrievers
on this benchmark, then the concentration observed in RQ1 cannot be
dismissed as an artifact of a broken retriever.

Gain mapping is the one used by the benchmark: Exact 1.0,
Substitute 0.1, Complement 0.01, Irrelevant 0.0.
"""
from pathlib import Path as _P
_ROOT = _P(__file__).resolve().parent.parent.parent
import argparse, json, time
from collections import defaultdict
from pathlib import Path

import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

OUT = Path(str(_ROOT / "experiments") + "/rq0_retriever")
GAIN = {"Exact": 1.0, "Substitute": 0.1, "Complement": 0.01, "Irrelevant": 0.0}

ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="+",
                default=["sentence-transformers/all-MiniLM-L6-v2",
                         "sentence-transformers/all-mpnet-base-v2"])
ap.add_argument("--n-queries", type=int, default=500)
ap.add_argument("--scan", type=int, default=600_000)
ap.add_argument("--ks", type=int, nargs="+", default=[10, 50])
a = ap.parse_args()

ds = load_dataset("tasksource/esci", split="train", streaming=True)
byq = defaultdict(list)
t0 = time.time()
for i, d in enumerate(ds):
    if i >= a.scan:
        break
    if d.get("product_locale") != "us":
        continue
    lab = d.get("esci_label")
    txt = (d.get("product_title") or "").strip()
    if lab in GAIN and txt:
        byq[d["query"]].append((txt, GAIN[lab]))
# keep queries with enough judged products and at least one relevant
qs = [q for q, v in byq.items()
      if len(v) >= 10 and any(g >= 1.0 for _, g in v)][:a.n_queries]
print(f"{len(qs)} judged queries, "
      f"{np.mean([len(byq[q]) for q in qs]):.1f} products each "
      f"({time.time()-t0:.0f}s)", flush=True)


def ndcg(order_gains, k):
    g = np.asarray(order_gains[:k], dtype=float)
    disc = 1.0 / np.log2(np.arange(2, len(g) + 2))
    dcg = float((g * disc).sum())
    ideal = np.sort(np.asarray(order_gains, dtype=float))[::-1][:k]
    idcg = float((ideal * disc[:len(ideal)]).sum())
    return dcg / idcg if idcg > 0 else np.nan


res = {}
for mn in a.models:
    m = SentenceTransformer(mn, device="cuda")
    qemb = m.encode(qs, batch_size=256, convert_to_numpy=True,
                    normalize_embeddings=True, show_progress_bar=False)
    scores = {k: [] for k in a.ks}
    for qi, q in enumerate(qs):
        docs = byq[q]
        demb = m.encode([t for t, _ in docs], batch_size=256,
                        convert_to_numpy=True, normalize_embeddings=True,
                        show_progress_bar=False)
        sim = demb @ qemb[qi]
        order = np.argsort(-sim)
        gains = [docs[j][1] for j in order]
        for k in a.ks:
            scores[k].append(ndcg(gains, k))
    res[mn] = {f"ndcg@{k}": float(np.nanmean(v)) for k, v in scores.items()}
    res[mn]["n_queries"] = len(qs)
    print(f"  {mn}: " + ", ".join(f"nDCG@{k}={np.nanmean(scores[k]):.3f}"
                                  for k in a.ks), flush=True)

summary = dict(
    benchmark="ESCI / KDD Cup 2022 Shopping Queries (us locale)",
    task="rank the judged products of each query by embedding similarity",
    gain_mapping=GAIN, n_queries=len(qs),
    products_per_query=float(np.mean([len(byq[q]) for q in qs])),
    results=res,
    purpose="control for RQ1: rules out the rival explanation that facet "
            "concentration is an artifact of a weak retriever",
)
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "results.json").write_text(json.dumps(summary, indent=2))
print(f"\nwrote {OUT/'results.json'}")
