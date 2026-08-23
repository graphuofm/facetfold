"""Second encoder, to test whether the findings depend on the embedding.

A reviewer will reasonably ask whether "similarity retrieval
concentrates into few facets" is a property of retrieval or of one
particular encoder. This re-embeds a corpus with a different
architecture and dimensionality (MPNet, 768-d, vs MiniLM, 384-d) so the
coverage study can be repeated against an independent representation.
"""
import argparse, json, time
from pathlib import Path
import numpy as np, pandas as pd
from sentence_transformers import SentenceTransformer

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
ap.add_argument("--model", default="sentence-transformers/all-mpnet-base-v2")
ap.add_argument("--batch", type=int, default=512)
ap.add_argument("--out", default="emb_mpnet.npy")
a = ap.parse_args()
d = Path(a.dir)
df = pd.read_parquet(d / "corpus.parquet", columns=["text"])
m = SentenceTransformer(a.model, device="cuda")
t0 = time.time()
e = m.encode(df.text.tolist(), batch_size=a.batch, convert_to_numpy=True,
             normalize_embeddings=True, show_progress_bar=True)
w = time.time() - t0
np.save(d / a.out, e.astype(np.float32))
mp = d / "meta.json"; meta = json.loads(mp.read_text())
meta.setdefault("alt_embeddings", {})[a.out] = dict(
    model=a.model, dim=int(e.shape[1]), normalized=True,
    seconds=round(w, 1), purpose="encoder-robustness check")
mp.write_text(json.dumps(meta, indent=2))
print(f"\n{a.out} {e.shape} in {w:.0f}s")
