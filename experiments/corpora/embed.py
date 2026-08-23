"""Embed a unified-schema corpus's `text` column.

Model: all-MiniLM-L6-v2 (384-d, normalized) -- the same encoder used
across every corpus in this paper so cross-corpus numbers are
comparable. Writes emb.npy (float32, row-aligned with corpus.parquet)
and records the encoder, dimension, and wall time in meta.json.
"""
import argparse, json, time
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True, help="corpus dir with corpus.parquet")
ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
ap.add_argument("--batch", type=int, default=1024)
a = ap.parse_args()

d = Path(a.dir)
df = pd.read_parquet(d / "corpus.parquet", columns=["text"])
print(f"{len(df):,} texts from {d/'corpus.parquet'}", flush=True)

m = SentenceTransformer(a.model, device="cuda")
t0 = time.time()
emb = m.encode(df.text.tolist(), batch_size=a.batch, convert_to_numpy=True,
               normalize_embeddings=True, show_progress_bar=True)
wall = time.time() - t0
np.save(d / "emb.npy", emb.astype(np.float32))
print(f"\nemb.npy {emb.shape} {emb.dtype} in {wall:.1f}s "
      f"({len(df)/wall:,.0f} texts/s)")

mp = d / "meta.json"
meta = json.loads(mp.read_text()) if mp.exists() else {}
meta["embedding"] = dict(model=a.model, dim=int(emb.shape[1]),
                         normalized=True, dtype="float32",
                         seconds=round(wall, 1), device="cuda")
mp.write_text(json.dumps(meta, indent=2))
