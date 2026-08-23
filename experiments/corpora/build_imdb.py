"""IMDb 460K -> unified corpus schema (controlled corpus).

No re-embedding: the existing title embeddings are reused, so this
corpus stays bit-comparable with every earlier measurement made on it.
Its role in the paper is the CONTROLLED corpus -- ablations, numerical
precision, the certified contract -- not the realism argument.
"""
from pathlib import Path as _P
_ROOT = _P(__file__).resolve().parents[2]
import json, shutil
from pathlib import Path
import numpy as np, pandas as pd

SRC = Path("" + str(_ROOT / "experiments" / "corpora" / "_imdb_source") + "")
OUT = Path("" + str(_ROOT / "experiments") + "/corpora/imdb")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(SRC / "movies.parquet")
emb = np.load(SRC / "emb.npy")
assert len(df) == len(emb), (len(df), len(emb))

out = pd.DataFrame(dict(
    item_id=df.movie_id.astype(str).values,
    facet=df.genre.fillna("(none)").values,
    value=df.rating.astype(float).values,
    filter_num=df.year.astype(float).values,          # release year
    filter_bool=(df.year.values >= 2000),             # the CIDR predicate
    text=df.title.astype(str).values,
))
out.to_parquet(OUT / "corpus.parquet", index=False)
np.save(OUT / "emb.npy", emb.astype(np.float32))
(OUT / "meta.json").write_text(json.dumps(dict(
    corpus="imdb_rated_movies", source="JOB/IMDb rated-movie subset",
    role="controlled corpus: ablations, precision, contract; not the realism argument",
    n_rows=int(len(out)), n_facets=int(out.facet.nunique()),
    value_col="user rating", filter_num_col="release year",
    filter_bool_col="year >= 2000",
    embedding=dict(model="sentence-transformers/all-MiniLM-L6-v2",
                   dim=int(emb.shape[1]), normalized=True,
                   note="reused verbatim from the earlier study so numbers stay comparable"),
), indent=2))
print(f"imdb: {len(out):,} rows, {out.facet.nunique()} facets, emb {emb.shape}")
print(out.groupby('facet').size().sort_values().describe().to_string())
