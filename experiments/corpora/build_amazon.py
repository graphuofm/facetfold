"""X1 / Amazon Reviews 2023 -> unified corpus schema.

Unified schema shared by all four WSDM corpora, so one experimental
harness runs everywhere:

    item_id     str    stable id
    facet       str    the GROUP BY column (here: product category)
    value       float  the aggregated numeric (here: star rating)
    filter_num  float  numeric column for the exact predicate
                       (here: review year)
    filter_bool bool   boolean predicate column
                       (here: verified purchase)
    text        str    the field that gets embedded

Sampling: each category file is streamed over HTTP and sampled with a
fixed stride over a bounded prefix -- we never download a whole 30 GB
category to keep 50k rows. The stride (rather than the first N lines)
avoids taking one contiguous clump of the file. Both the prefix bound
and the stride are recorded in meta.json so the sample is reproducible
and its bias is stated rather than hidden.
"""
from pathlib import Path as _P
_ROOT = _P(__file__).resolve().parents[2]
import argparse, gzip, io, json, os, sys, time
from pathlib import Path

import pandas as pd
import requests

REPO = "McAuley-Lab/Amazon-Reviews-2023"
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/main/raw/review_categories"
OUT = Path("" + str(_ROOT / "experiments") + "/corpora/amazon")


def stream_category(cat, cap, stride, prefix_cap):
    """Yield up to `cap` sampled reviews from one category file."""
    url = f"{BASE}/{cat}.jsonl"
    kept, seen = [], 0
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        for line in r.iter_lines(chunk_size=1 << 20, decode_unicode=False):
            if not line:
                continue
            seen += 1
            if seen > prefix_cap or len(kept) >= cap:
                break
            if (seen - 1) % stride:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            title = (d.get("title") or "").strip()
            body = (d.get("text") or "").strip()
            text = (title + ". " + body).strip()
            rating = d.get("rating")
            ts = d.get("timestamp")
            if not text or rating is None or ts is None:
                continue
            kept.append(
                dict(
                    item_id=f"{cat}:{seen}",
                    facet=cat,
                    value=float(rating),
                    filter_num=float(time.gmtime(ts / 1000).tm_year),
                    filter_bool=bool(d.get("verified_purchase", False)),
                    text=text[:1000],
                )
            )
    return kept, seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=50_000, help="rows per category")
    ap.add_argument("--stride", type=int, default=7)
    ap.add_argument("--prefix-cap", type=int, default=2_000_000)
    ap.add_argument("--categories", type=str, default="")
    a = ap.parse_args()

    cats = [c for c in a.categories.split(",") if c] or [
        "Subscription_Boxes", "Magazine_Subscriptions", "Gift_Cards",
        "Digital_Music", "Health_and_Personal_Care", "Handmade_Products",
        "All_Beauty", "Appliances", "Amazon_Fashion", "Musical_Instruments",
        "Software", "Industrial_and_Scientific", "Video_Games",
        "Baby_Products", "CDs_and_Vinyl", "Arts_Crafts_and_Sewing",
        "Office_Products", "Grocery_and_Gourmet_Food", "Toys_and_Games",
        "Patio_Lawn_and_Garden", "Pet_Supplies", "Movies_and_TV",
        "Automotive", "Sports_and_Outdoors", "Cell_Phones_and_Accessories",
        "Beauty_and_Personal_Care", "Health_and_Household",
        "Tools_and_Home_Improvement", "Kindle_Store", "Books",
        "Electronics", "Clothing_Shoes_and_Jewelry", "Home_and_Kitchen",
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    frames, report = [], []
    for i, cat in enumerate(cats, 1):
        t0 = time.time()
        try:
            rows, seen = stream_category(cat, a.cap, a.stride, a.prefix_cap)
        except Exception as e:
            print(f"[{i}/{len(cats)}] {cat}: FAILED {type(e).__name__}: {e}", flush=True)
            report.append(dict(category=cat, kept=0, error=str(e)[:120]))
            continue
        frames.append(pd.DataFrame(rows))
        report.append(dict(category=cat, kept=len(rows), lines_read=seen,
                           seconds=round(time.time() - t0, 1)))
        print(f"[{i}/{len(cats)}] {cat}: {len(rows)} rows "
              f"({seen} lines, {time.time()-t0:.0f}s)", flush=True)

    df = pd.concat(frames, ignore_index=True)
    df.to_parquet(OUT / "corpus.parquet", index=False)
    meta = dict(
        corpus="amazon_reviews_2023", source=REPO,
        sampling=dict(cap_per_category=a.cap, stride=a.stride,
                      prefix_cap=a.prefix_cap,
                      note="strided sample over a bounded file prefix; "
                           "not a uniform random sample of the full category"),
        n_rows=int(len(df)), n_facets=int(df.facet.nunique()),
        value_col="rating (1-5 stars)",
        filter_num_col="review year", filter_bool_col="verified_purchase",
        per_category=report,
    )
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nWROTE {OUT/'corpus.parquet'}: {len(df):,} rows, "
          f"{df.facet.nunique()} facets")
    print(df.groupby('facet').size().describe().to_string())


if __name__ == "__main__":
    main()
