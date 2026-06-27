"""Relocalise a query against the IE Tower locations index and report metrics.

Builds the retrieval index from the database split, relocalises one example
query (prints its top-K neighbours + correctness), then runs the full
evaluation over the held-out query split and prints top-1 accuracy and
Recall@K against the random baseline.

Usage:
    python demo.py                 # default example query
    python demo.py --location plaza --view 1
    python demo.py --k 5
"""
from __future__ import annotations

import argparse

from tower.data import load_database, load_queries
from tower.evaluate import evaluate
from tower.index import RetrievalIndex


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--location", default=None,
                    help="true location of the example query (default: first)")
    ap.add_argument("--view", type=int, default=0,
                    help="which query view index of that location to use")
    ap.add_argument("--k", type=int, default=5, help="top-K for retrieval")
    ap.add_argument("--n-words", type=int, default=64,
                    help="visual-word vocabulary size")
    args = ap.parse_args()

    print("Loading controlled tower-locations benchmark "
          "(synthetic, not real photos)...")
    db_imgs, db_labels, db_paths = load_database()
    q_imgs, q_labels, q_paths = load_queries()
    locations = sorted(set(db_labels))
    print(f"  database : {len(db_imgs)} views across {len(locations)} locations")
    print(f"  queries  : {len(q_imgs)} held-out views")
    print(f"  locations: {', '.join(locations)}")

    print("\nBuilding retrieval index "
          f"(ORB bag-of-visual-words + colour/gradient, {args.n_words} words)...")
    index = RetrievalIndex.build(db_imgs, db_labels, paths=db_paths,
                                 n_words=args.n_words)
    print(f"  index size: {index.size} vectors, dim {index.dim}")

    # pick an example query
    target = args.location or q_labels[0]
    cand = [i for i, lbl in enumerate(q_labels) if lbl == target]
    if not cand:
        raise SystemExit(f"No query for location '{target}'. "
                         f"Choose from: {sorted(set(q_labels))}")
    qi = cand[min(args.view, len(cand) - 1)]
    q_img, q_true = q_imgs[qi], q_labels[qi]

    print(f"\nRelocalising query: true location = '{q_true}'  "
          f"(file {q_paths[qi].split(chr(92))[-1]})")
    hits = index.search(q_img, k=args.k)
    print(f"  top-{args.k} retrieved:")
    for h in hits:
        mark = "<-- correct" if h["label"] == q_true else ""
        print(f"    rank {h['rank']}  sim={h['score']:.3f}  {h['label']:<11} {mark}")
    pred = index.predict(q_img, k=args.k)
    ok = "CORRECT" if pred == q_true else "WRONG"
    print(f"  predicted location (top-{args.k} vote): '{pred}'  -> {ok}")

    print("\nFull evaluation over held-out queries:")
    res = evaluate(index, q_imgs, q_labels, ks=(1, 3, 5))
    for line in res.summary_lines():
        print("  " + line)

    r1 = res.recall_at_k[1]
    rb1 = res.random_recall_at_k[1]
    if rb1 > 0:
        print(f"\n  Recall@1 is {r1 / rb1:.1f}x the random baseline "
              f"({r1:.3f} vs {rb1:.3f}).")


if __name__ == "__main__":
    main()
