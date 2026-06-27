"""Evaluation metrics for visual place recognition retrieval.

We treat "which location is this query?" as a retrieval problem: each query is
encoded and matched against the gallery index; a query is *correct at rank r* if
a gallery image of its true location appears within the top-r neighbours.

Metrics
-------
top1_accuracy : fraction of queries whose nearest neighbour has the right label.
recall_at_k   : for each K, fraction of queries with a correct-location image in
                the top-K neighbours (standard VPR Recall@K).
random_baseline_recall_at_k : analytic expected Recall@K for a retriever that
                returns K uniformly-random distinct gallery images, given the
                per-location gallery counts. Used as the "beats random" baseline.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .index import RetrievalIndex


@dataclass
class EvalResult:
    n_queries: int
    top1_accuracy: float
    recall_at_k: dict[int, float]
    random_recall_at_k: dict[int, float]

    def summary_lines(self) -> list[str]:
        lines = [
            f"queries evaluated : {self.n_queries}",
            f"top-1 accuracy    : {self.top1_accuracy:.3f}",
        ]
        for k in sorted(self.recall_at_k):
            lines.append(
                f"Recall@{k:<2d}        : {self.recall_at_k[k]:.3f}   "
                f"(random {self.random_recall_at_k[k]:.3f})"
            )
        return lines


def _random_recall_at_k(k: int, n_gallery: int, n_relevant: int) -> float:
    """Expected Recall@K of drawing K distinct gallery items uniformly.

    P(at least one relevant in K draws) = 1 - C(N-R, K) / C(N, K), computed as a
    product of falling factors to avoid overflow.
    """
    if n_relevant <= 0 or n_gallery <= 0:
        return 0.0
    k = min(k, n_gallery)
    p_none = 1.0
    for i in range(k):
        # probability the i-th draw is also non-relevant
        p_none *= (n_gallery - n_relevant - i) / (n_gallery - i)
        if p_none <= 0:
            return 1.0
    return 1.0 - p_none


def evaluate(
    index: RetrievalIndex,
    query_images: list[np.ndarray],
    query_labels: list[str],
    ks: tuple[int, ...] = (1, 3, 5),
) -> EvalResult:
    """Run retrieval for every query and compute top-1 + Recall@K vs random."""
    if len(query_images) != len(query_labels):
        raise ValueError("query_images and query_labels must align")

    max_k = max(ks)
    gallery_labels = np.array(index.labels)
    n_gallery = index.size

    top1_correct = 0
    recall_hits = {k: 0 for k in ks}

    for img, true_label in zip(query_images, query_labels):
        hits = index.search(img, k=max_k)
        ranked_labels = [h["label"] for h in hits]
        if ranked_labels and ranked_labels[0] == true_label:
            top1_correct += 1
        for k in ks:
            if true_label in ranked_labels[:k]:
                recall_hits[k] += 1

    n = len(query_images)
    recall = {k: recall_hits[k] / n for k in ks}

    # random baseline depends on how many gallery items share each query's label
    rand = {}
    for k in ks:
        per_query = []
        for true_label in query_labels:
            n_rel = int(np.sum(gallery_labels == true_label))
            per_query.append(_random_recall_at_k(k, n_gallery, n_rel))
        rand[k] = float(np.mean(per_query))

    return EvalResult(
        n_queries=n,
        top1_accuracy=top1_correct / n,
        recall_at_k=recall,
        random_recall_at_k=rand,
    )
