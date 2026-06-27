"""Tests for evaluation metrics: Recall@K must beat the random baseline."""
from __future__ import annotations

from tower.evaluate import _random_recall_at_k, evaluate


def test_recall_at_k_beats_random_by_clear_margin(index, dataset):
    res = evaluate(index, dataset["q_imgs"], dataset["q_labels"], ks=(1, 3, 5))
    for k in (1, 3, 5):
        got = res.recall_at_k[k]
        rand = res.random_recall_at_k[k]
        # clear margin over random retrieval at every K
        assert got > rand + 0.2, (k, got, rand)
    # Recall is monotonic non-decreasing in K
    assert res.recall_at_k[1] <= res.recall_at_k[3] <= res.recall_at_k[5]


def test_top1_accuracy_high(index, dataset):
    res = evaluate(index, dataset["q_imgs"], dataset["q_labels"], ks=(1,))
    assert res.top1_accuracy > 0.7


def test_random_baseline_formula():
    # K=N with R relevant in N gallery -> always at least one relevant -> 1.0
    assert abs(_random_recall_at_k(10, 10, 2) - 1.0) < 1e-9
    # single draw from N with R relevant -> R/N
    assert abs(_random_recall_at_k(1, 6, 1) - (1 / 6)) < 1e-9
    # no relevant items -> 0
    assert _random_recall_at_k(5, 10, 0) == 0.0
    # monotonic non-decreasing in K
    r1 = _random_recall_at_k(1, 30, 5)
    r3 = _random_recall_at_k(3, 30, 5)
    assert r3 >= r1
