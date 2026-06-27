"""Tests for the exact cosine-NN retrieval index."""
from __future__ import annotations

import numpy as np


def test_index_builds_with_expected_shape(index, dataset):
    assert index.size == len(dataset["db_imgs"])
    assert index.dim > 0
    # gallery vectors are L2-normalized
    norms = np.linalg.norm(index.vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4)


def test_same_location_query_retrieves_correct_top1_for_clear_cases(index, dataset):
    """For clear in-database cases, the nearest neighbour shares the label.

    Each gallery image queried against the index must return itself (or another
    image of the same location) at rank 0.
    """
    for img, label in zip(dataset["db_imgs"], dataset["db_labels"]):
        hits = index.search(img, k=1)
        assert hits[0]["label"] == label


def test_query_split_top1_is_strong(index, dataset):
    """Held-out queries are relocalised correctly at top-1 well above chance."""
    correct = 0
    for img, label in zip(dataset["q_imgs"], dataset["q_labels"]):
        if index.search(img, k=1)[0]["label"] == label:
            correct += 1
    acc = correct / len(dataset["q_imgs"])
    n_loc = len(set(dataset["db_labels"]))
    # comfortably better than 1/n_loc random guessing
    assert acc > 0.7
    assert acc > 2.0 / n_loc


def test_predict_majority_vote(index, dataset):
    """predict() returns a valid known location label."""
    known = set(dataset["db_labels"])
    pred = index.predict(dataset["q_imgs"][0], k=5)
    assert pred in known


def test_search_results_sorted_descending(index, dataset):
    hits = index.search(dataset["q_imgs"][0], k=5)
    scores = [h["score"] for h in hits]
    assert scores == sorted(scores, reverse=True)
    assert all(-1.0001 <= s <= 1.0001 for s in scores)
