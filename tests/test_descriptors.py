"""Tests for classical descriptors and the visual-word vocabulary."""
from __future__ import annotations

import numpy as np

from tower.descriptors import (
    build_vocabulary,
    color_histogram,
    describe,
    gradient_histogram,
    orb_descriptors,
)


def test_vocabulary_builds_and_encodes(dataset):
    """KMeans vocabulary builds, and BoVW encoding yields the expected dim."""
    vocab = build_vocabulary(dataset["db_imgs"], n_words=48)
    assert 0 < vocab.n_words <= 48
    # encode an image -> finite, correctly-sized, L2-normalized vector
    vec = describe(dataset["db_imgs"][0], vocab)
    assert vec.ndim == 1
    assert np.all(np.isfinite(vec))
    assert abs(np.linalg.norm(vec) - 1.0) < 1e-5


def test_orb_descriptors_shape(dataset):
    desc = orb_descriptors(dataset["db_imgs"][0])
    assert desc.ndim == 2
    assert desc.shape[1] == 32  # ORB = 32-byte descriptors


def test_color_and_gradient_histograms_are_distributions(dataset):
    img = dataset["db_imgs"][0]
    ch = color_histogram(img)
    gh = gradient_histogram(img)
    for h in (ch, gh):
        assert np.all(h >= 0)
        assert abs(h.sum() - 1.0) < 1e-4


def test_within_location_similarity_exceeds_across_location(dataset):
    """Descriptor cosine similarity is higher within a location than across.

    This is the core assumption that makes retrieval work.
    """
    imgs = dataset["db_imgs"]
    labels = np.array(dataset["db_labels"])
    vocab = build_vocabulary(imgs, n_words=64)
    vecs = np.vstack([describe(im, vocab) for im in imgs])
    sims = vecs @ vecs.T  # cosine sim (vectors are L2-normalized)

    within, across = [], []
    n = len(imgs)
    for i in range(n):
        for j in range(i + 1, n):
            (within if labels[i] == labels[j] else across).append(sims[i, j])

    mean_within = float(np.mean(within))
    mean_across = float(np.mean(across))
    assert mean_within > mean_across, (mean_within, mean_across)
    # and by a clear margin
    assert mean_within - mean_across > 0.05
