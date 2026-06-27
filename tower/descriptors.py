"""Classical global image descriptors for place recognition.

Two complementary, hand-crafted descriptor families are combined into one
L2-normalized global vector per image:

1. ORB Bag-of-Visual-Words (BoVW)
   - detect ORB keypoints + binary descriptors,
   - quantise each descriptor against a KMeans vocabulary of "visual words",
   - histogram the word assignments (term-frequency), with optional L2 norm.
   The vocabulary is learned once from a corpus of images (see build_vocabulary).

2. Colour + gradient histograms
   - HSV colour histogram (coarse, captures the palette of a location),
   - histogram of oriented gradients over the whole image (captures structure).

The final image descriptor is the concatenation of the (weighted) BoVW
histogram and the colour/gradient histogram, then L2-normalized so that cosine
similarity == dot product downstream.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from sklearn.cluster import KMeans

# ORB produces 32-byte (256-bit) binary descriptors.
ORB_DESC_BYTES = 32


def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def orb_descriptors(img: np.ndarray, n_features: int = 500) -> np.ndarray:
    """Return ORB descriptors as a float32 array of shape (n_kp, 32).

    Each byte of the binary descriptor is treated as a feature value in [0, 255].
    Returns an empty (0, 32) array when no keypoints are found.
    """
    gray = _to_gray(img)
    orb = cv2.ORB_create(nfeatures=n_features)
    _, desc = orb.detectAndCompute(gray, None)
    if desc is None:
        return np.zeros((0, ORB_DESC_BYTES), dtype=np.float32)
    return desc.astype(np.float32)


@dataclass
class Vocabulary:
    """A KMeans visual-word vocabulary over ORB descriptor space."""

    kmeans: KMeans
    n_words: int

    def assign(self, descriptors: np.ndarray) -> np.ndarray:
        """Assign each descriptor row to its nearest visual word (cluster id)."""
        if descriptors.shape[0] == 0:
            return np.zeros((0,), dtype=np.int64)
        return self.kmeans.predict(descriptors)


def build_vocabulary(
    images: list[np.ndarray],
    n_words: int = 64,
    n_features: int = 500,
    random_state: int = 0,
) -> Vocabulary:
    """Learn a visual-word vocabulary by clustering ORB descriptors.

    Pools ORB descriptors across all training images and runs KMeans with
    ``n_words`` clusters. ``n_words`` is clamped to the number of available
    descriptors so tiny corpora still work.
    """
    pool = [orb_descriptors(im, n_features=n_features) for im in images]
    pool = [d for d in pool if d.shape[0] > 0]
    if not pool:
        raise ValueError("No ORB descriptors found in any training image.")
    stacked = np.vstack(pool)
    k = int(min(n_words, stacked.shape[0]))
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    kmeans.fit(stacked)
    return Vocabulary(kmeans=kmeans, n_words=k)


def bovw_histogram(
    img: np.ndarray, vocab: Vocabulary, n_features: int = 500, l2: bool = True
) -> np.ndarray:
    """Bag-of-visual-words term-frequency histogram for one image."""
    desc = orb_descriptors(img, n_features=n_features)
    words = vocab.assign(desc)
    hist = np.bincount(words, minlength=vocab.n_words).astype(np.float32)
    if hist.sum() > 0:
        hist /= hist.sum()  # term frequency
    if l2:
        n = np.linalg.norm(hist)
        if n > 0:
            hist = hist / n
    return hist


def color_histogram(img: np.ndarray, bins: tuple[int, int, int] = (8, 8, 8)) -> np.ndarray:
    """Normalized HSV colour histogram, flattened."""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, list(bins),
                        [0, 180, 0, 256, 0, 256])
    hist = hist.flatten().astype(np.float32)
    s = hist.sum()
    if s > 0:
        hist /= s
    return hist


def gradient_histogram(img: np.ndarray, n_bins: int = 16) -> np.ndarray:
    """Global histogram of oriented gradients (magnitude-weighted)."""
    gray = _to_gray(img).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    ang = (np.arctan2(gy, gx) + np.pi) * (180.0 / np.pi)  # [0, 360)
    bin_idx = np.minimum((ang / (360.0 / n_bins)).astype(np.int64), n_bins - 1)
    hist = np.bincount(bin_idx.ravel(), weights=mag.ravel(), minlength=n_bins)
    hist = hist.astype(np.float32)
    s = hist.sum()
    if s > 0:
        hist /= s
    return hist


def describe(
    img: np.ndarray,
    vocab: Vocabulary,
    n_features: int = 500,
    bovw_weight: float = 1.0,
    color_weight: float = 1.0,
    grad_weight: float = 1.0,
) -> np.ndarray:
    """Full classical global descriptor: [BoVW | colour | gradient], L2-normalized.

    The three blocks are individually weighted before concatenation, then the
    whole vector is L2-normalized so cosine similarity reduces to a dot product.
    """
    bovw = bovw_histogram(img, vocab, n_features=n_features) * bovw_weight
    color = color_histogram(img) * color_weight
    grad = gradient_histogram(img) * grad_weight
    vec = np.concatenate([bovw, color, grad]).astype(np.float32)
    n = np.linalg.norm(vec)
    if n > 0:
        vec = vec / n
    return vec


def describe_many(images: list[np.ndarray], vocab: Vocabulary, **kw) -> np.ndarray:
    """Stack ``describe`` over a list of images into an (N, D) matrix."""
    return np.vstack([describe(im, vocab, **kw) for im in images]).astype(np.float32)
