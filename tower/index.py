"""A single retrieval index with exact cosine nearest-neighbour search.

Honest scope note
-----------------
In the original BCSAI group project this was a FAISS index that also held DINOv2
ViT and CNN embeddings alongside the classical descriptors. Here there is no GPU
/ torch / faiss available, so this index holds the *classical* global descriptors
(ORB bag-of-visual-words + colour/gradient histograms from ``tower.descriptors``)
and does **exact** cosine nearest-neighbour search with numpy / scikit-learn.

Because descriptors are L2-normalized, cosine similarity is just a dot product,
so a single matrix multiply against the gallery gives exact ranked retrieval —
the same top-K semantics FAISS would give with an exact (IndexFlatIP) index, only
without the approximate-search acceleration.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .descriptors import Vocabulary, build_vocabulary, describe, describe_many


@dataclass
class RetrievalIndex:
    """Exact cosine-similarity retrieval index over classical descriptors.

    Attributes
    ----------
    vocab    : the visual-word vocabulary used to encode images.
    vectors  : (N, D) L2-normalized gallery descriptors.
    labels   : (N,) location label per gallery image.
    paths    : optional source path per gallery image (for display/debug).
    """

    vocab: Vocabulary
    vectors: np.ndarray
    labels: list[str]
    paths: list[str] = field(default_factory=list)
    describe_kw: dict = field(default_factory=dict)

    @property
    def size(self) -> int:
        return self.vectors.shape[0]

    @property
    def dim(self) -> int:
        return self.vectors.shape[1]

    @classmethod
    def build(
        cls,
        images: list[np.ndarray],
        labels: list[str],
        paths: list[str] | None = None,
        n_words: int = 64,
        vocab: Vocabulary | None = None,
        describe_kw: dict | None = None,
    ) -> "RetrievalIndex":
        """Build an index from gallery images + their location labels.

        If ``vocab`` is not supplied, a vocabulary is learned from ``images``.
        """
        if len(images) != len(labels):
            raise ValueError("images and labels must have the same length")
        describe_kw = dict(describe_kw or {})
        if vocab is None:
            vocab = build_vocabulary(images, n_words=n_words)
        vectors = describe_many(images, vocab, **describe_kw)
        return cls(
            vocab=vocab,
            vectors=vectors.astype(np.float32),
            labels=list(labels),
            paths=list(paths) if paths is not None else [],
            describe_kw=describe_kw,
        )

    def encode(self, img: np.ndarray) -> np.ndarray:
        """Encode a single query image into the index's descriptor space."""
        return describe(img, self.vocab, **self.describe_kw)

    def _similarities(self, vec: np.ndarray) -> np.ndarray:
        # vec and gallery rows are L2-normalized -> dot product == cosine sim.
        return self.vectors @ vec.astype(np.float32)

    def search(self, img: np.ndarray, k: int = 5):
        """Return the top-k gallery neighbours for a query image.

        Returns a list of dicts (best first) with keys:
        ``rank``, ``score`` (cosine sim), ``label``, ``db_index``, ``path``.
        """
        vec = self.encode(img)
        sims = self._similarities(vec)
        k = int(min(k, self.size))
        # argpartition for the top-k, then sort those k by score desc.
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]
        results = []
        for rank, idx in enumerate(top):
            results.append(
                {
                    "rank": rank,
                    "score": float(sims[idx]),
                    "label": self.labels[idx],
                    "db_index": int(idx),
                    "path": self.paths[idx] if self.paths else None,
                }
            )
        return results

    def predict(self, img: np.ndarray, k: int = 5) -> str:
        """Predict a location label by majority vote over the top-k neighbours.

        Ties are broken by the summed similarity of the tied labels (so the
        label with the strongest neighbours wins).
        """
        hits = self.search(img, k=k)
        votes: dict[str, int] = {}
        weight: dict[str, float] = {}
        for h in hits:
            votes[h["label"]] = votes.get(h["label"], 0) + 1
            weight[h["label"]] = weight.get(h["label"], 0.0) + h["score"]
        # sort by (votes, summed-similarity) descending
        best = max(votes, key=lambda lbl: (votes[lbl], weight[lbl]))
        return best
