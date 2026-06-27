"""IE Tower VPR — classical visual place recognition by image retrieval.

A small, runnable image-retrieval relocalisation pipeline:

  descriptors -> ORB bag-of-visual-words + colour/gradient histograms
  index       -> single exact cosine-NN retrieval index (numpy/sklearn)
  evaluate    -> top-1 accuracy + Recall@K vs a random baseline
  data        -> loaders for the controlled tower-locations benchmark

Scope: classical descriptors only. The original group project also fused DINOv2
ViT and CNN embeddings into a FAISS index; those need a GPU / torch and are out
of scope in this repo.
"""
from .descriptors import (
    Vocabulary,
    bovw_histogram,
    build_vocabulary,
    color_histogram,
    describe,
    describe_many,
    gradient_histogram,
    orb_descriptors,
)
from .evaluate import EvalResult, evaluate
from .index import RetrievalIndex

__all__ = [
    "Vocabulary",
    "build_vocabulary",
    "orb_descriptors",
    "bovw_histogram",
    "color_histogram",
    "gradient_histogram",
    "describe",
    "describe_many",
    "RetrievalIndex",
    "evaluate",
    "EvalResult",
]
