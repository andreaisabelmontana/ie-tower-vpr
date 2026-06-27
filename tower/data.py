"""Loading helpers for the controlled tower-locations dataset.

Reads the augmented database / query views produced by
``data/scripts/make_dataset.py`` from disk into (images, labels, paths).
"""
from __future__ import annotations

import os

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.normpath(os.path.join(_HERE, "..", "data"))
DB_DIR = os.path.join(DATA_ROOT, "database")
Q_DIR = os.path.join(DATA_ROOT, "queries")


def load_split(root: str):
    """Load every image under root/<label>/*.png.

    Returns (images, labels, paths) as parallel lists.
    """
    images: list[np.ndarray] = []
    labels: list[str] = []
    paths: list[str] = []
    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"{root} not found. Run data/scripts/make_base_images.py then "
            "data/scripts/make_dataset.py first."
        )
    for label in sorted(os.listdir(root)):
        loc_dir = os.path.join(root, label)
        if not os.path.isdir(loc_dir):
            continue
        for fn in sorted(os.listdir(loc_dir)):
            if not fn.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            path = os.path.join(loc_dir, fn)
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                continue
            images.append(img)
            labels.append(label)
            paths.append(path)
    if not images:
        raise FileNotFoundError(f"No images loaded from {root}.")
    return images, labels, paths


def load_database():
    """Load the gallery (database) split."""
    return load_split(DB_DIR)


def load_queries():
    """Load the held-out query split."""
    return load_split(Q_DIR)
