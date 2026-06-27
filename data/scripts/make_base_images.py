"""Synthesize a handful of distinct, textured "location" base images.

These stand in for photos of different locations/floors of the IE Tower.
This is a CONTROLLED BENCHMARK: the images are procedurally generated so the
dataset is fully reproducible and committable, not real tower photographs.

Each location gets a deterministic look:
  - a dominant colour palette,
  - a structural pattern (stripes / grid / arcs / blobs),
  - fine texture noise,
so that locations are visually distinct yet internally consistent. Augmented
database/query views are produced later by make_dataset.py.

Run:  python data/scripts/make_base_images.py
"""
from __future__ import annotations

import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.normpath(os.path.join(HERE, "..", "base"))

H, W = 256, 256

# (name, palette as 3 BGR colours, pattern)
LOCATIONS = [
    ("atrium",      [(120, 138, 30), (200, 210, 90), (60, 70, 20)],  "arcs"),
    ("auditorium",  [(110, 70, 40), (180, 120, 80), (60, 35, 20)],   "stripes"),
    ("plaza",       [(40, 110, 200), (90, 170, 240), (20, 60, 130)], "grid"),
    ("library",     [(50, 120, 70), (110, 190, 130), (25, 70, 40)],  "blobs"),
    ("cafeteria",   [(150, 60, 150), (210, 110, 210), (90, 30, 90)], "grid"),
    ("rooftop",     [(180, 150, 60), (230, 210, 120), (120, 100, 35)], "arcs"),
]


def _rng(name: str) -> np.random.Generator:
    # deterministic per-location seed from the name
    seed = abs(hash(name)) % (2**32)
    return np.random.default_rng(seed)


def _base_gradient(palette, rng) -> np.ndarray:
    c0 = np.array(palette[0], dtype=np.float32)
    c1 = np.array(palette[1], dtype=np.float32)
    # diagonal gradient between the two main palette colours
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    t = (xx + yy) / (H + W)
    img = c0[None, None, :] * (1 - t)[..., None] + c1[None, None, :] * t[..., None]
    return img


def _add_pattern(img, palette, pattern, rng):
    accent = np.array(palette[2], dtype=np.float32)
    out = img.copy()
    if pattern == "stripes":
        period = rng.integers(14, 26)
        for x in range(0, W, int(period)):
            cv2.rectangle(out, (x, 0), (x + int(period) // 2, H), accent.tolist(), -1)
    elif pattern == "grid":
        step = int(rng.integers(20, 34))
        for x in range(0, W, step):
            cv2.line(out, (x, 0), (x, H), accent.tolist(), 2)
        for y in range(0, H, step):
            cv2.line(out, (0, y), (W, y), accent.tolist(), 2)
    elif pattern == "arcs":
        for _ in range(rng.integers(8, 14)):
            c = (int(rng.integers(0, W)), int(rng.integers(0, H)))
            r = int(rng.integers(20, 90))
            cv2.circle(out, c, r, accent.tolist(), 2)
    elif pattern == "blobs":
        for _ in range(rng.integers(14, 22)):
            c = (int(rng.integers(0, W)), int(rng.integers(0, H)))
            ax = (int(rng.integers(10, 40)), int(rng.integers(10, 40)))
            ang = int(rng.integers(0, 180))
            cv2.ellipse(out, c, ax, ang, 0, 360, accent.tolist(), -1)
    # blend pattern so palette still reads through
    return cv2.addWeighted(img, 0.45, out, 0.55, 0)


def _add_texture(img, rng) -> np.ndarray:
    noise = rng.normal(0, 14, img.shape).astype(np.float32)
    out = img + noise
    return np.clip(out, 0, 255).astype(np.uint8)


def make_base_image(name: str, palette, pattern) -> np.ndarray:
    rng = _rng(name)
    img = _base_gradient(palette, rng)
    img = _add_pattern(img, palette, pattern, rng)
    img = _add_texture(img, rng)
    return img


def main() -> None:
    os.makedirs(BASE_DIR, exist_ok=True)
    for name, palette, pattern in LOCATIONS:
        img = make_base_image(name, palette, pattern)
        path = os.path.join(BASE_DIR, f"{name}.png")
        cv2.imwrite(path, img)
        print(f"wrote {path}")
    print(f"\n{len(LOCATIONS)} base location images in {BASE_DIR}")


if __name__ == "__main__":
    main()
