"""Build a controlled "tower locations" retrieval dataset.

From each base location image we synthesize several augmented VIEWS by applying
photometric and geometric perturbations meant to mimic real-world capture
variation:
  - lighting (brightness / gamma / contrast shifts),
  - viewpoint (small rotations, scaling, perspective warps, crops),
  - blur (Gaussian),
  - JPEG-style compression noise.

Views are split into a DATABASE set (the gallery we index) and a QUERY set
(held-out views we try to relocalise). Database and query views of the same
location are produced with *different* augmentation seeds, so a query is never
a pixel-identical copy of a database image.

This is a CONTROLLED BENCHMARK, not real IE Tower photographs.

Layout produced under data/:
    database/<location>/view_XX.png
    queries/<location>/view_XX.png

Run:  python data/scripts/make_dataset.py
"""
from __future__ import annotations

import os
import shutil

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(HERE, ".."))
BASE_DIR = os.path.join(DATA_DIR, "base")
DB_DIR = os.path.join(DATA_DIR, "database")
Q_DIR = os.path.join(DATA_DIR, "queries")

N_DB_PER_LOC = 6
N_Q_PER_LOC = 4


def _brightness_gamma(img, rng):
    beta = float(rng.uniform(-45, 45))           # brightness
    alpha = float(rng.uniform(0.75, 1.3))        # contrast
    out = np.clip(alpha * img.astype(np.float32) + beta, 0, 255).astype(np.uint8)
    gamma = float(rng.uniform(0.7, 1.5))
    lut = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)],
                   dtype=np.uint8)
    return cv2.LUT(out, lut)


def _geometric(img, rng):
    h, w = img.shape[:2]
    # rotation + scale
    angle = float(rng.uniform(-18, 18))
    scale = float(rng.uniform(0.85, 1.15))
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    out = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    # mild perspective warp
    d = 0.06 * w
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = src + rng.uniform(-d, d, src.shape).astype(np.float32)
    P = cv2.getPerspectiveTransform(src, dst)
    out = cv2.warpPerspective(out, P, (w, h), borderMode=cv2.BORDER_REFLECT)
    # random crop back to full size (slight translation/zoom)
    cy, cx = int(rng.uniform(0, 0.1 * h)), int(rng.uniform(0, 0.1 * w))
    out = out[cy:h - 1, cx:w - 1]
    return cv2.resize(out, (w, h))


def _blur_noise(img, rng):
    out = img
    if rng.random() < 0.7:
        k = int(rng.choice([3, 5, 7]))
        out = cv2.GaussianBlur(out, (k, k), 0)
    if rng.random() < 0.5:
        # JPEG-style compression artefacts
        q = int(rng.integers(35, 80))
        ok, enc = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, q])
        if ok:
            out = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return out


def augment(img, rng):
    out = _geometric(img, rng)
    out = _brightness_gamma(out, rng)
    out = _blur_noise(out, rng)
    return out


def _gen_views(name, base_img, n, out_root, seed_offset):
    loc_dir = os.path.join(out_root, name)
    os.makedirs(loc_dir, exist_ok=True)
    seed = (abs(hash(name)) + seed_offset) % (2**32)
    rng = np.random.default_rng(seed)
    for i in range(n):
        view = augment(base_img, rng)
        cv2.imwrite(os.path.join(loc_dir, f"view_{i:02d}.png"), view)


def main() -> None:
    if not os.path.isdir(BASE_DIR) or not os.listdir(BASE_DIR):
        raise SystemExit(
            "No base images found. Run data/scripts/make_base_images.py first."
        )
    for d in (DB_DIR, Q_DIR):
        if os.path.isdir(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

    bases = sorted(f for f in os.listdir(BASE_DIR) if f.endswith(".png"))
    for f in bases:
        name = os.path.splitext(f)[0]
        base_img = cv2.imread(os.path.join(BASE_DIR, f), cv2.IMREAD_COLOR)
        # database and query use disjoint seed offsets -> different views
        _gen_views(name, base_img, N_DB_PER_LOC, DB_DIR, seed_offset=1)
        _gen_views(name, base_img, N_Q_PER_LOC, Q_DIR, seed_offset=99991)

    n_loc = len(bases)
    print(f"locations: {n_loc}")
    print(f"database:  {n_loc * N_DB_PER_LOC} views  -> {DB_DIR}")
    print(f"queries:   {n_loc * N_Q_PER_LOC} views  -> {Q_DIR}")


if __name__ == "__main__":
    main()
