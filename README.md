# IE Tower VPR — classical visual place recognition by image retrieval

Given a query photo, predict **which location** it was taken at by retrieving the
most similar gallery images and voting on their labels. Framed around recognising
locations/floors of a building ("IE Tower"), evaluated on a small **controlled,
synthetic benchmark** (not real tower photos).

🔗 **Showcase site:** https://andreaisabelmontana.github.io/ie-tower-vpr/

## Scope (read this first)

This repo implements the **classical** retrieval track end to end and runs it for
real. The original BCSAI Computer Vision group project additionally fused **DINOv2
ViT** and a **supervised CNN** embedding into a single **FAISS** index. Those deep
tracks need a GPU / PyTorch and are **out of scope here** — this machine has no
torch/faiss. So instead of faking those numbers, this repo:

- builds the classical global descriptors for real,
- puts them in one retrieval index with **exact cosine NN** search (numpy/sklearn)
  in place of FAISS (same top-K semantics, just no approximate-search speedup),
- and reports only metrics from real runs on the committed benchmark.

## Pipeline

```
query image
  → describe   ORB bag-of-visual-words  +  HSV colour hist  +  gradient hist   (L2-normalized)
  → index      one gallery matrix, exact cosine nearest-neighbour (dot product)
  → retrieve   top-K most similar gallery views
  → predict    majority vote over top-K labels  → location
```

### Descriptors (`tower/descriptors.py`)
- **ORB Bag-of-Visual-Words** — ORB keypoints/descriptors, quantised against a
  **KMeans** visual-word vocabulary learned from the gallery, then a term-frequency
  word histogram.
- **Colour histogram** — coarse HSV histogram (captures a location's palette).
- **Gradient histogram** — global magnitude-weighted orientation histogram
  (captures structure).
- All three are weighted, concatenated, and **L2-normalized** into one vector
  (here, 592-dim) so cosine similarity reduces to a dot product.

### Index (`tower/index.py`)
A single `RetrievalIndex` holding the gallery descriptor matrix + labels. Search
is an exact cosine NN: one matrix–vector product, then top-K. Honest note: in the
original this was a FAISS index also holding DINOv2/CNN embeddings; here it holds
the classical descriptors with exact NN.

### Evaluation (`tower/evaluate.py`)
Top-1 accuracy and **Recall@K** ("is a same-location gallery image in the top-K?"),
each compared against an analytic **random-retrieval baseline** computed from the
per-location gallery counts.

## Dataset (controlled benchmark — synthetic, not real photos)

`data/scripts/make_base_images.py` procedurally synthesizes 6 distinct, textured
"location" base images (per-location palette + structural pattern + noise).
`data/scripts/make_dataset.py` augments each base into **database** and **query**
views with disjoint seeds, applying lighting (brightness/gamma/contrast),
viewpoint (rotation/scale/perspective/crop), Gaussian blur, and JPEG-compression
perturbations. The generator and base images are committed for full reproducibility.

- 6 locations · 36 database (gallery) views · 24 held-out query views.

## Results (real run of `python demo.py`)

```
locations: atrium, auditorium, cafeteria, library, plaza, rooftop
index size: 36 vectors, dim 592

Relocalising query: true location = 'atrium'
  rank 0  sim=0.834  atrium   <-- correct
  rank 1  sim=0.803  atrium   <-- correct
  rank 2  sim=0.748  atrium   <-- correct
  ...
  predicted location (top-5 vote): 'atrium'  -> CORRECT

queries evaluated : 24
top-1 accuracy    : 1.000
Recall@1          : 1.000   (random 0.167)
Recall@3          : 1.000   (random 0.431)
Recall@5          : 1.000   (random 0.622)
```

On this clean controlled split the classical descriptors separate the six
locations perfectly: **Recall@1 = 1.000 vs 0.167 random (6.0×)**. Under heavier
synthetic distortion (≈±30° rotation, 9×9 blur, strong lighting shifts) top-1
degrades to ~0.75 while still far above the 0.167 random baseline — i.e. the
retriever is doing real work, not trivially separating by colour.

## Run it

```bash
pip install -r requirements.txt          # numpy, opencv-python, scikit-learn, pytest

python data/scripts/make_base_images.py  # synthesize base location images
python data/scripts/make_dataset.py      # build database + query views
python demo.py                           # relocalise + full Recall@K vs random
python -m pytest -q                      # tests (regenerate dataset if missing)
```

`demo.py` flags: `--location <name>`, `--view <i>`, `--k <K>`, `--n-words <N>`.

## Tests (`tests/`)

`python -m pytest -q` → **12 passed**. Coverage:
- the KMeans vocabulary builds and encodes images to a finite, L2-normalized vector;
- ORB/colour/gradient descriptors have the expected shapes and are distributions;
- descriptor similarity is **higher within a location than across** (clear margin);
- a same-location query retrieves the correct location **top-1** for clear cases;
- held-out **Recall@K beats the random baseline** by a clear margin at every K.

## Layout

```
tower/descriptors.py   ORB BoVW + colour/gradient global descriptors
tower/index.py         exact cosine-NN retrieval index
tower/evaluate.py      top-1 + Recall@K vs random baseline
tower/data.py          dataset loaders
data/scripts/          base-image synthesizer + dataset generator
data/base|database|queries/   committed controlled benchmark
demo.py                end-to-end relocalisation demo
tests/                 pytest suite
index.html             self-contained showcase page (GitHub Pages)
```

**Stack:** Python · OpenCV (ORB) · scikit-learn (KMeans, exact NN) · NumPy · pytest.
MIT licensed.
