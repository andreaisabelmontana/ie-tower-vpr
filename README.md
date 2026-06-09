# IE Tower Visual Place Recognition — Interactive Showcase

A from-scratch static showcase site for a BCSAI **AI: Computer Vision** final project:
a fast **image-retrieval / visual place recognition** system. Given a query photo, it returns
the top-K most similar gallery images and predicts the location around the IE Tower.

🔗 **Live site:** https://andreaisabelmontana.github.io/ie-tower-vpr/

## What the original project does
Three retrieval tracks share one data loader, FAISS index, and evaluation harness:
- **Classical** — SIFT/ORB local features aggregated with VLAD.
- **Deep** — DINOv2 ViT-S/14 global embeddings (ResNet50 fallback).
- **CNN baseline** — a small supervised CNN trained on gallery labels, reused as an embedding extractor.

All three produce L2-normalized vectors → the same FAISS index → top-K nearest → predicted location.
Reproducible: `prepare_data` → `build_index` (per method) → `run_eval` (held-out test set) → a `streamlit` demo UI.
Evaluated with regression, classification, and ranking metrics.

**Stack:** Python 3.11 · FAISS · DINOv2 · OpenCV (SIFT/ORB/VLAD) · PyTorch (CNN/ResNet50) · Streamlit · pytest.

## About this repo
An **original, hand-built static site** (single `index.html`, no framework, no fork) presenting the
project, with a scripted interactive retrieval demo (pick a query → ranked matches → predicted location,
switchable across the three tracks). It is not a reimplementation of the system.

Original source code: [saeka8/computer_vision_final](https://github.com/saeka8/computer_vision_final).
