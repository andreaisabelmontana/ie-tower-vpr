"""Shared fixtures: ensure the controlled dataset exists, then load it once."""
from __future__ import annotations

import importlib.util
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _run_script(rel_path: str) -> None:
    path = os.path.join(ROOT, rel_path)
    spec = importlib.util.spec_from_file_location("_gen", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main()


@pytest.fixture(scope="session", autouse=True)
def ensure_dataset():
    """Generate base images + dataset if they are not already present."""
    from tower.data import DB_DIR, Q_DIR

    if not (os.path.isdir(DB_DIR) and os.path.isdir(Q_DIR)
            and os.listdir(DB_DIR) and os.listdir(Q_DIR)):
        base = os.path.join(ROOT, "data", "base")
        if not (os.path.isdir(base) and os.listdir(base)):
            _run_script(os.path.join("data", "scripts", "make_base_images.py"))
        _run_script(os.path.join("data", "scripts", "make_dataset.py"))


@pytest.fixture(scope="session")
def dataset():
    from tower.data import load_database, load_queries

    db_imgs, db_labels, db_paths = load_database()
    q_imgs, q_labels, q_paths = load_queries()
    return {
        "db_imgs": db_imgs, "db_labels": db_labels, "db_paths": db_paths,
        "q_imgs": q_imgs, "q_labels": q_labels, "q_paths": q_paths,
    }


@pytest.fixture(scope="session")
def index(dataset):
    from tower.index import RetrievalIndex

    return RetrievalIndex.build(
        dataset["db_imgs"], dataset["db_labels"],
        paths=dataset["db_paths"], n_words=64,
    )
