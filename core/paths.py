"""Repo-root path anchors shared by all packages (files moved one level down;
these constants keep checkpoints/results/figures at the repo root)."""
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINTS = os.path.join(REPO_ROOT, "checkpoints")
RESULTS = os.path.join(REPO_ROOT, "results")
FIGURES = os.path.join(REPO_ROOT, "figures")
