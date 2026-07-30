"""Load flat color-centroid maps for the maintained read pipeline."""

import json
from pathlib import Path

import numpy as np


def load_centroids(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {color: np.asarray(value, float) for color, value in payload.items()}
