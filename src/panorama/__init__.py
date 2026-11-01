"""Feature-based image stitching with inspectable algorithms."""

import os

# Predictable CPU use on shared hosts; users may explicitly override these values.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")

__version__ = "0.1.0"
