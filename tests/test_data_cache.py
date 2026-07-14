"""Regression test for the feature-cache decompression bug.

The cache path built feat_map with `loaded["X"][i]` inside a loop; because
np.load returns a lazy NpzFile, each index re-decompressed the ENTIRE array,
making a warm-cache load O(N) full decompressions (98 GB / minutes for an 11k x
2048 endpoint). We materialise the array once. This test asserts the warm load
is fast and byte-identical to the cold load."""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data import featurize_smiles  # noqa: E402


def test_warm_cache_is_fast_and_identical(tmp_path):
    smiles = ["CCO", "c1ccccc1", "CC(=O)O", "CCN", "O=C=O"] * 40
    cold = featurize_smiles(smiles, "morgan", 2, 512, cache_dir=tmp_path)
    t0 = time.time()
    warm = featurize_smiles(smiles, "morgan", 2, 512, cache_dir=tmp_path)
    dt = time.time() - t0
    assert np.array_equal(cold, warm), "warm cache load differs from cold compute"
    assert cold.shape == (200, 512)
    # warm load must be quick: the bug made this scale with N full decompressions.
    assert dt < 2.0, f"warm cache load too slow ({dt:.2f}s) -- decompression regression?"
