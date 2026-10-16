"""test_edge_refiner.py -- BiRefNet may move an edge, and nothing else."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.birefnet_refiner import merge_in_band


def _square() -> np.ndarray:
    mask = np.zeros((200, 200), dtype=np.uint8)
    mask[50:150, 50:150] = 1
    return mask


class TestMergeInBand:
    def test_the_edge_follows_the_refiner_inside_the_band(self) -> None:
        refined = np.zeros((200, 200), dtype=np.float32)
        refined[50:150, 50:152] = 1.0  # the true edge is two pixels further right

        assert merge_in_band(_square(), refined)[100, 151] == 1

    def test_a_neighbour_the_refiner_also_found_stays_out(self) -> None:
        # The crop's salient region swallowed everything around the object.
        everything = np.ones((200, 200), dtype=np.float32)

        merged = merge_in_band(_square(), everything)

        assert merged[100, 10] == 0

    def test_the_interior_is_never_opened(self) -> None:
        nothing = np.zeros((200, 200), dtype=np.float32)

        assert merge_in_band(_square(), nothing)[100, 100] == 1
