"""Selection rules for MLX SAM 3 grounding scores."""
from __future__ import annotations

import numpy as np

from models.sam3_scoring import SelectionPolicy, select_queries


def _policy(**overrides: float | bool) -> SelectionPolicy:
    base: dict[str, float | bool] = {
        "confidence": 0.2,
        "localization_confidence": 0.65,
        "require_presence": False,
    }
    base.update(overrides)
    return SelectionPolicy(**base)  # type: ignore[arg-type]


def test_presence_weighted_queries_are_kept_in_score_order() -> None:
    probs = np.array([0.30, 0.95, 0.10, 0.80], dtype=np.float32)
    assert select_queries(probs, presence=0.9, policy=_policy()).indices == [1, 3, 0]


def test_a_confident_query_survives_a_collapsed_presence_score() -> None:
    """The defect: one image-level scalar discarded every confident detection.

    g4cube.png scores 0.816 on its best query and 0.0026 on presence, so the
    product is 0.0021 and the whole image returns nothing.
    """
    probs = np.array([0.816, 0.173, 0.172], dtype=np.float32)
    assert select_queries(probs, presence=0.0026, policy=_policy()).indices == [0]


def test_the_rescue_returns_only_the_dominant_query() -> None:
    probs = np.array([0.816, 0.790, 0.173], dtype=np.float32)
    assert select_queries(probs, presence=0.0026, policy=_policy()).indices == [0]


def test_the_rescue_needs_a_confident_localization() -> None:
    probs = np.array([0.55, 0.42], dtype=np.float32)
    assert select_queries(probs, presence=0.0026, policy=_policy()).indices == []


def test_a_part_label_keeps_the_presence_gate() -> None:
    """Parts are speculative, so recognition still has to agree."""
    probs = np.array([0.901, 0.300], dtype=np.float32)
    policy = _policy(require_presence=True)
    assert select_queries(probs, presence=0.0080, policy=policy).indices == []


def test_the_rescue_never_displaces_a_healthy_presence_result() -> None:
    probs = np.array([0.40, 0.95], dtype=np.float32)
    assert select_queries(probs, presence=0.95, policy=_policy()).indices == [1, 0]


def test_a_non_probability_query_score_is_rejected_rather_than_rescued() -> None:
    """`presence` is validated here; `query_probs` was not.

    A NaN passes neither threshold test, so the ordinary rule keeps nothing
    and the rescue then reads it as the best query and returns it -- a
    detection whose reported confidence is NaN, which every later comparison
    silently answers False.
    """
    for bad in (
        np.array([0.1, np.nan, 0.3], dtype=np.float32),
        np.array([0.1, np.inf], dtype=np.float32),
        np.array([1.5, 0.3], dtype=np.float32),
        np.array([-0.2, 0.3], dtype=np.float32),
    ):
        try:
            select_queries(bad, presence=0.5, policy=_policy())
        except ValueError:
            continue
        raise AssertionError(f"accepted non-probability query scores: {bad}")
