"""Which MLX SAM 3 grounding queries become detections.

SAM 3 scores a text prompt with two separate heads. Each of the ~200 object
queries gets its own probability -- "this region matches the phrase" -- and the
decoder's presence token produces a single scalar for the whole image --
"the phrase describes something in this picture at all". The vendored
processor multiplies them and thresholds the product, which lets the one
scalar veto every query at once.

That veto is wrong for this pipeline. A label arriving here has already been
confirmed by the operator or by whole-image interrogation, so the question is
where the object is, not whether it is there.

Measured over all 231 images of the APPLE50 corpus with the label ``computer``,
comparing the 89 that produced no layer against the 142 that did: the best
query probability spans 0.478-0.949 in the first group and 0.631-0.989 in the
second -- overlapping ranges that do not separate them -- while presence spans
0.0015-0.1034 against 0.022-0.970. All of the loss is in the presence scalar.
Rendering the best query's mask on the failing images shows an accurate
segmentation of the subject in each. The veto is global rather than
per-detection: pasting g4cube.png beside 128k.png on one canvas lifts presence
from 0.0026 to 0.9363 and the query over the unchanged cube pixels then scores
0.472 instead of 0.0021.

So presence stays the primary rule, and a bounded rescue runs only when it has
rejected the entire image: the single best-localised query is kept, provided it
clears ``localization_confidence``. Speculative part labels set
``require_presence`` and keep the strict rule, because for a part the
recognition question is genuinely open -- on this corpus a presence-free rule
would accept "printer" on g4cube.png at 0.901 and "mouse" on imac_dal.png at
0.946.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class SelectionPolicy:
    """Thresholds for one detection call.

    Attributes
    ----------
    confidence:
        Minimum presence-weighted score, the ``sam3_confidence`` preference.
    localization_confidence:
        Minimum query probability for the rescue described in the module
        docstring. It is a different quantity from ``confidence`` and is not
        comparable with it. The shipped 0.65 sits below the weakest genuine
        subject measured on the corpus above -- 0.702, for imac_3_colors.png
        -- with margin, rather than being fitted to it.
    require_presence:
        Skip the rescue entirely, leaving the vendored rule untouched.
    """

    confidence: float
    localization_confidence: float
    require_presence: bool


@dataclass(frozen=True)
class Selection:
    """The queries one detection call keeps.

    Attributes
    ----------
    indices:
        Query indices, best first.
    scores:
        One reported confidence per kept index. Under the ordinary rule this
        is the presence-weighted score, the quantity ``MIN_SAM3_PART_SCORE``
        in ``core.orchestrator`` is written against. A rescued detection
        reports its query probability instead, because its presence-weighted
        score is a number the presence head produced by refusing to recognise
        the object at all. The two never appear in the same call.
    rescued:
        True when the presence gate rejected the image and the localisation
        rescue supplied the result.
    """

    indices: list[int]
    scores: list[float]
    rescued: bool


def select_queries(
    query_probs: NDArray[np.float32], presence: float, policy: SelectionPolicy
) -> Selection:
    """Return the queries to keep, best first.

    Parameters
    ----------
    query_probs:
        Per-query match probabilities, already through a sigmoid.
    presence:
        The image-level presence probability for this phrase.
    """
    if query_probs.ndim != 1:
        raise ValueError(f"query_probs must be one-dimensional, got shape {query_probs.shape}")
    if not np.isfinite(presence) or not 0.0 <= presence <= 1.0:
        raise ValueError(f"presence must be a probability, got {presence!r}")
    empty = Selection(indices=[], scores=[], rescued=False)
    if query_probs.size == 0:
        return empty

    scores = query_probs * presence
    order = np.argsort(scores)[::-1]
    kept = [int(index) for index in order if scores[index] > policy.confidence]
    if kept:
        return Selection(kept, [float(scores[index]) for index in kept], rescued=False)
    if policy.require_presence:
        return empty

    best = int(np.argmax(query_probs))
    if float(query_probs[best]) <= policy.localization_confidence:
        return empty
    return Selection([best], [float(query_probs[best])], rescued=True)
