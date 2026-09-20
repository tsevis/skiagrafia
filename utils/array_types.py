"""array_types.py  --  narrowing helpers for OpenCV return values.

cv2's type stubs return `MatLike`, which is an ndarray of *any* dtype. Most
functions in this project declare `NDArray[np.uint8]`, because that is what
they are given and what they hand back — but nothing connected the two, so a
type checker reported every `return cv2.<op>(...)` as a mismatch.

These helpers close the gap by making the declared type true rather than by
asserting it. `np.asarray` returns the SAME object when the input already has
the requested dtype, so on the intended path this costs nothing and copies
nothing; on an unintended path it converts instead of silently handing back
an array whose dtype contradicts the signature.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def as_uint8(array: Any) -> NDArray[np.uint8]:
    """Return `array` as a uint8 ndarray, without copying when it already is.

    Intended for the immediate return value of an OpenCV call whose inputs
    were uint8. It is a narrowing, not a conversion: passing float data here
    truncates towards zero, which is why it belongs directly around a cv2
    call and not at an arbitrary boundary.
    """
    return np.asarray(array, dtype=np.uint8)


def as_intp(array: Any) -> NDArray[np.intp]:
    """Return `array` as an index-typed ndarray, for use as a fancy index.

    `cv2.connectedComponentsWithStats` hands back a label image as MatLike,
    which numpy will not accept as an index expression even though the values
    are integer labels.
    """
    return np.asarray(array, dtype=np.intp)
