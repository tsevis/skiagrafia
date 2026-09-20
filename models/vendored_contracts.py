"""vendored_contracts.py  --  what this project uses from vendored models.

GroundingDINO, SAM 2.1 and MLX SAM 3 are not installed packages: they are
source trees restored into the model directory and imported at call time, so
a type checker cannot see them and every object they return arrives as
`object`. Typing those attributes as `object` then made real attribute
access an error, and hid whether a value could be None.

These Protocols describe only the surface this project actually touches —
not the vendored classes in full. That is deliberate: the narrow version is
the contract worth stating, and it fails if an upstream refactor removes a
method we depend on, rather than describing methods nobody here calls.

Structural, so nothing at runtime inherits from them and no import of the
vendored code is introduced.
"""
from __future__ import annotations

from typing import Any, Protocol


class SamPredictorLike(Protocol):
    """The SAM 2.1 image predictor surface used by models/grounded_sam.py."""

    def set_image(self, image: Any) -> None: ...

    def predict(self, **kwargs: Any) -> tuple[Any, Any, Any]: ...

    def reset_predictor(self) -> None: ...


class Sam3ImageModelLike(Protocol):
    """The MLX SAM 3 model surface used by models/mlx_sam3.py."""

    def set_image(self, image: Any) -> Any: ...

    def reset_all_prompts(self, *args: Any, **kwargs: Any) -> Any: ...

    def set_text_prompt(self, *args: Any, **kwargs: Any) -> Any: ...


class VitMatteModelLike(Protocol):
    """The VitMatte surface used by models/vitmatte_refiner.py: it is called."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...
