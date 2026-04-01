"""contracts.py  --  v5.0 Capability Protocols

Defines the six capability interfaces that the Orchestrator depends on.
Uses typing.Protocol (structural subtyping) so concrete clients need only
match the method signatures -- no inheritance required.

All protocols are @runtime_checkable for defensive isinstance() assertions.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from core.knowledge import KnowledgePack


@runtime_checkable
class Interrogator(Protocol):
    """Semantic image interrogation -> structured candidates."""

    def interrogate(
        self,
        image: NDArray[np.uint8],
        confirmed_labels: list[str] | None = None,
        knowledge_pack: KnowledgePack | None = None,
    ) -> "InterrogationResult":
        """Return an InterrogationResult with .candidates and .children_by_parent.

        The return type must have at minimum:
          - candidates: list[InterrogationCandidate]
          - children_by_parent: dict[str, list[str]]
        Import the actual InterrogationResult from core.interrogation for the
        concrete implementation. The Protocol uses a forward ref to avoid
        circular imports.
        """
        ...


@runtime_checkable
class Detector(Protocol):
    """Text-guided bounding box detection."""

    def detect_box(
        self,
        image: NDArray[np.uint8],
        label: str,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
    ) -> object | None:
        """Return a DetectionResult or None."""
        ...


@runtime_checkable
class Segmenter(Protocol):
    """Bounding-box-guided mask segmentation."""

    def segment(
        self,
        image: NDArray[np.uint8],
        bbox: tuple[int, int, int, int],
        label: str = "",
    ) -> NDArray[np.uint8]:
        """Return binary mask (H, W) uint8, values 0 or 255."""
        ...

    def clear_cache(self) -> None:
        """Release any per-image caches."""
        ...


@runtime_checkable
class AlphaRefiner(Protocol):
    """Coarse mask -> alpha matte refinement."""

    def predict(
        self,
        image: NDArray[np.uint8],
        mask: NDArray[np.uint8],
    ) -> NDArray[np.uint8]:
        """Return alpha matte (H, W) uint8, 0-255."""
        ...


@runtime_checkable
class Vectorizer(Protocol):
    """Binary mask -> SVG path data."""

    def trace(
        self,
        mask: NDArray[np.uint8],
    ) -> str:
        """Return raw SVG string for this mask layer."""
        ...


class CapabilitySet(BaseModel):
    """All capabilities the Orchestrator needs, bundled for injection."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    interrogator: Interrogator
    detector: Detector
    segmenter: Segmenter
    alpha_refiner: AlphaRefiner
    vectorizer: Vectorizer
