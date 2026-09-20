"""Keep editable layers, alpha files and assembled paths in sync."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from processors.output_writer import write_svg, write_tiff
from processors.vectorizer import assemble_svg

if TYPE_CHECKING:
    from core.contracts import CapabilitySet
    from core.pipeline_results import LayerResult, PipelineResult


def all_objects_alpha(
    alphas: list[NDArray[np.uint8]], shape: tuple[int, int]
) -> NDArray[np.uint8]:
    """Return the alpha union of every accepted layer at *shape*.

    The all-objects asset is deliberately a union, not a stack of body
    mattes: it must contain every visible object once while preserving soft
    edges and excluding all pixels outside the accepted masks.  An empty
    layer set is represented faithfully by a transparent matte, which lets
    callers create a predictable sidecar without inventing foreground.
    """
    if not alphas:
        return np.zeros(shape, dtype=np.uint8)
    if any(alpha is None or alpha.shape != shape for alpha in alphas):
        raise ValueError("All-objects alpha layers must match the source dimensions")
    return np.maximum.reduce(alphas).astype(np.uint8, copy=False)


def body_alpha(
    parent: NDArray[np.uint8], children: list[NDArray[np.uint8]]
) -> NDArray[np.uint8]:
    """Alpha for a body that recomposes correctly beneath its visible parts."""
    child = np.maximum.reduce(children).astype(np.float32) / 255
    parent_f = parent.astype(np.float32) / 255
    body = np.divide(parent_f - child, 1 - child, out=np.zeros_like(parent_f), where=child < 1)
    return np.rint(body.clip(0, 1) * 255).astype(np.uint8)


def save_layer_outputs(
    result: PipelineResult,
    image: NDArray[np.uint8] | None = None,
    icc_profile: bytes | None = None,
) -> None:
    """Write exactly this result's layers; never glob a shared output folder."""
    if result.svg_path:
        write_svg(assemble_svg(result.width, result.height, [
            {"id": layer.layer_id, "label": layer.label, "parent_id": layer.parent_id or "",
             "svg_data": layer.svg_data, "dx": layer.dx, "dy": layer.dy}
            for layer in result.layers
        ]), Path(result.svg_path))
    if image is None:
        return
    files = []
    for layer in result.layers:
        if layer.alpha is not None and layer.alpha_path:
            path = Path(layer.alpha_path)
            write_tiff(image, path, layer.alpha, icc_profile=icc_profile)
            files.append(str(path))
            children = [c.alpha for c in result.layers if c.parent_id == layer.layer_id and c.alpha is not None]
            if children:
                body_path = path.with_name(f"{path.stem}-body.tiff")
                write_tiff(image, body_path, body_alpha(layer.alpha, children), icc_profile=icc_profile)
                files.append(str(body_path))
    all_objects_path = getattr(result, "all_objects_tiff_path", None)
    if all_objects_path:
        # Vector-only runs have no per-layer soft alpha, but still have the
        # authoritative binary masks.  Use them on a later layer edit rather
        # than accidentally replacing the sidecar with an empty image.
        alphas = [layer.alpha for layer in result.layers if layer.alpha is not None]
        if not alphas:
            alphas = [layer.mask for layer in result.layers if layer.mask is not None]
        alpha = all_objects_alpha(
            alphas,
            image.shape[:2],
        )
        write_tiff(image, Path(all_objects_path), alpha, icc_profile=icc_profile)
        files.append(str(all_objects_path))
    result.tiff_files = files


def _segmented_mask(layer: LayerResult) -> NDArray[np.uint8]:
    """A layer's mask, or a ValueError naming the layer that lacks one.

    LayerResult.mask is Optional because a layer can exist without ever
    having been segmented. Reclipping against such a layer used to hand None
    to numpy, which answers `'<=' not supported between instances of 'int'
    and 'NoneType'` -- naming neither the layer nor the field.
    """
    if layer.mask is None:
        raise ValueError(
            f"Layer '{layer.layer_id}' has no mask: it was never segmented, "
            "so a replacement cannot be reclipped against it."
        )
    return layer.mask


def _refined_alpha(layer: LayerResult) -> NDArray[np.uint8]:
    """A layer's alpha matte, or a ValueError naming the layer that lacks one."""
    if layer.alpha is None:
        raise ValueError(
            f"Layer '{layer.layer_id}' has no alpha matte: it was never "
            "refined, so a child matte cannot be limited to it."
        )
    return layer.alpha


def replace_layer_mask(
    result: PipelineResult,
    layer_id: str,
    mask: NDArray[np.uint8],
    image: NDArray[np.uint8],
    capabilities: CapabilitySet,
    alpha_limit: NDArray[np.uint8] | None = None,
    icc_profile: bytes | None = None,
) -> None:
    """Apply a replacement and reclip any descendants to the new silhouette."""
    by_id = {layer.layer_id: layer for layer in result.layers}
    layer = by_id[layer_id]
    if mask.shape != (result.height, result.width):
        raise ValueError("Replacement mask must match the source image")
    layer.mask = (mask > 127).astype(np.uint8) * 255
    if alpha_limit is not None:
        layer.mask[alpha_limit == 0] = 0
    if layer.parent_id:
        layer.mask = np.minimum(layer.mask, _segmented_mask(by_id[layer.parent_id]))
    affected = [layer] + [child for child in result.layers if child.parent_id == layer_id]
    for target in affected:
        if target.parent_id:
            target.mask = np.minimum(
                _segmented_mask(target), _segmented_mask(by_id[target.parent_id])
            )
        target_mask = _segmented_mask(target)
        target.svg_data = capabilities.vectorizer.trace(target_mask)
        if target.alpha_path:
            target.alpha = capabilities.alpha_refiner.predict(image, target_mask)
            if alpha_limit is not None:
                target.alpha = np.minimum(target.alpha, alpha_limit)
            if target.parent_id:
                target.alpha = np.minimum(
                    target.alpha, _refined_alpha(by_id[target.parent_id])
                )
    save_layer_outputs(result, image, icc_profile)
