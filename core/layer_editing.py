"""Keep editable layers, alpha files and assembled paths in sync."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from processors.output_writer import write_svg, write_tiff
from processors.vectorizer import assemble_svg


def all_objects_alpha(alphas, shape):
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


def body_alpha(parent, children):
    """Alpha for a body that recomposes correctly beneath its visible parts."""
    child = np.maximum.reduce(children).astype(np.float32) / 255
    parent = parent.astype(np.float32) / 255
    body = np.divide(parent - child, 1 - child, out=np.zeros_like(parent), where=child < 1)
    return np.rint(body.clip(0, 1) * 255).astype(np.uint8)


def save_layer_outputs(result, image=None, icc_profile=None):
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


def replace_layer_mask(result, layer_id, mask, image, capabilities, alpha_limit=None, icc_profile=None):
    """Apply a replacement and reclip any descendants to the new silhouette."""
    by_id = {layer.layer_id: layer for layer in result.layers}
    layer = by_id[layer_id]
    if mask.shape != (result.height, result.width):
        raise ValueError("Replacement mask must match the source image")
    layer.mask = (mask > 127).astype(np.uint8) * 255
    if alpha_limit is not None:
        layer.mask[alpha_limit == 0] = 0
    if layer.parent_id:
        layer.mask = np.minimum(layer.mask, by_id[layer.parent_id].mask)
    affected = [layer] + [child for child in result.layers if child.parent_id == layer_id]
    for target in affected:
        if target.parent_id:
            target.mask = np.minimum(target.mask, by_id[target.parent_id].mask)
        target.svg_data = capabilities.vectorizer.trace(target.mask)
        if target.alpha_path:
            target.alpha = capabilities.alpha_refiner.predict(image, target.mask)
            if alpha_limit is not None:
                target.alpha = np.minimum(target.alpha, alpha_limit)
            if target.parent_id:
                target.alpha = np.minimum(target.alpha, by_id[target.parent_id].alpha)
    save_layer_outputs(result, image, icc_profile)
