"""test_vectorizer.py -- VTracer mask tracing and multi-layer SVG assembly.

Real VTracer calls are used (it is installed) but kept to tiny (<=32x32)
masks so the suite stays fast. No network access, no ML model loading.
"""
from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from processors.vectorizer import (
    LAYER_PALETTE,
    VTracerVectorizer,
    _extract_svg_content,
    _strip_vtracer_fills,
    assemble_svg,
    trace_mask,
)


def _square_mask(size: int = 32, box: tuple[int, int, int, int] = (8, 8, 24, 24)) -> np.ndarray:
    mask = np.zeros((size, size), dtype=np.uint8)
    y0, x0, y1, x1 = box
    mask[y0:y1, x0:x1] = 255
    return mask


# ── trace_mask ───────────────────────────────────────────────────────────────


class TestTraceMask:
    def test_traces_filled_square_to_svg_path_data(self) -> None:
        mask = _square_mask()
        svg = trace_mask(mask)
        assert "<svg" in svg
        assert "<path" in svg

    def test_empty_mask_still_produces_svg_document(self) -> None:
        mask = np.zeros((16, 16), dtype=np.uint8)
        svg = trace_mask(mask)
        assert "<svg" in svg

    def test_single_pixel_mask_does_not_crash(self) -> None:
        mask = np.zeros((16, 16), dtype=np.uint8)
        mask[8, 8] = 255
        svg = trace_mask(mask)
        assert "<svg" in svg

    def test_mask_touching_border_does_not_crash(self) -> None:
        mask = np.zeros((16, 16), dtype=np.uint8)
        mask[0:16, 0:4] = 255
        svg = trace_mask(mask)
        assert "<svg" in svg

    def test_custom_thresholds_accepted(self) -> None:
        mask = _square_mask()
        svg = trace_mask(
            mask,
            mode="polygon",
            corner_threshold=30,
            length_threshold=2.0,
            splice_threshold=20,
            filter_speckle=1,
        )
        assert "<svg" in svg


# ── assemble_svg ─────────────────────────────────────────────────────────────


class TestAssembleSvg:
    def test_wraps_in_svg_with_correct_viewbox_and_size(self) -> None:
        svg = assemble_svg(100, 50, [])
        assert 'viewBox="0 0 100 50"' in svg
        assert 'width="100" height="50"' in svg
        assert "<svg" in svg and "</svg>" in svg

    def test_no_layers_produces_empty_body(self) -> None:
        svg = assemble_svg(10, 10, [])
        assert "<g" not in svg

    def test_single_layer_uses_first_palette_colour_and_default_id(self) -> None:
        layers = [{"svg_data": '<path d="M0 0 L1 1"/>'}]
        svg = assemble_svg(10, 10, layers)
        assert f'fill="{LAYER_PALETTE[0]}"' in svg
        assert 'id="layer_0"' in svg
        assert '<path d="M0 0 L1 1"/>' in svg

    def test_explicit_id_is_used(self) -> None:
        layers = [{"id": "body", "svg_data": '<path d="M0 0"/>'}]
        svg = assemble_svg(10, 10, layers)
        assert 'id="body"' in svg

    def test_translate_applied_when_dx_or_dy_nonzero(self) -> None:
        layers = [{"id": "child", "svg_data": '<path d="M0 0"/>', "dx": 5, "dy": -3}]
        svg = assemble_svg(20, 20, layers)
        assert "translate(5,-3)" in svg

    def test_no_translate_when_dx_and_dy_zero(self) -> None:
        layers = [{"id": "child", "svg_data": '<path d="M0 0"/>', "dx": 0, "dy": 0}]
        svg = assemble_svg(20, 20, layers)
        assert "translate" not in svg

    def test_layer_colours_cycle_through_palette(self) -> None:
        layers = [{"svg_data": "<path/>"} for _ in range(len(LAYER_PALETTE) + 1)]
        svg = assemble_svg(10, 10, layers)
        assert f'fill="{LAYER_PALETTE[0]}"' in svg
        # The (len(LAYER_PALETTE)+1)-th layer wraps back to the first colour.
        assert svg.count(f'fill="{LAYER_PALETTE[0]}"') == 2

    def test_multiple_layers_preserve_order(self) -> None:
        layers = [
            {"id": "parent", "svg_data": "<path d='P'/>"},
            {"id": "child", "svg_data": "<path d='C'/>"},
        ]
        svg = assemble_svg(10, 10, layers)
        assert svg.index('id="parent"') < svg.index('id="child"')

    def test_unsafe_layer_id_cannot_inject_markup(self) -> None:
        """A label containing a quote must not break out of the id attribute."""
        unsafe_id = 'a"><script>alert(1)</script>'
        layers = [{"id": unsafe_id, "svg_data": "<path/>"}]
        svg = assemble_svg(10, 10, layers)
        assert "<script>" not in svg
        assert "&quot;" in svg

    def test_escaped_layer_id_keeps_document_well_formed(self) -> None:
        layers = [{"id": 'q"&<>', "svg_data": "<path/>"}]
        svg = assemble_svg(10, 10, layers)
        # The whole document must still parse as XML.
        ET.fromstring(svg)

    def test_ampersand_in_layer_id_is_escaped(self) -> None:
        layers = [{"id": "cup & saucer", "svg_data": "<path/>"}]
        svg = assemble_svg(10, 10, layers)
        assert "&amp;" in svg
        ET.fromstring(svg)


# ── VTracerVectorizer ────────────────────────────────────────────────────────


class TestVTracerVectorizer:
    def test_trace_delegates_to_trace_mask_with_configured_params(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, object] = {}

        def fake_trace_mask(mask, mode="spline", corner_threshold=60,
                             length_threshold=4.0, splice_threshold=45,
                             filter_speckle=8):
            captured.update(
                corner_threshold=corner_threshold,
                length_threshold=length_threshold,
                splice_threshold=splice_threshold,
                filter_speckle=filter_speckle,
            )
            return "<svg></svg>"

        import processors.vectorizer as vectorizer_module

        monkeypatch.setattr(vectorizer_module, "trace_mask", fake_trace_mask)

        vectorizer = VTracerVectorizer(
            corner_threshold=10,
            length_threshold=1.5,
            splice_threshold=20,
            filter_speckle=2,
        )
        mask = _square_mask()
        result = vectorizer.trace(mask)

        assert result == "<svg></svg>"
        assert captured == {
            "corner_threshold": 10,
            "length_threshold": 1.5,
            "splice_threshold": 20,
            "filter_speckle": 2,
        }

    def test_trace_with_real_vtracer_backend(self) -> None:
        vectorizer = VTracerVectorizer()
        svg = vectorizer.trace(_square_mask())
        assert "<svg" in svg


# ── SVG string helpers ───────────────────────────────────────────────────────


class TestExtractSvgContent:
    def test_strips_xml_declaration_and_svg_wrapper(self) -> None:
        full = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10">'
            '<path d="M0 0"/>'
            "</svg>"
        )
        inner = _extract_svg_content(full)
        assert inner == '<path d="M0 0"/>'

    def test_handles_content_without_xml_declaration(self) -> None:
        full = '<svg width="10" height="10"><path d="M1 1"/></svg>'
        inner = _extract_svg_content(full)
        assert inner == '<path d="M1 1"/>'

    def test_handles_plain_content_without_svg_wrapper(self) -> None:
        content = '<path d="M2 2"/>'
        assert _extract_svg_content(content) == content


class TestStripVtracerFills:
    def test_removes_white_background_paths(self) -> None:
        svg = (
            '<path d="M0 0" fill="#ffffff"/>'
            '<path d="M1 1" fill="#000000"/>'
        )
        result = _strip_vtracer_fills(svg)
        assert "#ffffff" not in result
        assert 'd="M1 1"' in result

    def test_strips_fill_attribute_from_object_paths(self) -> None:
        svg = '<path d="M1 1" fill="#000000"/>'
        result = _strip_vtracer_fills(svg)
        assert "fill=" not in result
        assert 'd="M1 1"' in result

    def test_case_insensitive_white_removal(self) -> None:
        svg = '<path d="M0 0" fill="#FFFFFF"/><path d="M1 1" fill="#123456"/>'
        result = _strip_vtracer_fills(svg)
        assert "#FFFFFF" not in result
        assert 'd="M1 1"' in result
