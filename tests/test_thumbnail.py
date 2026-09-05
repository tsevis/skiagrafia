"""test_thumbnail.py  --  SVG thumbnail rendering and caching.

All tests are offline and GUI-free: CairoSVG is faked and PIL's
``ImageTk.PhotoImage`` (which requires a live Tk root) is stubbed out with a
plain passthrough so no Tk window is ever constructed.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import thumbnail


@pytest.fixture(autouse=True)
def _stub_photoimage(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace ImageTk.PhotoImage with an identity stub (no Tk root needed)."""
    monkeypatch.setattr(thumbnail.ImageTk, "PhotoImage", lambda img: img)


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    """Every test uses a distinct cache key, but clear defensively either way."""
    thumbnail.render_svg_thumbnail.cache_clear()
    yield
    thumbnail.render_svg_thumbnail.cache_clear()


def _png_bytes(size: int, rgba: tuple[int, int, int, int]) -> bytes:
    img = Image.new("RGBA", (size, size), rgba)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestRenderSvgThumbnailWithCairo:
    def test_successful_render_tints_with_requested_colour(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        source_png = _png_bytes(4, (9, 9, 9, 200))
        fake_cairo = SimpleNamespace(svg2png=lambda **kwargs: source_png)
        monkeypatch.setattr(thumbnail, "_get_cairosvg", lambda: fake_cairo)

        result = thumbnail.render_svg_thumbnail("icon-a.svg", "#112233", size=4)

        assert result.mode == "RGBA"
        assert result.size == (4, 4)
        # Interior pixel: tinted colour with source alpha preserved.
        assert result.getpixel((1, 1)) == (0x11, 0x22, 0x33, 200)
        # Border pixel: outline colour drawn on top, fully opaque.
        assert result.getpixel((0, 0)) == (0x11, 0x22, 0x33, 255)

    def test_svg2png_receives_requested_dimensions_and_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict = {}

        def fake_svg2png(**kwargs):
            captured.update(kwargs)
            return _png_bytes(8, (1, 2, 3, 255))

        monkeypatch.setattr(
            thumbnail, "_get_cairosvg", lambda: SimpleNamespace(svg2png=fake_svg2png)
        )

        thumbnail.render_svg_thumbnail("path/to/icon-b.svg", "#abcdef", size=8)

        assert captured["url"] == "path/to/icon-b.svg"
        assert captured["output_width"] == 8
        assert captured["output_height"] == 8
        assert captured["background_color"] == "transparent"

    def test_render_failure_falls_back_to_solid_placeholder(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def boom(**kwargs):
            raise ValueError("bad svg")

        monkeypatch.setattr(
            thumbnail, "_get_cairosvg", lambda: SimpleNamespace(svg2png=boom)
        )

        result = thumbnail.render_svg_thumbnail("icon-c.svg", "#00ff00", size=6)

        assert result.mode == "RGBA"
        assert result.size == (6, 6)
        assert result.getpixel((2, 2)) == (0x00, 0xFF, 0x00, 180)
        assert result.getpixel((0, 0)) == (0x00, 0xFF, 0x00, 255)


class TestRenderSvgThumbnailWithoutCairo:
    def test_no_cairo_returns_solid_colour_placeholder(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(thumbnail, "_get_cairosvg", lambda: None)

        result = thumbnail.render_svg_thumbnail("icon-d.svg", "#ff00aa", size=5)

        assert result.mode == "RGBA"
        assert result.size == (5, 5)
        assert result.getpixel((2, 2)) == (0xFF, 0x00, 0xAA, 180)
        assert result.getpixel((0, 4)) == (0xFF, 0x00, 0xAA, 255)  # outline pixel

    def test_default_size_is_thirty_two(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(thumbnail, "_get_cairosvg", lambda: None)

        result = thumbnail.render_svg_thumbnail("icon-e.svg", "#010203")

        assert result.size == (32, 32)


class TestThumbnailCache:
    def test_repeated_calls_are_cached_until_invalidated(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = {"n": 0}

        def fake_get_cairosvg():
            calls["n"] += 1
            return None

        monkeypatch.setattr(thumbnail, "_get_cairosvg", fake_get_cairosvg)

        thumbnail.render_svg_thumbnail("icon-f.svg", "#123456", size=4)
        thumbnail.render_svg_thumbnail("icon-f.svg", "#123456", size=4)
        assert calls["n"] == 1  # second call served from lru_cache

        thumbnail.invalidate_thumbnail_cache()
        thumbnail.render_svg_thumbnail("icon-f.svg", "#123456", size=4)
        assert calls["n"] == 2  # cache was cleared, function body re-ran

    def test_different_arguments_are_not_cache_collisions(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(thumbnail, "_get_cairosvg", lambda: None)

        red = thumbnail.render_svg_thumbnail("icon-g.svg", "#ff0000", size=4)
        blue = thumbnail.render_svg_thumbnail("icon-g.svg", "#0000ff", size=4)

        assert red.getpixel((2, 2)) != blue.getpixel((2, 2))


class TestGetCairosvgLazyLoad:
    def test_loads_once_and_caches_result(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sentinel = SimpleNamespace(svg2png=lambda **_: b"")
        calls = {"n": 0}

        def fake_load_cairosvg(logger):
            calls["n"] += 1
            return sentinel

        monkeypatch.setattr(thumbnail, "load_cairosvg", fake_load_cairosvg)
        monkeypatch.setattr(thumbnail, "_cairosvg_checked", False)
        monkeypatch.setattr(thumbnail, "_cairosvg", None)

        first = thumbnail._get_cairosvg()
        second = thumbnail._get_cairosvg()

        assert first is sentinel
        assert second is sentinel
        assert calls["n"] == 1

    def test_missing_cairo_is_cached_as_none_without_retrying(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = {"n": 0}

        def fake_load_cairosvg(logger):
            calls["n"] += 1
            return None

        monkeypatch.setattr(thumbnail, "load_cairosvg", fake_load_cairosvg)
        monkeypatch.setattr(thumbnail, "_cairosvg_checked", False)
        monkeypatch.setattr(thumbnail, "_cairosvg", None)

        assert thumbnail._get_cairosvg() is None
        assert thumbnail._get_cairosvg() is None
        assert calls["n"] == 1
