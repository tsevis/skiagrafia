"""test_output_writer.py -- SVG/TIFF/PNG/PDF file writers.

All files are written under pytest's tmp_path fixture; nothing touches the
user's real filesystem. PDF conversion goes through the real cairosvg
(installed) except where the "cairo not available" branch is exercised via
monkeypatch.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import processors.output_writer as output_writer
from processors.output_writer import write_pdf, write_png, write_svg, write_tiff


# ── write_svg ────────────────────────────────────────────────────────────────


class TestWriteSvg:
    def test_writes_content_and_returns_path(self, tmp_path: Path) -> None:
        target = tmp_path / "out.svg"
        result = write_svg("<svg></svg>", target)

        assert result == target
        assert target.read_text(encoding="utf-8") == "<svg></svg>"

    def test_creates_missing_parent_directories(self, tmp_path: Path) -> None:
        target = tmp_path / "nested" / "dir" / "out.svg"
        write_svg("<svg/>", target)
        assert target.exists()

    def test_empty_svg_content_is_allowed(self, tmp_path: Path) -> None:
        target = tmp_path / "empty.svg"
        write_svg("", target)
        assert target.read_text(encoding="utf-8") == ""

    def test_very_long_unsafe_filename_raises_oserror(self, tmp_path: Path) -> None:
        """Documents current behaviour: write_svg does not sanitise or
        truncate output_path -- an overly long filename (as could result
        from an unsanitised VLM label) raises OSError instead of being
        handled gracefully. See final report."""
        unsafe_name = ("x" * 300) + ".svg"
        target = tmp_path / unsafe_name
        with pytest.raises(OSError):
            write_svg("<svg/>", target)


# ── write_tiff ───────────────────────────────────────────────────────────────


class TestWriteTiff:
    def test_writes_rgb_image_without_alpha(self, tmp_path: Path) -> None:
        image = np.full((8, 8, 3), 100, dtype=np.uint8)
        target = tmp_path / "rgb.tiff"

        result = write_tiff(image, target)

        assert result == target
        with Image.open(target) as reloaded:
            assert reloaded.mode == "RGB"
            assert reloaded.size == (8, 8)

    def test_writes_grayscale_image_without_alpha(self, tmp_path: Path) -> None:
        image = np.full((8, 8), 50, dtype=np.uint8)
        target = tmp_path / "gray.tiff"

        write_tiff(image, target)

        with Image.open(target) as reloaded:
            assert reloaded.mode == "L"

    def test_writes_rgb_image_with_alpha(self, tmp_path: Path) -> None:
        image = np.full((8, 8, 3), 100, dtype=np.uint8)
        alpha = np.full((8, 8), 128, dtype=np.uint8)
        target = tmp_path / "rgba.tiff"

        write_tiff(image, target, alpha=alpha)

        with Image.open(target) as reloaded:
            assert reloaded.mode == "RGBA"
            r, g, b, a = reloaded.split()
            assert np.array(a).max() == 128

    def test_writes_grayscale_image_with_alpha(self, tmp_path: Path) -> None:
        image = np.full((8, 8), 60, dtype=np.uint8)
        alpha = np.full((8, 8), 200, dtype=np.uint8)
        target = tmp_path / "gray_alpha.tiff"

        write_tiff(image, target, alpha=alpha)

        with Image.open(target) as reloaded:
            assert reloaded.mode == "RGBA"

    def test_creates_missing_parent_directories(self, tmp_path: Path) -> None:
        image = np.zeros((4, 4, 3), dtype=np.uint8)
        target = tmp_path / "a" / "b" / "out.tiff"
        write_tiff(image, target)
        assert target.exists()


# ── write_png ────────────────────────────────────────────────────────────────


class TestWritePng:
    def test_writes_rgb_image_without_alpha(self, tmp_path: Path) -> None:
        image = np.full((6, 6, 3), 30, dtype=np.uint8)
        target = tmp_path / "rgb.png"

        write_png(image, target)

        with Image.open(target) as reloaded:
            assert reloaded.mode == "RGB"

    def test_writes_grayscale_image_without_alpha(self, tmp_path: Path) -> None:
        image = np.full((6, 6), 90, dtype=np.uint8)
        target = tmp_path / "gray.png"

        write_png(image, target)

        with Image.open(target) as reloaded:
            assert reloaded.mode == "L"

    def test_writes_rgb_image_with_alpha(self, tmp_path: Path) -> None:
        image = np.full((6, 6, 3), 10, dtype=np.uint8)
        alpha = np.zeros((6, 6), dtype=np.uint8)
        alpha[0:3, 0:3] = 255
        target = tmp_path / "rgba.png"

        write_png(image, target, alpha=alpha)

        with Image.open(target) as reloaded:
            assert reloaded.mode == "RGBA"
            a_channel = np.array(reloaded.split()[3])
            assert a_channel[0, 0] == 255
            assert a_channel[5, 5] == 0

    def test_writes_grayscale_image_with_alpha(self, tmp_path: Path) -> None:
        image = np.full((6, 6), 40, dtype=np.uint8)
        alpha = np.full((6, 6), 255, dtype=np.uint8)
        target = tmp_path / "gray_alpha.png"

        write_png(image, target, alpha=alpha)

        with Image.open(target) as reloaded:
            assert reloaded.mode == "RGBA"

    def test_single_pixel_image(self, tmp_path: Path) -> None:
        image = np.array([[[10, 20, 30]]], dtype=np.uint8)
        target = tmp_path / "pixel.png"

        write_png(image, target)

        with Image.open(target) as reloaded:
            assert reloaded.size == (1, 1)


# ── write_pdf ────────────────────────────────────────────────────────────────


class TestWritePdf:
    def test_raises_runtime_error_when_cairosvg_unavailable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(output_writer, "load_cairosvg", lambda logger: None)

        with pytest.raises(RuntimeError, match="CairoSVG"):
            write_pdf("<svg/>", tmp_path / "out.pdf")

    def test_writes_pdf_via_real_cairosvg(self, tmp_path: Path) -> None:
        svg_content = (
            '<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10">'
            '<rect width="10" height="10" fill="red"/></svg>'
        )
        target = tmp_path / "out.pdf"

        result = write_pdf(svg_content, target)

        assert result == target
        assert target.exists()
        assert target.stat().st_size > 0
        assert target.read_bytes().startswith(b"%PDF")

    def test_creates_missing_parent_directories(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[tuple[str, str]] = []

        class _FakeCairoSvg:
            @staticmethod
            def svg2pdf(bytestring: bytes, write_to: str) -> None:
                calls.append((bytestring.decode("utf-8"), write_to))
                Path(write_to).write_bytes(b"%PDF-fake")

        monkeypatch.setattr(output_writer, "load_cairosvg", lambda logger: _FakeCairoSvg())

        target = tmp_path / "nested" / "dir" / "out.pdf"
        write_pdf("<svg/>", target)

        assert target.exists()
        assert calls == [("<svg/>", str(target))]
