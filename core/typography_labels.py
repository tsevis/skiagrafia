"""typography_labels.py  --  reading typography out of a label or a reply.

Two questions, both about text rather than pixels: does this label name
typography at all, and does it name INDIVIDUAL glyphs (so the pipeline should
try to separate them) rather than a block of text. Plus the parser for a
vision model's glyph reading.

Split out of interrogation.py, which had grown past this project's 800-line
limit. Nothing here talks to a model or holds interrogation state: it is
string and shape handling, which is why it can be read on its own.

See core/typography_matching.py for the other half of this story -- deciding
which detector box each glyph named here actually corresponds to.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass

_TYPOGRAPHY_TERMS = frozenset(
    {
        "letter",
        "letters",
        "glyph",
        "glyphs",
        "character",
        "characters",
        "digit",
        "digits",
        "numeral",
        "numerals",
        "number",
        "numbers",
        "text",
        "typography",
    }
)
_INDIVIDUAL_GLYPH_TERMS = frozenset(
    {
        "letter",
        "letters",
        "glyph",
        "glyphs",
        "character",
        "characters",
        "digit",
        "digits",
        "numeral",
        "numerals",
        "number",
        "numbers",
    }
)


@dataclass(frozen=True)
class TypographyElement:
    """One visible glyph and its approximate normalized visual location.

    Coordinates use a 0..1000 image-relative coordinate system.  They are a
    semantic validation hint only: the detector and segmenter still provide
    the actual, pixel-accurate output geometry.
    """

    glyph: str
    bbox: tuple[int, int, int, int]


@dataclass(frozen=True)
class TypographyObservation:
    """Validated VLM reading for a generic individual-glyph request."""

    elements: tuple[TypographyElement, ...]

    @property
    def glyphs(self) -> tuple[str, ...]:
        return tuple(element.glyph for element in self.elements)


def _label_words(label: str) -> list[str]:
    """Return natural-language words without treating ``letterbox`` as letter."""
    return re.findall(r"[^\W\d_]+|\d+", label.casefold(), flags=re.UNICODE)


def is_typography_label(label: str) -> bool:
    """Whether a label denotes typography rather than a physical object.

    This intentionally uses whole words.  A ``letter opener`` remains a
    physical object, while a request for ``printed letters`` is typography.
    """
    words = _label_words(label)
    return bool(words) and words[-1] in _TYPOGRAPHY_TERMS


def is_individual_glyph_label(label: str) -> bool:
    """Whether the requested output should be one layer per visible glyph.

    ``text`` can reasonably mean one text block, whereas ``letters`` and
    ``digits`` explicitly ask for individual visible typographic objects.
    """
    words = _label_words(label)
    if not words or not (set(words) & _INDIVIDUAL_GLYPH_TERMS):
        return False
    # Do not mistake physical compounds such as "letter opener" for glyphs.
    last_typography_word = max(
        index for index, word in enumerate(words) if word in _INDIVIDUAL_GLYPH_TERMS
    )
    return last_typography_word == len(words) - 1


def parse_typography_observation(raw: str) -> TypographyObservation | None:
    """Parse a bounded, fail-closed local-VLM glyph observation.

    The VLM is never allowed to create output geometry.  A malformed answer,
    a word-level answer, or an unusable normalized box simply disables this
    optional validator and leaves the normal detector path intact.
    """
    content = raw.strip()
    if content.startswith("```"):
        content = content.removeprefix("```json").removeprefix("```")
        content = content.removesuffix("```")
        content = content.strip()
    try:
        payload = json.loads(content)
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    items = payload.get("elements")
    if not isinstance(items, list) or not 1 <= len(items) <= 64:
        return None

    elements: list[TypographyElement] = []
    for item in items:
        if not isinstance(item, dict):
            return None
        glyph = item.get("glyph")
        bbox = item.get("bbox")
        if (
            not isinstance(glyph, str)
            or not (1 <= len(glyph.strip()) <= 12)
            or not glyph.strip().isprintable()
            or not isinstance(bbox, list)
            or len(bbox) != 4
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                for value in bbox
            )
        ):
            return None
        x0, y0, x1, y1 = (max(0, min(1000, round(value))) for value in bbox)
        if x1 <= x0 or y1 <= y0:
            return None
        elements.append(TypographyElement(glyph=glyph.strip(), bbox=(x0, y0, x1, y1)))
    return TypographyObservation(elements=tuple(elements))
