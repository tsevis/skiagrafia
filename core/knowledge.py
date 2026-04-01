from __future__ import annotations

import logging
import tomllib
from pathlib import Path
from textwrap import dedent

from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)


class KnowledgeDomain(BaseModel):
    name: str = ""
    description: str = ""


class BatchGuideDefaults(BaseModel):
    preferred_vlm: str | None = None
    fallback_vlms: list[str] = Field(default_factory=list)
    enable_tiling: bool | None = None
    max_aliases_per_object: int | None = None


class ObjectKnowledge(BaseModel):
    canonical: str
    aliases: list[str] = Field(default_factory=list)
    generic_terms: list[str] = Field(default_factory=list)
    description: str = ""
    parts: list[str] = Field(default_factory=list)
    detector_phrases: list[str] = Field(default_factory=list)

    def all_terms(self) -> list[str]:
        terms: list[str] = []
        for value in [self.canonical, *self.aliases, *self.generic_terms, *self.detector_phrases]:
            cleaned = value.strip()
            if cleaned and cleaned not in terms:
                terms.append(cleaned)
        return terms

    def ranked_detector_phrases(self, limit: int = 4) -> list[str]:
        phrases: list[str] = []
        for value in [
            *self.detector_phrases,
            *self.generic_terms,
            *self.aliases,
            self.canonical,
        ]:
            cleaned = value.strip()
            if cleaned and cleaned not in phrases:
                phrases.append(cleaned)
        if self.description:
            phrases.append(self.description.strip())
        return phrases[:limit]


class KnowledgePack(BaseModel):
    path: str = ""
    domain: KnowledgeDomain = Field(default_factory=KnowledgeDomain)
    objects: list[ObjectKnowledge] = Field(default_factory=list)
    batch_defaults: BatchGuideDefaults = Field(default_factory=BatchGuideDefaults)
    notes_markdown: str | None = None

    @property
    def name(self) -> str:
        return self.domain.name or Path(self.path).stem

    def find_object(self, label: str) -> ObjectKnowledge | None:
        wanted = label.strip().lower()
        for obj in self.objects:
            if obj.canonical.lower() == wanted:
                return obj
            if wanted in {term.lower() for term in obj.all_terms()}:
                return obj
        return None

    @classmethod
    def load(cls, path: str | Path) -> KnowledgePack:
        path = Path(path)
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
        try:
            pack = cls.model_validate(
                {
                    "path": str(path),
                    "domain": raw.get("domain", {}),
                    "objects": raw.get("objects", []),
                    "batch_defaults": raw.get("batch_defaults", {}),
                }
            )
        except ValidationError as exc:
            raise ValueError(f"Invalid knowledge pack: {path}") from exc

        md_path = path.with_suffix(".md")
        if md_path.exists():
            pack.notes_markdown = md_path.read_text(encoding="utf-8")
        return pack

    def to_toml(self) -> str:
        lines: list[str] = [
            "[domain]",
            f'name = "{_toml_escape(self.domain.name)}"',
            f'description = "{_toml_escape(self.domain.description)}"',
            "",
        ]

        defaults = self.batch_defaults
        if (
            defaults.preferred_vlm is not None
            or defaults.fallback_vlms
            or defaults.enable_tiling is not None
            or defaults.max_aliases_per_object is not None
        ):
            lines.extend(["[batch_defaults]"])
            if defaults.preferred_vlm is not None:
                lines.append(
                    f'preferred_vlm = "{_toml_escape(defaults.preferred_vlm)}"'
                )
            if defaults.fallback_vlms:
                lines.append(
                    "fallback_vlms = "
                    + _toml_list(defaults.fallback_vlms)
                )
            if defaults.enable_tiling is not None:
                lines.append(
                    f"enable_tiling = {'true' if defaults.enable_tiling else 'false'}"
                )
            if defaults.max_aliases_per_object is not None:
                lines.append(
                    f"max_aliases_per_object = {defaults.max_aliases_per_object}"
                )
            lines.append("")

        for obj in self.objects:
            lines.extend([
                "[[objects]]",
                f'canonical = "{_toml_escape(obj.canonical)}"',
                f"aliases = {_toml_list(obj.aliases)}",
                f"generic_terms = {_toml_list(obj.generic_terms)}",
                f'description = "{_toml_escape(obj.description)}"',
                f"parts = {_toml_list(obj.parts)}",
                f"detector_phrases = {_toml_list(obj.detector_phrases)}",
                "",
            ])

        return "\n".join(lines).rstrip() + "\n"

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.write_text(self.to_toml(), encoding="utf-8")
        if self.notes_markdown:
            path.with_suffix(".md").write_text(self.notes_markdown, encoding="utf-8")


def build_knowledge_pack(
    path: str | Path,
    domain_name: str,
    domain_description: str = "",
    object_specs: list[dict[str, object]] | None = None,
    preferred_vlm: str | None = None,
    fallback_vlms: list[str] | None = None,
    enable_tiling: bool | None = None,
    max_aliases_per_object: int | None = None,
    notes_markdown: str | None = None,
) -> KnowledgePack:
    objects = [
        ObjectKnowledge.model_validate(spec)
        for spec in (object_specs or [])
    ]
    return KnowledgePack(
        path=str(path),
        domain=KnowledgeDomain(name=domain_name, description=domain_description),
        objects=objects,
        batch_defaults=BatchGuideDefaults(
            preferred_vlm=preferred_vlm,
            fallback_vlms=fallback_vlms or [],
            enable_tiling=enable_tiling,
            max_aliases_per_object=max_aliases_per_object,
        ),
        notes_markdown=notes_markdown,
    )


def default_guide_markdown(domain_name: str) -> str:
    return dedent(
        f"""\
        # {domain_name or 'Skiagrafia Domain Guide'}

        Add collection-specific notes here.

        - Describe the kinds of objects expected in this batch.
        - Add naming hints the vision models should prefer.
        - Include visual clues for rare or ambiguous items.
        """
    ).strip() + "\n"


def _toml_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _toml_list(values: list[str]) -> str:
    return "[" + ", ".join(f'"{_toml_escape(v)}"' for v in values) + "]"


def load_knowledge_pack(folder: str | Path) -> KnowledgePack | None:
    folder = Path(folder)
    guide_path = folder / "skiagrafia_guide.toml"
    if not guide_path.exists():
        return None
    try:
        return KnowledgePack.load(guide_path)
    except Exception:
        logger.error("Failed to load knowledge pack %s", guide_path, exc_info=True)
        return None
