"""One folder per format under a run's output root.

A run wrote everything it produced into one flat directory. For a single
image that is a handful of files; for a 376-image batch it is every SVG and
every layer TIFF of every page in one folder, which is not something anyone
can open and use. Shapearator already separates its exports this way, so the
two behave alike.

Runs finished before this change are still flat on disk, so anything that
reads a run's files looks in both places.
"""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from utils.security import safe_child_path

# A file whose name carries no extension. Nothing in the pipeline produces
# one today; it exists so an unforeseen file lands somewhere rather than
# beside the shelves.
UNSHELVED = "other"


def format_shelf(filename: str) -> str:
    """The folder a produced file belongs in, named for its format."""
    return Path(filename).suffix.lower().lstrip(".") or UNSHELVED


def output_path(root: Path, filename: str) -> Path:
    """Where one produced file goes, with its folder made.

    Containment is still enforced: a name that is not a single relative
    component is refused rather than resolved.
    """
    return safe_child_path(root / format_shelf(filename), filename)


def collect_by_format(root: Path, extension: str) -> list[Path]:
    """Every regular file of one format in a run, shelved or flat.

    Symlinks are skipped: an export copies what it finds, and following a
    link would copy something the run did not produce.
    """
    shelf = extension.lower().lstrip(".")

    def candidates() -> Iterator[Path]:
        yield from (root / shelf).glob(f"*.{shelf}")
        yield from root.glob(f"*.{shelf}")

    seen: set[Path] = set()
    found: list[Path] = []
    for path in candidates():
        if path in seen or not path.is_file() or path.is_symlink():
            continue
        seen.add(path)
        found.append(path)
    return sorted(found, key=lambda p: p.name)
