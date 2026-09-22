"""test_console_scripts.py  --  a command that is installed must be able to run.

`[project.scripts]` created a `skiagrafia` command that imported `cli`, but
`cli` was not listed in `[tool.setuptools] py-modules`, so it was never
installed beside the entry point. The command existed and failed on its
first line:

    ModuleNotFoundError: No module named 'cli'

No test caught it. Running from the repository root puts that root on
sys.path, so `import cli` always works there; only an installed copy can
tell the difference. This checks the manifest instead of the import.
"""
from __future__ import annotations

import sys
import tomllib
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"
MANIFEST = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
SCRIPTS: dict[str, str] = MANIFEST.get("project", {}).get("scripts", {})


def _shipped_top_level() -> set[str]:
    setuptools = MANIFEST.get("tool", {}).get("setuptools", {})
    shipped = set(setuptools.get("py-modules", []))
    shipped.update(setuptools.get("packages", {}).get("find", {}).get("include", []))
    return shipped


@pytest.mark.parametrize("name,target", sorted(SCRIPTS.items()))
def test_a_console_script_points_at_a_module_that_is_shipped(
    name: str, target: str
) -> None:
    module = target.split(":")[0].split(".")[0]

    assert module in _shipped_top_level(), (
        f"the {name!r} command imports {module!r}, which pyproject does not "
        f"ship. It will install and then fail on its first line."
    )


@pytest.mark.parametrize("name,target", sorted(SCRIPTS.items()))
def test_a_console_script_names_a_callable_that_exists(name: str, target: str) -> None:
    module_name, _, attribute = target.partition(":")
    module = __import__(module_name, fromlist=[attribute or "__doc__"])

    assert callable(getattr(module, attribute)), (
        f"the {name!r} command calls {target!r}, which is not callable"
    )


def test_this_project_declares_at_least_one_command() -> None:
    # Without one the two checks above are vacuously true, which is the
    # shape of a test that cannot fail.
    assert SCRIPTS, "pyproject declares no console scripts"
