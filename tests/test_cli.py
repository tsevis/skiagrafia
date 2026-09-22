"""test_cli.py  --  the batch command line.

The 376-page APPLE50 book had to be run from a script written for the
occasion, because the application has only a GUI. That script carried four
faults before it ran correctly, and each is a trap this command line has to
close rather than re-open:

  * an empty confirmed-label list is not None, and the interrogator reads
    it as "the user confirmed nothing", returning nothing without ever
    looking at an image;
  * an empty frozen image list made the runner process the whole folder;
  * `is_running` is a property, not a method;
  * `summary()` after `close()` reads a shut database.

No models are loaded here: the interrogation and the run are injected.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cli
from cli import build_parser, main, plan


class _Summary:
    def __init__(self, failed: int = 0) -> None:
        self.total = 3
        self.completed = 3 - failed
        self.failed = failed
        self.svg_count = self.completed
        self.avg_layers = 4.0
        self.stop_reason = ""
        self.failed_image_paths: list[str] = []


def _folder(tmp_path: Path, count: int = 3) -> Path:
    source = tmp_path / "pages"
    source.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        (source / f"page{index}.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    return source


def _args(tmp_path: Path, *extra: str) -> object:
    return build_parser().parse_args(
        [str(_folder(tmp_path)), "--output", str(tmp_path / "out"), *extra]
    )


def _labels_everything(paths: list[Path], prefs: dict, guide: Path | None) -> dict:
    return {str(p): ["screen"] for p in paths}


def test_the_input_folder_and_destination_reach_the_run(tmp_path: Path) -> None:
    config = plan(_args(tmp_path), prefs={})

    assert config.input_folder == str(tmp_path / "pages")
    assert config.output_dir == str(tmp_path / "out")


def test_labels_given_on_the_command_line_are_used_as_they_are(tmp_path: Path) -> None:
    config = plan(_args(tmp_path, "--labels", "screen,keyboard"), prefs={})

    assert config.confirmed_labels == ["screen", "keyboard"]


def test_a_guide_is_passed_through_when_one_is_named(tmp_path: Path) -> None:
    guide = tmp_path / "guide.toml"
    guide.write_text('[domain]\nname = "x"\n\n[[objects]]\ncanonical = "screen"\n')

    config = plan(_args(tmp_path, "--guide", str(guide)), prefs={})

    assert config.guide_path == str(guide)


def test_a_missing_input_folder_is_refused_before_anything_loads(tmp_path: Path) -> None:
    args = build_parser().parse_args(
        [str(tmp_path / "nowhere"), "--output", str(tmp_path / "out")]
    )

    with pytest.raises(SystemExit):
        plan(args, prefs={})


def test_a_folder_with_no_images_is_refused_by_name(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    args = build_parser().parse_args([str(empty), "--output", str(tmp_path / "out")])

    with pytest.raises(SystemExit):
        plan(args, prefs={})


def test_a_rehearsal_limit_shortens_the_set(tmp_path: Path) -> None:
    # Measuring on three pages before committing to a two-hour run is how
    # every real fault in this book was found.
    config = plan(_args(tmp_path, "--limit", "2"), prefs={})

    assert len(config.input_images or []) == 2


def test_labels_are_interrogated_when_none_are_given(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asked: list[int] = []

    def _fake(paths: list[Path], prefs: dict, guide: Path | None) -> dict:
        asked.append(len(paths))
        return _labels_everything(paths, prefs, guide)

    monkeypatch.setattr(cli, "interrogate", _fake)
    monkeypatch.setattr(cli, "execute", lambda config, quiet: _Summary())

    main([str(_folder(tmp_path)), "--output", str(tmp_path / "out")])

    assert asked == [3]


def test_no_interrogation_happens_when_labels_are_given(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _never(paths: list[Path], prefs: dict, guide: Path | None) -> dict:
        raise AssertionError("the labels were given; nothing should be asked")

    monkeypatch.setattr(cli, "interrogate", _never)
    monkeypatch.setattr(cli, "execute", lambda config, quiet: _Summary())

    exit_code = main(
        [str(_folder(tmp_path)), "--output", str(tmp_path / "out"), "--labels", "screen"]
    )

    assert exit_code == 0


def test_a_run_where_nothing_got_a_label_stops_instead_of_taking_the_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The trap the script fell into: an empty image list made the runner
    # scan the whole directory and process all 376 pages blind.
    monkeypatch.setattr(cli, "interrogate", lambda paths, prefs, guide: {})
    monkeypatch.setattr(
        cli, "execute", lambda config, quiet: pytest.fail("nothing should run")
    )

    assert main([str(_folder(tmp_path)), "--output", str(tmp_path / "out")]) != 0


def test_only_the_pages_that_got_a_label_are_frozen_into_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[list[str]] = []

    def _some(paths: list[Path], prefs: dict, guide: Path | None) -> dict:
        return {str(paths[0]): ["screen"], str(paths[1]): []}

    monkeypatch.setattr(cli, "interrogate", _some)
    monkeypatch.setattr(
        cli, "execute", lambda config, quiet: (seen.append(config.input_images or []), _Summary())[1]
    )

    main([str(_folder(tmp_path)), "--output", str(tmp_path / "out")])

    assert seen and len(seen[0]) == 1


def test_the_exit_code_reports_a_run_that_lost_pages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "interrogate", _labels_everything)
    monkeypatch.setattr(cli, "execute", lambda config, quiet: _Summary(failed=2))

    assert main([str(_folder(tmp_path)), "--output", str(tmp_path / "out")]) != 0


def test_a_clean_run_reports_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "interrogate", _labels_everything)
    monkeypatch.setattr(cli, "execute", lambda config, quiet: _Summary())

    assert main([str(_folder(tmp_path)), "--output", str(tmp_path / "out")]) == 0


def test_the_interrogator_is_asked_to_look_not_told_there_is_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The one that cost an hour: `[]` is not `None`.

    An empty list means "the user confirmed these, and there are none", and
    the interrogator honours it by returning nothing without ever running a
    vision stage -- a hundred images a second, every one of them blank.
    """
    import core.interrogation as interrogation
    import processors.source_image as source_image
    from core.interrogation_types import InterrogationResult

    passed: list[object] = []

    class Recording:
        def __init__(self, settings: object) -> None:
            pass

        def interrogate(
            self, image, confirmed_labels=None, knowledge_pack=None
        ) -> InterrogationResult:
            passed.append(confirmed_labels)
            return InterrogationResult()

    monkeypatch.setattr(interrogation, "GuidedInterrogator", Recording)
    monkeypatch.setattr(cli, "interrogate", cli.interrogate)
    monkeypatch.setattr(source_image, "load_source_image", lambda p: (None, None, None))
    monkeypatch.setattr(source_image, "detection_image", lambda rgb, alpha: None)

    cli.interrogate([_folder(tmp_path, 1) / "page0.png"], {}, None)

    assert passed == [None], "an empty list is not the same as asking"
