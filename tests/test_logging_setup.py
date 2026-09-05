"""test_logging_setup.py  --  Rotating debug log configuration.

The debug log runs at DEBUG level for the lifetime of the app, so it must be
size-bounded and must not be drowned by third-party transport chatter.
Nothing here touches the real log directory.
"""
from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from main import (
    LOG_BACKUP_COUNT,
    LOG_MAX_BYTES,
    NOISY_LOGGERS,
    build_file_handler,
)


class TestRotatingFileHandler:
    def test_handler_is_size_rotating(self, tmp_path: Path) -> None:
        handler = build_file_handler(tmp_path / "skiagrafia.log")
        try:
            assert isinstance(handler, logging.handlers.RotatingFileHandler)
            assert handler.maxBytes == LOG_MAX_BYTES
            assert handler.backupCount == LOG_BACKUP_COUNT
        finally:
            handler.close()

    def test_bounded_history_is_not_unlimited(self) -> None:
        # backupCount=0 would disable rotation entirely.
        assert LOG_MAX_BYTES > 0
        assert LOG_BACKUP_COUNT > 0

    def test_oversized_log_rolls_over(self, tmp_path: Path) -> None:
        log_file = tmp_path / "skiagrafia.log"
        handler = build_file_handler(log_file)
        handler.maxBytes = 512  # keep the test cheap
        record_logger = logging.getLogger("test_rollover_probe")
        record_logger.propagate = False
        record_logger.setLevel(logging.DEBUG)
        record_logger.addHandler(handler)
        try:
            for i in range(200):
                record_logger.debug("padding record %03d %s", i, "x" * 40)
        finally:
            record_logger.removeHandler(handler)
            handler.close()

        assert log_file.exists()
        assert log_file.stat().st_size <= 512 * 4, "active log outgrew its cap"
        assert (tmp_path / "skiagrafia.log.1").exists(), "no rollover happened"

    def test_total_files_capped_by_backup_count(self, tmp_path: Path) -> None:
        log_file = tmp_path / "skiagrafia.log"
        handler = build_file_handler(log_file)
        handler.maxBytes = 256
        record_logger = logging.getLogger("test_cap_probe")
        record_logger.propagate = False
        record_logger.setLevel(logging.DEBUG)
        record_logger.addHandler(handler)
        try:
            # ~90 bytes/record against a 256-byte cap: comfortably more
            # rollovers than backupCount, without writing 2000 records.
            for i in range(400):
                record_logger.debug("padding record %04d %s", i, "y" * 60)
        finally:
            record_logger.removeHandler(handler)
            handler.close()

        written = list(tmp_path.glob("skiagrafia.log*"))
        assert len(written) == LOG_BACKUP_COUNT + 1


class TestNoisyLoggers:
    def test_http_transport_loggers_are_muted(self) -> None:
        # httpcore/httpx emit one record per HTTP frame and accounted for the
        # bulk of the previous unbounded log.
        assert {"httpcore", "httpx"} <= set(NOISY_LOGGERS)

    def test_previously_muted_loggers_still_listed(self) -> None:
        assert {"PIL", "urllib3"} <= set(NOISY_LOGGERS)
