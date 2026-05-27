"""
atrium_paradata.py
==================
Lightweight provenance / paradata logger for ATRIUM pipeline scripts.

Exports
-------
_sanitise(obj)        – recursively make any object JSON-serialisable
ParadataLogger        – context-manager that writes a JSON paradata record
"""
from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


# ── helpers ──────────────────────────────────────────────────────────────────

def _sanitise(obj: Any, _depth: int = 0) -> Any:
    """
    Recursively convert *obj* into a JSON-serialisable value.

    Rules
    -----
    * dict  → dict  (keys coerced to str, values recursed)
    * tuple → list  (values recursed)
    * list  → list  (values recursed)
    * bool / int / float / str / None  → unchanged
    * Anything else that is already JSON-serialisable → unchanged
    * Everything else (and anything deeper than 10 levels) → str(obj)
    """
    if _depth > 10:
        return str(obj)

    if isinstance(obj, dict):
        return {str(k): _sanitise(v, _depth + 1) for k, v in obj.items()}

    if isinstance(obj, tuple):
        return [_sanitise(i, _depth + 1) for i in obj]

    if isinstance(obj, list):
        return [_sanitise(i, _depth + 1) for i in obj]

    # bool must be tested before int because bool is a subclass of int
    if isinstance(obj, bool) or obj is None:
        return obj

    if isinstance(obj, (int, float, str)):
        return obj

    # Last resort: attempt a round-trip through json; if it fails, stringify
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return str(obj)


# ── main class ────────────────────────────────────────────────────────────────

class ParadataLogger:
    """
    Accumulates run statistics and writes a JSON paradata record on finalise.

    Usage (context-manager form – recommended)
    ------------------------------------------
    >>> with ParadataLogger("my-script", config, paradata_dir="paradata") as log:
    ...     log.log_success("csv", count=5)
    ...     log.log_skip("bad.png", "PIL cannot open")

    Usage (explicit form)
    ----------------------
    >>> log = ParadataLogger("my-script", config)
    >>> log.log_success("csv", 10)
    >>> path = log.finalize(input_total=12)
    """

    LICENSE      = "CC BY-NC 4.0"
    LICENSE_URL  = "https://creativecommons.org/licenses/by-nc/4.0/"

    def __init__(
        self,
        program_name: str,
        config: Dict[str, Any],
        output_types: Optional[List[str]] = None,
        paradata_dir: str = "paradata",
    ) -> None:
        self._program       = program_name
        self._config        = config
        self._paradata_dir  = Path(paradata_dir)
        self._output_types  = list(output_types or [])

        self._start_time    = datetime.datetime.now()
        # run_id is a compact timestamp used in the filename
        self._run_id        = self._start_time.strftime("%y%m%d-%H%M%S")

        # Counters
        self._successes: Dict[str, int] = {ot: 0 for ot in self._output_types}
        self._skips:     List[Dict[str, str]] = []
        self._finalized  = False

    # ── public API ────────────────────────────────────────────────────────────

    def log_success(self, output_type: str, count: int = 1) -> None:
        """Record *count* successfully produced outputs of *output_type*."""
        self._successes[output_type] = self._successes.get(output_type, 0) + count

    def log_skip(self, filename: str, reason: str) -> None:
        """Record one skipped input file with an explanation."""
        self._skips.append({"file": filename, "reason": reason})

    def finalize(self, input_total: Optional[int] = None) -> str:
        """
        Write the JSON paradata record and return its path.

        Parameters
        ----------
        input_total : int, optional
            Total number of input files processed.  When *None*, the value is
            inferred as ``sum(successes) + len(skips)``.

        Raises
        ------
        RuntimeError
            If called a second time on the same instance.
        """
        if self._finalized:
            raise RuntimeError(
                "finalize() has already been called on this ParadataLogger instance."
            )
        self._finalized = True

        end_time    = datetime.datetime.now()
        duration    = (end_time - self._start_time).total_seconds()

        total_processed = sum(self._successes.values())
        total_skipped   = len(self._skips)

        if input_total is None:
            input_total = total_processed + total_skipped

        record: Dict[str, Any] = {
            "program":          self._program,
            "license":          self.LICENSE,
            "license_url":      self.LICENSE_URL,
            "start_time":       self._start_time.isoformat(),
            "end_time":         end_time.isoformat(),
            "duration_seconds": duration,
            "python_version":   sys.version,
            "config":           _sanitise(self._config),
            "statistics": {
                "input_files_total":       input_total,
                "successfully_processed":  total_processed,
                "skipped_files":           total_skipped,
                "output_counts_by_type":   dict(self._successes),
            },
            "skipped_files_detail": self._skips,
        }

        self._paradata_dir.mkdir(parents=True, exist_ok=True)
        out_path = self._paradata_dir / f"{self._run_id}_{self._program}.json"
        out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        return str(out_path)

    # ── context-manager protocol ──────────────────────────────────────────────

    def __enter__(self) -> "ParadataLogger":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if not self._finalized:
            self.finalize()
        return False   # never suppress exceptions
