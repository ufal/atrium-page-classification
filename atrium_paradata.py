"""
atrium_paradata.py  –  Unified provenance/paradata logger for ATRIUM pipelines.

DROP THIS FILE AS-IS into every ATRIUM repository root.

License of the log files themselves: CC BY-NC 4.0
https://creativecommons.org/licenses/by-nc/4.0/

Usage
-----
    from atrium_paradata import ParadataLogger

    logger = ParadataLogger(
        program="page-classification",          # short identifier for the tool
        config=vars(args),                      # any dict of run-time parameters
        paradata_dir="paradata",                # directory to write logs into (created if absent)
        output_types=["csv", "png"],            # declare all expected output file types
    )

    # during the run:
    logger.log_skip("bad_file.xml", "parse error: …")
    logger.log_success("csv")           # one csv produced
    logger.log_success("png", count=3)  # three pngs produced at once

    # at the very end (call inside a finally block):
    logger.finalize(input_total=1200)

The resulting file is written to:
    <paradata_dir>/YYMMDD-HHmmss_<program>.json
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

LICENSE_NAME = "CC BY-NC 4.0"
LICENSE_URL  = "https://creativecommons.org/licenses/by-nc/4.0/"

_REPO_URLS: Dict[str, str] = {
    "page-classification": "https://github.com/ufal/atrium-page-classification",
    "alto-postprocess":    "https://github.com/ufal/atrium-alto-postprocess",
    "nlp-enrich":          "https://github.com/ufal/atrium-nlp-enrich",
    "translator":          "https://github.com/ufal/atrium-translator",
}


# ──────────────────────────────────────────────────────────────────────────────
# ParadataLogger
# ──────────────────────────────────────────────────────────────────────────────

class ParadataLogger:
    """
    Context-manager-friendly paradata recorder.

    Parameters
    ----------
    program : str
        Short tool name, e.g. "page-classification".
    config : dict
        Snapshot of the run-time configuration (argparse namespace, config-file
        values, model identifiers, …).  Nested dicts are accepted; non-JSON-
        serialisable values are coerced to str automatically.
    paradata_dir : str
        Path to the directory where the JSON log will be written.
        Created automatically if it does not exist.
    output_types : list[str], optional
        Declare the output file types this run will produce so that performance
        counters are initialised up-front (e.g. ["csv", "png"]).
        Additional types can still be added at runtime via log_success().
    """

    def __init__(
        self,
        program: str,
        config: Dict[str, Any],
        paradata_dir: str = "paradata",
        output_types: Optional[List[str]] = None,
    ) -> None:
        self.program      = program
        self.paradata_dir = paradata_dir
        self._start_dt    = datetime.now(tz=timezone.utc)
        self._run_id      = self._start_dt.strftime("%y%m%d-%H%M%S")

        # sanitise config so it stays JSON-serialisable
        self.config = _sanitise(config)

        # counters
        self._output_counts: Dict[str, int] = {}
        if output_types:
            for t in output_types:
                self._output_counts[t] = 0

        self._skipped:  List[Dict[str, str]] = []
        self._input_total: int = 0
        self._finalised: bool  = False

        # make sure the paradata directory exists
        os.makedirs(paradata_dir, exist_ok=True)

    # ── public API ─────────────────────────────────────────────────────────────

    def log_skip(self, filepath: str, reason: str) -> None:
        """Record a file that was skipped because of an error or unsupported format."""
        self._skipped.append({
            "file":      str(filepath),
            "reason":    str(reason),
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        })

    def log_success(self, output_type: str, count: int = 1) -> None:
        """
        Increment the counter for *output_type* by *count*.

        Call this every time one or more output files of a given type are
        successfully produced.  E.g.:

            logger.log_success("csv")
            logger.log_success("xml", count=batch_size)
        """
        self._output_counts[output_type] = (
            self._output_counts.get(output_type, 0) + count
        )

    def finalize(self, input_total: Optional[int] = None) -> str:
        """
        Write the paradata JSON file and return its path.

        Parameters
        ----------
        input_total : int, optional
            Total number of input files/documents that were submitted to the
            pipeline (including skipped ones).  If None, it is inferred as
            successfully_processed + skipped.
        """
        if self._finalised:
            raise RuntimeError("finalize() has already been called.")

        end_dt          = datetime.now(tz=timezone.utc)
        duration_sec    = (end_dt - self._start_dt).total_seconds()
        duration_min    = duration_sec / 60.0 if duration_sec > 0 else 0.0

        skipped_count   = len(self._skipped)
        success_total   = sum(self._output_counts.values())
        # "successfully processed documents" = output files divided by the
        # number of output types (i.e. unique input docs that produced output)
        n_types         = max(len(self._output_counts), 1)
        processed_docs  = max(v for v in self._output_counts.values()) if self._output_counts else 0

        if input_total is None:
            input_total = processed_docs + skipped_count

        # per-type throughput (files per minute)
        perf_per_min: Dict[str, float] = {}
        for otype, cnt in self._output_counts.items():
            perf_per_min[otype] = round(cnt / duration_min, 4) if duration_min > 0 else 0.0

        payload = {
            # ── provenance ──────────────────────────────────────────────────
            "license":             LICENSE_NAME,
            "license_url":         LICENSE_URL,
            "program":             self.program,
            "repository":          _REPO_URLS.get(self.program, "https://github.com/ufal"),
            "python_version":      sys.version,
            "run_id":              self._run_id,

            # ── timing ──────────────────────────────────────────────────────
            "start_time":          self._start_dt.isoformat(),
            "end_time":            end_dt.isoformat(),
            "duration_seconds":    round(duration_sec, 3),

            # ── configuration snapshot ───────────────────────────────────────
            "config":              self.config,

            # ── statistics ───────────────────────────────────────────────────
            "statistics": {
                "input_files_total":         input_total,
                "successfully_processed":    processed_docs,
                "skipped_files":             skipped_count,
                "output_counts_by_type":     dict(self._output_counts),
                "performance_per_minute":    perf_per_min,
            },

            # ── skipped file details ─────────────────────────────────────────
            "skipped_files_detail": self._skipped,
        }

        out_path = os.path.join(
            self.paradata_dir,
            f"{self._run_id}_{self.program}.json",
        )
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)

        self._finalised = True
        print(f"[paradata] Log written → {out_path}", flush=True)
        return out_path

    # ── context manager support ───────────────────────────────────────────────

    def __enter__(self) -> "ParadataLogger":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Automatically finalise on exit, even if an exception was raised."""
        if not self._finalised:
            try:
                self.finalize()
            except Exception as e:   # never let logging crash the program
                print(f"[paradata] WARNING – could not write log: {e}", file=sys.stderr)
        return False   # do not suppress exceptions


# ──────────────────────────────────────────────────────────────────────────────
# CLI shim – used by Bash scripts (atrium-nlp-enrich)
# ──────────────────────────────────────────────────────────────────────────────

def _cli() -> None:
    """
    Thin command-line interface so that Bash scripts can drive the logger via a
    persistent state file in the paradata directory.

    Commands
    --------
    start   --program NAME --config KEY=VAL [KEY=VAL ...]  [--paradata-dir DIR]
    skip    --state STATE_FILE --file PATH --reason REASON
    success --state STATE_FILE --type TYPE [--count N]
    finish  --state STATE_FILE [--input-total N]
    """
    import argparse, pickle

    p = argparse.ArgumentParser(prog="python atrium_paradata.py")
    sub = p.add_subparsers(dest="cmd", required=True)

    # start
    s = sub.add_parser("start")
    s.add_argument("--program",      required=True)
    s.add_argument("--config",       nargs="*", default=[],
                   help="KEY=VALUE pairs")
    s.add_argument("--output-types", nargs="*", default=[])
    s.add_argument("--paradata-dir", default="paradata")

    # skip
    sk = sub.add_parser("skip")
    sk.add_argument("--state",  required=True)
    sk.add_argument("--file",   required=True)
    sk.add_argument("--reason", required=True)

    # success
    su = sub.add_parser("success")
    su.add_argument("--state", required=True)
    su.add_argument("--type",  required=True)
    su.add_argument("--count", type=int, default=1)

    # finish
    fi = sub.add_parser("finish")
    fi.add_argument("--state",       required=True)
    fi.add_argument("--input-total", type=int, default=None)

    args = p.parse_args()

    if args.cmd == "start":
        cfg: Dict[str, Any] = {}
        for kv in (args.config or []):
            k, _, v = kv.partition("=")
            cfg[k.strip()] = v.strip()
        logger = ParadataLogger(
            program=args.program,
            config=cfg,
            paradata_dir=args.paradata_dir,
            output_types=args.output_types or None,
        )
        state_path = os.path.join(
            args.paradata_dir, f".state_{logger._run_id}_{args.program}.pkl"
        )
        with open(state_path, "wb") as fh:
            pickle.dump(logger, fh)
        # print the state file path so the shell script can capture it
        print(state_path)

    elif args.cmd in ("skip", "success", "finish"):
        with open(args.state, "rb") as fh:
            logger = pickle.load(fh)

        if args.cmd == "skip":
            logger.log_skip(args.file, args.reason)
        elif args.cmd == "success":
            logger.log_success(args.type, args.count)
        elif args.cmd == "finish":
            logger.finalize(input_total=getattr(args, "input_total", None))
            os.remove(args.state)
            return

        # persist updated state
        with open(args.state, "wb") as fh:
            pickle.dump(logger, fh)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _sanitise(obj: Any, _depth: int = 0) -> Any:
    """Recursively coerce a dict/list to be JSON-serialisable."""
    if _depth > 10:
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _sanitise(v, _depth + 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitise(v, _depth + 1) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


if __name__ == "__main__":
    _cli()