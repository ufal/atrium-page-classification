"""
atrium_paradata.py  –  Unified paradata logger for ATRIUM pipelines (paradata schema 3.0).

A run's paradata describe the run that produced the current version of a distribution
(contract §6, atrium-project#84). Schema 3.0:

* **No licence.** No `license`, `license_url` or `license_detail`: nothing is computed or
  merged (D2). `components` lists each component used — name, version when known, and its
  terms (`para_licenses.component_terms`) — as facts.
* **Inputs and outputs by distribution name and SHA-512** (`add_input()`, `add_output()`):
  never a local path or a SHA-256 `ni:` identifier. An input or output that is not stored is
  named by its SHA-512 and marked `stored: false` (D23); for the record (`atrium/document-json`)
  an output names the blocks AND fields the run wrote (`record_fields`, from
  `DocumentRecord.written()`).
* **`action_status`** `completed`, or `failed` with `error` when the run raised (D13: what AMČR
  stores is always completed; development and tests use the same frame).
* **`parameters`** only what shaped the output (`set_parameter()`); **`limits`** the sampling and
  splitting notes only. A trimmed, skipped or stopped content is not a note: it raises
  `ContentLimitError` and the job fails (D13).

`run_uuid` (``urn:uuid:`` + a random UUID, minted when the run starts) is the run's stable id
and the `@id` of its RO-Crate `CreateAction` (`atrium_rocrate.create_action`). `run_agent` is
``ATRIUM_RUN_AGENT``: the IRI of the organisation that operates the deployment.

A service creates its logger with ``paradata_dir=None``: nothing is written to the container's
filesystem, ``record`` holds the finalized payload, and ``paradata_ref`` is the ``run_uuid``.
"""

from __future__ import annotations

import configparser
import hashlib
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

try:
    from para_licenses import component_terms
except ImportError:  # a bundle that copies only this file still lists its components

    def component_terms(name: str, license: Optional[str], version: Optional[str] = None) -> Dict[str, str]:
        row = {"name": str(name)}
        if version:
            row["version"] = str(version)
        row["license"] = (license or "").strip() or "UNKNOWN"
        return row


try:
    from atrium_limits import ContentLimitError
except ImportError:  # the same contract without the module beside this one

    class ContentLimitError(RuntimeError):  # type: ignore[no-redef]
        """A limit trimmed, skipped or stopped content: the step fails (D13)."""

        reason = "limit_exceeded"

        def __init__(self, limit: str, effect: str, count: int = 1, detail: str = "") -> None:
            self.limit, self.effect, self.count, self.detail = limit, effect, count, detail
            super().__init__(f"limit {limit!r} {effect} content ({count} unit(s)) — the step fails (D13)")


# ──────────────────────────────────────────────────────────────────────────────
# Constants & Schema version
# ──────────────────────────────────────────────────────────────────────────────

SCHEMA_VERSION = "3.0"
# Current programs first; the predecessors (alto-postprocess, llm-enrich) stay for records written
# before the repository moves of 2026-10-01 -- see atrium_document.PROGRAM_SUCCESSORS.
_REPO_URLS: Dict[str, str] = {
    "page-classification": "https://github.com/ufal/atrium-page-classification",
    "ocr-postprocess": "https://github.com/ufal/atrium-ocr-postprocess",
    "alto-postprocess": "https://github.com/ufal/atrium-alto-postprocess",
    "nlp-enrich": "https://github.com/ufal/atrium-nlp-enrich",
    "keyword-extract": "https://github.com/ufal/atrium-keyword-extract",
    "translator": "https://github.com/ufal/atrium-translator",
    "llm-enrich": "https://github.com/ufal/atrium-llm-enrich",
    "digital-convert": "https://github.com/ufal/atrium-digital-convert",
}

#: Every effect a limit note may name (the same tuple as ``atrium_limits.EFFECTS``; read-compatible
#: with notes written before 3.0), the two a 3.0 run records, and the three that fail it (D13).
_LIMIT_EFFECTS = ("sampled", "split", "trimmed", "skipped", "stopped")
_RECORDED_EFFECTS = ("sampled", "split")
_FAILING_EFFECTS = ("trimmed", "skipped", "stopped")

#: `action_status` values (schema.org ActionStatusType names, as atrium_rocrate maps them).
COMPLETED = "completed"
FAILED = "failed"

#: The record's distribution: an output with this name may name the record fields written.
RECORD_DISTRIBUTION = "atrium/document-json"

_ENV_RUNNER_IMAGE = "ATRIUM_RUNNER_IMAGE"
_ENV_RUNNER_REPO = "ATRIUM_RUNNER_REPO"
_ENV_RUNNER_REF = "ATRIUM_RUNNER_REF"
_ENV_RUN_AGENT = "ATRIUM_RUN_AGENT"


def _new_run_uuid() -> str:
    """A run's stable id: ``urn:uuid:`` and a random (version 4) UUID, lower case."""
    return uuid.uuid4().urn


def _load_para_config(start_dir: str = ".") -> Dict[str, Any]:
    path = os.path.join(start_dir, "para_config.txt")
    out: Dict[str, Any] = {"components": []}
    if not os.path.exists(path):
        return out

    cfg = configparser.ConfigParser()
    cfg.read(path, encoding="utf-8")

    if cfg.has_section("tool"):
        out["program"] = cfg.get("tool", "program", fallback=None)
        out["version"] = cfg.get("tool", "version", fallback=None)
        out["repository_fallback"] = cfg.get("tool", "repository_fallback", fallback=None)

    if cfg.has_section("components"):
        for name, spec in cfg.items("components"):
            fields = [s.strip() for s in spec.split(";")]
            lic = fields[0] if len(fields) > 0 else ""
            loaded = fields[1] if len(fields) > 1 else "always"
            role = fields[2] if len(fields) > 2 else ""
            out["components"].append(
                {
                    "name": name.strip(),
                    "license": lic,
                    "loaded": loaded,
                    "role": role,
                }
            )
    return out


def sha512_of(data: Any = None, path: Optional[str] = None) -> str:
    """The lower-case hex SHA-512 of `data` (bytes, or str as UTF-8) or of the file at `path`."""
    digest = hashlib.sha512()
    if path is not None:
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
    elif isinstance(data, str):
        digest.update(data.encode("utf-8"))
    elif data is not None:
        digest.update(bytes(data))
    else:
        raise ValueError("sha512_of() needs data or a path")
    return digest.hexdigest()


def _distribution(
    name: str,
    data: Any,
    path: Optional[str],
    sha512: Optional[str],
    media_type: Optional[str],
    stored: bool,
) -> Dict[str, Any]:
    if not name:
        raise ValueError("an input or output is named by its AMČR distribution name (e.g. atrium/document-json)")
    if sha512 is None:
        sha512 = sha512_of(data, path)
    sha512 = str(sha512).lower()
    if len(sha512) != 128 or any(c not in "0123456789abcdef" for c in sha512):
        raise ValueError(f"{name}: sha512 must be 128 lower-case hex characters")
    entry: Dict[str, Any] = {"name": str(name), "sha512": sha512}
    if media_type:
        entry["media_type"] = str(media_type)
    entry["stored"] = bool(stored)
    return entry


class ParadataLogger:
    def __init__(
        self,
        program: str,
        config: Dict[str, Any],
        paradata_dir: Optional[str] = "paradata",
        output_types: Optional[List[str]] = None,
        version: Optional[str] = None,
        docker_image: Optional[str] = None,
        config_dir: str = ".",
    ) -> None:
        self.program = program
        self.paradata_dir = paradata_dir
        self._start_dt = datetime.now(tz=timezone.utc)
        self._run_id = self._start_dt.strftime("%y%m%d-%H%M%S")
        self._run_uuid = _new_run_uuid()
        self._record: Optional[Dict[str, Any]] = None

        self._para_cfg = _load_para_config(config_dir)
        self.version = version or self._para_cfg.get("version") or "unknown"
        self.docker_image = docker_image or os.environ.get(_ENV_RUNNER_IMAGE) or ""
        self.config = _sanitise(config)

        self._output_counts: Dict[str, int] = {}
        if output_types:
            for t in output_types:
                self._output_counts[t] = 0

        self._docs_processed: int = 0
        #: name -> terms, and name -> version when known (component_terms() renders the row).
        self._components_used: Dict[str, str] = {}
        self._component_versions: Dict[str, str] = {}
        for comp in self._para_cfg.get("components", []):
            if comp.get("loaded") == "always":
                self._components_used[comp["name"]] = comp["license"]

        self._skipped: List[Dict[str, str]] = []
        self._limits: List[Dict[str, Any]] = []
        self._inputs: List[Dict[str, Any]] = []
        self._outputs: List[Dict[str, Any]] = []
        self._parameters: Dict[str, Any] = {}
        self._action_status: str = COMPLETED
        self._error: str = ""
        self._input_total: int = 0
        self._finalised: bool = False

        if paradata_dir:
            os.makedirs(paradata_dir, exist_ok=True)

    # ── what the run read, wrote and was given ──────────────────────────────

    def add_input(
        self,
        name: str,
        data: Any = None,
        *,
        path: Optional[str] = None,
        sha512: Optional[str] = None,
        media_type: Optional[str] = None,
        stored: bool = True,
    ) -> Dict[str, Any]:
        """Record an input (the action's `object`) by AMČR distribution name and SHA-512.

        Pass the bytes, a `path` to hash, or the `sha512` itself; the path is read, never recorded.
        For a record input this is the record version read. An input that is not stored (e.g.
        digital-convert's docproc of a mixed document, H14) is `stored=False` and still named by
        its SHA-512 (D23).
        """
        entry = _distribution(name, data, path, sha512, media_type, stored)
        self._inputs.append(entry)
        return dict(entry)

    def add_output(
        self,
        name: str,
        data: Any = None,
        *,
        path: Optional[str] = None,
        sha512: Optional[str] = None,
        media_type: Optional[str] = None,
        stored: bool = True,
        record_fields: Optional[Dict[str, List[str]]] = None,
    ) -> Dict[str, Any]:
        """Record an output (the action's `result`) by AMČR distribution name and SHA-512.

        For the record (`atrium/document-json`), `record_fields` names the blocks and fields this
        run wrote — `DocumentRecord.written()` (#84 §4). An output that is not stored but reaches
        a stored distribution is `stored=False` (D23); an output that reaches none is not recorded.
        """
        entry = _distribution(name, data, path, sha512, media_type, stored)
        if record_fields is not None:
            entry["record_fields"] = {str(k): [str(f) for f in v] for k, v in record_fields.items()}
        self._outputs.append(entry)
        return dict(entry)

    def set_parameter(self, key: str, value: Any) -> None:
        """Record a parameter that SHAPED the output (#84 §4): a model revision, a threshold, the
        forced pages, the translator's field list. Not a path, not run data."""
        self._parameters[str(key)] = _sanitise(value)

    def fail(self, error: str) -> None:
        """Mark the run failed (`action_status: failed`) with a short error. Nothing it produced may
        be stored (D13)."""
        self._action_status = FAILED
        self._error = str(error or "failed")

    @property
    def action_status(self) -> str:
        return self._action_status

    def log_skip(self, filepath: str, reason: str) -> None:
        """A batch input not processed. Named by its file name only: a local path is not paradata."""
        self._skipped.append(
            {
                "file": os.path.basename(str(filepath)),
                "reason": str(reason),
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            }
        )

    def note_limit(self, limit: str, value: Any, effect: str, count: int = 1, detail: str = "") -> None:
        """Record that ``limit`` (its ``/info`` key) shaped the result by sampling or splitting.

        Notes for the same limit and effect are merged: counts add up, the first non-empty detail
        is kept. A ``trimmed``, ``skipped`` or ``stopped`` effect raises ``ContentLimitError``: the
        job fails instead (D13).
        """
        if effect not in _LIMIT_EFFECTS:
            raise ValueError(f"note_limit(): effect must be one of {_LIMIT_EFFECTS}, not {effect!r}")
        if effect in _FAILING_EFFECTS and count > 0:
            raise ContentLimitError(limit, effect, count, detail)
        _merge_limit_notes(
            self._limits,
            [{"limit": limit, "value": value, "effect": effect, "count": count, "detail": detail}],
        )

    def note_limits(self, notes: Any) -> None:
        """Add every note of an ``atrium_limits.LimitNotes`` (or an iterable of note dicts); a
        failing effect among them raises ``ContentLimitError`` (D13)."""
        dicts = _notes_as_dicts(notes)
        for note in dicts:
            if note.get("effect") in _FAILING_EFFECTS and int(note.get("count", 1) or 0) > 0:
                raise ContentLimitError(
                    str(note.get("limit", "")), str(note["effect"]), int(note.get("count", 1) or 1), str(note.get("detail") or "")
                )
        _merge_limit_notes(self._limits, dicts)

    @property
    def limits(self) -> List[Dict[str, Any]]:
        """The sampling and splitting notes recorded so far, as dicts (a copy)."""
        return [dict(n) for n in self._limits]

    @property
    def limits_applied(self) -> List[Dict[str, Any]]:
        """Alias of `limits`, the 2.x name, kept for callers that read it."""
        return self.limits

    def get_license_block(self) -> dict:
        """Removed in 3.0 (D2): no licence is computed. Use `components()` — the facts."""
        raise RuntimeError(
            "paradata 3.0 computes no licence (D2, atrium-project#84): list components with their terms "
            "(log_component) and read them with components()"
        )

    def components(self) -> List[Dict[str, str]]:
        """Each component used, with its version when known and its terms — facts, never merged."""
        return [component_terms(n, lic, self._component_versions.get(n)) for n, lic in self._components_used.items()]

    @property
    def run_id(self) -> str:
        """This logger's run id (one-second resolution)."""
        return self._run_id

    @property
    def run_uuid(self) -> str:
        """This run's stable id, ``urn:uuid:...``: unique where ``run_id`` is not."""
        return self._run_uuid

    @property
    def paradata_ref(self) -> str:
        """The file ``finalize()`` writes, or, for a logger that writes none (``paradata_dir=None``),
        the ``run_uuid``."""
        if not self.paradata_dir:
            return self._run_uuid
        return os.path.join(self.paradata_dir, f"{self._run_id}_{self.program}.json")

    @property
    def record(self) -> Optional[Dict[str, Any]]:
        """The paradata record ``finalize()`` built, or None before it ran."""
        return self._record

    def log_success(self, output_type: str, count: int = 1) -> None:
        self._output_counts[output_type] = self._output_counts.get(output_type, 0) + count

    def log_document_success(self) -> None:
        self._docs_processed += 1

    def log_component(self, name: str, license: Optional[str] = None, version: Optional[str] = None) -> None:
        """Record a component the run used, with its terms (from `para_config.txt` when not given)
        and its version when known. A fact; nothing is derived from it (D2)."""
        if license is None:
            for comp in self._para_cfg.get("components", []):
                if comp["name"] == name:
                    license = comp["license"]
                    break
        self._components_used[name] = license or "UNKNOWN"
        if version:
            self._component_versions[name] = str(version)

    def _resolve_repository(self) -> str:
        return (
            os.environ.get(_ENV_RUNNER_REPO)
            or self._para_cfg.get("repository_fallback")
            or _REPO_URLS.get(self.program, "https://github.com/ufal")
        )

    def finalize(
        self,
        input_total: Optional[int] = None,
        processed_total: Optional[int] = None,
    ) -> str:
        """
        Build the paradata record and write it to ``paradata_dir``; return the path written.
        With ``paradata_dir=None`` nothing is written and ``""`` is returned: read ``record``.
        Precedence for processed_docs: processed_total (arg) -> _docs_processed -> max(output_counts).
        """
        if self._finalised:
            raise RuntimeError("finalize() has already been called.") from None

        end_dt = datetime.now(tz=timezone.utc)
        duration_sec = (end_dt - self._start_dt).total_seconds()
        duration_min = duration_sec / 60.0 if duration_sec > 0 else 0.0

        skipped_count = len(self._skipped)

        if processed_total is not None:
            processed_docs = processed_total
        elif self._docs_processed > 0:
            processed_docs = self._docs_processed
        else:
            processed_docs = max(self._output_counts.values()) if self._output_counts else 0

        if input_total is None:
            input_total = processed_docs + skipped_count

        perf_per_min: Dict[str, float] = {}
        for otype, cnt in self._output_counts.items():
            perf_per_min[otype] = round(cnt / duration_min, 4) if duration_min > 0 else 0.0

        payload: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "program": self.program,
            "tool_version": self.version,
            "repository": self._resolve_repository(),
            "runner_ref": os.environ.get(_ENV_RUNNER_REF, ""),
            "docker_image": self.docker_image,
            "python_version": sys.version,
            "run_id": self._run_id,
            "run_uuid": self._run_uuid,
            "run_agent": os.environ.get(_ENV_RUN_AGENT, "").strip(),
            "action_status": self._action_status,
        }
        if self._error:
            payload["error"] = self._error
        payload.update(
            {
                "start_time": self._start_dt.isoformat(),
                "end_time": end_dt.isoformat(),
                "duration_seconds": round(duration_sec, 3),
                "config": self.config,
                "parameters": dict(self._parameters),
                "inputs": [dict(e) for e in self._inputs],
                "outputs": [dict(e) for e in self._outputs],
                "components": self.components(),
                "statistics": {
                    "input_files_total": input_total,
                    "successfully_processed": processed_docs,
                    "skipped_files": skipped_count,
                    "output_counts_by_type": dict(self._output_counts),
                    "performance_per_minute": perf_per_min,
                },
                "skipped_files_detail": self._skipped,
                "limits": self.limits,
            }
        )

        self._record = payload
        self._finalised = True
        if not self.paradata_dir:
            return ""
        out_path = self.paradata_ref
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        print(f"[paradata] Log written → {out_path}", flush=True)
        return out_path

    def __enter__(self) -> "ParadataLogger":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_type is not None and self._action_status != FAILED:
            self.fail(f"{exc_type.__name__}: {exc_val}")
        if not self._finalised:
            try:
                self.finalize()
            except Exception as e:
                print(f"[paradata] WARNING – could not write log: {e}", file=sys.stderr)
        return False

    def _to_state_dict(self) -> Dict[str, Any]:
        return {
            "program": self.program,
            "version": self.version,
            "config": self.config,
            "paradata_dir": self.paradata_dir,
            "output_counts": self._output_counts,
            "components_used": self._components_used,
            "component_versions": self._component_versions,
            "skipped": self._skipped,
            "limits": self._limits,
            "inputs": self._inputs,
            "outputs": self._outputs,
            "parameters": self._parameters,
            "action_status": self._action_status,
            "error": self._error,
            "docs_processed": self._docs_processed,
            "start_iso": self._start_dt.isoformat(),
            "run_id": self._run_id,
            "run_uuid": self._run_uuid,
            "para_cfg": self._para_cfg,
        }

    @classmethod
    def from_state_file(cls, path: str) -> "ParadataLogger":
        """The run a CLI ``start`` state file holds, for a step of a shell-driven stage that needs
        it mid-run: its ``run_id``, ``run_uuid`` and ``paradata_ref``. Reading writes nothing back."""
        with open(path, "r", encoding="utf-8") as fh:
            return cls._from_state_dict(json.load(fh))

    @classmethod
    def _from_state_dict(cls, d: Dict[str, Any]) -> "ParadataLogger":
        inst = cls.__new__(cls)
        inst.program = d["program"]
        inst.version = d.get("version", "unknown")
        inst.config = d["config"]
        inst.paradata_dir = d["paradata_dir"]
        inst._output_counts = d["output_counts"]
        inst._components_used = d.get("components_used", {})
        inst._component_versions = d.get("component_versions", {})
        inst._skipped = d["skipped"]
        # A 2.x state file calls them limits_applied; only the recorded effects carry over.
        notes = d.get("limits", d.get("limits_applied", [])) or []
        inst._limits = [n for n in notes if n.get("effect") in _RECORDED_EFFECTS]
        inst._inputs = list(d.get("inputs", []) or [])
        inst._outputs = list(d.get("outputs", []) or [])
        inst._parameters = dict(d.get("parameters", {}) or {})
        inst._action_status = d.get("action_status", COMPLETED)
        inst._error = d.get("error", "")
        inst._docs_processed = d.get("docs_processed", 0)
        inst._run_id = d["run_id"]
        # A state file written before run_uuid existed gets one now: the run is still one run.
        inst._run_uuid = d.get("run_uuid") or _new_run_uuid()
        inst._record = None
        inst._start_dt = datetime.fromisoformat(d["start_iso"])
        inst._para_cfg = d.get("para_cfg", {"components": []})
        inst.docker_image = d.get("docker_image", "")
        inst._input_total = 0
        inst._finalised = False
        return inst


# ──────────────────────────────────────────────────────────────────────────────
# Reader & Migration
# ──────────────────────────────────────────────────────────────────────────────


def _migrate_1_0_to_2_0(record: Dict[str, Any]) -> Dict[str, Any]:
    new_record = dict(record)
    new_record["schema_version"] = "2.0"
    if "docker_image" not in new_record:
        new_record["docker_image"] = ""
    return new_record


def _migrate_2_0_to_3_0(record: Dict[str, Any]) -> Dict[str, Any]:
    """2.x → 3.0: the calculated licence goes (D2); its component list becomes `components`
    (facts); `limits_applied` becomes `limits`; `action_status` is `completed` (a 2.x record was
    written by a run that finished)."""
    new_record = {k: v for k, v in record.items() if k not in ("license", "license_url", "license_detail", "limits_applied")}
    detail = record.get("license_detail") or {}
    comps = detail.get("components") if isinstance(detail, dict) else None
    if "components" not in new_record:
        new_record["components"] = [
            component_terms(c.get("name", ""), c.get("license")) for c in (comps or []) if isinstance(c, dict)
        ]
    new_record.setdefault("limits", list(record.get("limits_applied") or []))
    new_record.setdefault("action_status", COMPLETED)
    new_record["schema_version"] = "3.0"
    return new_record


def migrate_paradata(record: Dict[str, Any]) -> Dict[str, Any]:
    """Applies schema migrations up to the current SCHEMA_VERSION."""
    v = str(record.get("schema_version") or "1.0")
    if v.startswith("1."):
        record = _migrate_1_0_to_2_0(record)
        v = "2.0"
    if v.startswith("2."):
        record = _migrate_2_0_to_3_0(record)
    return record


def load_paradata(path: str) -> Dict[str, Any]:
    """Reads a paradata file, migrating older schemas transparently."""
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    v = data.get("schema_version", "1.0")
    major = int(v.split(".")[0])
    current_major = int(SCHEMA_VERSION.split(".")[0])

    if major > current_major:
        raise ValueError(f"Schema version {v} is newer than supported {SCHEMA_VERSION}. Please update tools.")
    elif major < current_major:
        data = migrate_paradata(data)

    return data


# ──────────────────────────────────────────────────────────────────────────────
# Merging Logic
# ──────────────────────────────────────────────────────────────────────────────


def _union_components(into: List[Dict[str, str]], rows: Any) -> None:
    """Add component rows not already listed (same name, version and terms). Facts are listed,
    never combined into one licence (D2)."""
    for row in rows or []:
        if isinstance(row, dict) and row not in into:
            into.append(dict(row))


def merge_paradata_files(json_paths: List[str], input_file: str, out_path: str) -> str:
    steps: List[Dict[str, Any]] = []
    components: List[Dict[str, str]] = []
    limits: List[Dict[str, Any]] = []
    total_duration = 0.0

    for p in json_paths:
        data = load_paradata(p)
        limits.extend(_tag_notes(data))
        steps.append(
            {
                "program": data.get("program"),
                "tool_version": data.get("tool_version"),
                "repository": data.get("repository"),
                "docker_image": data.get("docker_image"),
                "run_id": data.get("run_id"),
                "run_uuid": data.get("run_uuid", ""),
                "action_status": data.get("action_status", COMPLETED),
                "duration_seconds": data.get("duration_seconds"),
                "config": data.get("config"),
            }
        )
        _union_components(components, data.get("components"))
        total_duration += float(data.get("duration_seconds") or 0.0)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "record_type": "single-file-merged",
        "run_uuid": _new_run_uuid(),
        "input_file": os.path.basename(str(input_file)),
        "pipeline_steps": steps,
        "step_count": len(steps),
        "total_duration_seconds": round(total_duration, 3),
        "components": components,
        "limits": limits,
        "merged_at": datetime.now(tz=timezone.utc).isoformat(),
    }

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    print(f"[paradata] Merged single-file log → {out_path}", flush=True)
    return out_path


def merge_run_paradata(
    json_paths: List[str],
    out_path: str,
    pipeline: Optional[str] = None,
    method: Optional[str] = None,
    skipped_stages: Optional[List[str]] = None,
) -> str:
    stages: List[Dict[str, Any]] = []
    components: List[Dict[str, str]] = []
    formats: Dict[str, int] = {}
    total_duration = 0.0
    total_inputs = 0
    total_processed = 0
    total_skipped = 0
    all_skips: List[Dict[str, Any]] = []
    limits: List[Dict[str, Any]] = []
    repo = ""
    tool_version = ""
    docker_image = ""
    earliest: Optional[str] = None
    latest: Optional[str] = None
    first_stage = True
    status = COMPLETED

    for order, p in enumerate(json_paths, 1):
        data = load_paradata(p)

        repo = repo or data.get("repository", "")
        tool_version = tool_version or data.get("tool_version", "")
        docker_image = docker_image or data.get("docker_image", "")

        cfg = data.get("config", {}) or {}
        stats = data.get("statistics", {}) or {}
        out_counts = stats.get("output_counts_by_type", {}) or {}

        for ftype, cnt in out_counts.items():
            formats[ftype] = formats.get(ftype, 0) + int(cnt or 0)

        total_duration += float(data.get("duration_seconds") or 0.0)

        # Only take input_files_total from the very first stage
        if first_stage:
            total_inputs = int(stats.get("input_files_total") or 0)
            first_stage = False

        # Overwrite successfully processed so the final value mirrors the last stage's successes
        total_processed = int(stats.get("successfully_processed") or 0)
        total_skipped += int(stats.get("skipped_files") or 0)
        all_skips.extend(data.get("skipped_files_detail", []) or [])
        limits.extend(_tag_notes(data))
        _union_components(components, data.get("components"))
        if data.get("action_status") == FAILED:
            status = FAILED

        st = data.get("start_time")
        en = data.get("end_time")
        if st and (earliest is None or st < earliest):
            earliest = st
        if en and (latest is None or en > latest):
            latest = en

        stages.append(
            {
                "order": order,
                "program": data.get("program"),
                "script": cfg.get("script"),
                "method": cfg.get("method"),
                "run_id": data.get("run_id"),
                "run_uuid": data.get("run_uuid", ""),
                "action_status": data.get("action_status", COMPLETED),
                "repository": data.get("repository"),
                "tool_version": data.get("tool_version"),
                "docker_image": data.get("docker_image"),
                "inputs": data.get("inputs", []),
                "outputs": data.get("outputs", []),
                "output_formats": out_counts,
                "start_time": st or "",
                "end_time": en or "",
                "duration_seconds": data.get("duration_seconds"),
                "input_files_total": stats.get("input_files_total"),
                "successfully_processed": stats.get("successfully_processed"),
                "skipped_files": stats.get("skipped_files"),
            }
        )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "record_type": "pipeline-run-merged",
        "pipeline": pipeline or (stages[0].get("program") if stages else "unknown"),
        "method": method or "",
        "repository": repo,
        "tool_version": tool_version,
        "docker_image": docker_image,
        "runner_ref": os.environ.get(_ENV_RUNNER_REF, ""),
        "request_id": os.environ.get("ATRIUM_REQUEST_ID", ""),
        "python_version": sys.version,
        "run_id": datetime.now(tz=timezone.utc).strftime("%y%m%d-%H%M%S"),
        "run_uuid": _new_run_uuid(),
        "run_agent": os.environ.get(_ENV_RUN_AGENT, "").strip(),
        "action_status": status,
        "stage_count": len(stages),
        "pipeline_stages": stages,
        "intermediate_formats": formats,
        "components": components,
        "start_time": earliest or "",
        "end_time": latest or "",
        "total_duration_seconds": round(total_duration, 3),
        "statistics": {
            "stages_total": len(stages),
            "input_files_total": total_inputs,
            "successfully_processed": total_processed,
            "skipped_files": total_skipped,
        },
        "skipped_files_detail": all_skips,
        "limits": limits,
        "skipped_stages": skipped_stages or [],
        "merged_at": datetime.now(tz=timezone.utc).isoformat(),
    }

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    print(f"[paradata] Merged pipeline-run log → {out_path}", flush=True)
    return out_path


def _notes_as_dicts(notes: Any) -> List[Dict[str, Any]]:
    if notes is None:
        return []
    as_list = getattr(notes, "as_list", None)
    if callable(as_list):
        return list(as_list())
    out: List[Dict[str, Any]] = []
    for n in notes:
        to_dict = getattr(n, "to_dict", None)
        out.append(dict(to_dict()) if callable(to_dict) else dict(n))
    return out


def _merge_limit_notes(into: List[Dict[str, Any]], notes: List[Dict[str, Any]]) -> None:
    for note in notes:
        count = int(note.get("count", 1) or 0)
        if count <= 0 or note.get("effect") not in _LIMIT_EFFECTS:
            continue
        for existing in into:
            if existing.get("limit") == note.get("limit") and existing.get("effect") == note.get("effect"):
                existing["count"] = int(existing.get("count", 0)) + count
                if not existing.get("detail") and note.get("detail"):
                    existing["detail"] = str(note["detail"])
                break
        else:
            into.append(
                {
                    "limit": str(note.get("limit", "")),
                    "value": _sanitise(note.get("value")),
                    "effect": str(note.get("effect", "")),
                    "count": count,
                    "detail": str(note.get("detail", "") or ""),
                }
            )


def _tag_notes(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """A record's ``limits`` (``limits_applied`` before 3.0), each note tagged with its ``program``."""
    program = data.get("program")
    notes = data.get("limits", data.get("limits_applied")) or []
    return [{"program": note.get("program", program), **note} for note in notes]


def _sanitise(obj: Any, _depth: int = 0) -> Any:
    if _depth > 10:
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _sanitise(v, _depth + 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitise(v, _depth + 1) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def _cli() -> None:
    import argparse

    p = argparse.ArgumentParser(prog="python atrium_paradata.py")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("start")
    s.add_argument("--program", required=True)
    s.add_argument("--config", nargs="*", default=[])
    s.add_argument("--output-types", nargs="*", default=[])
    s.add_argument("--paradata-dir", default="paradata")
    s.add_argument("--component", nargs="*", default=[])

    sk = sub.add_parser("skip")
    sk.add_argument("--state", required=True)
    sk.add_argument("--file", required=True)
    sk.add_argument("--reason", required=True)

    su = sub.add_parser("success")
    su.add_argument("--state", required=True)
    su.add_argument("--type", required=True)
    su.add_argument("--count", type=int, default=1)
    su.add_argument("--component", nargs="*", default=[])

    co = sub.add_parser("component")
    co.add_argument("--state", required=True)
    co.add_argument("--name", required=True)
    co.add_argument("--license", default=None)
    co.add_argument("--version", default=None)

    nl = sub.add_parser("note-limit", help="record sampling or splitting; a trimmed/skipped/stopped effect fails (exit 1)")
    nl.add_argument("--state", required=True)
    nl.add_argument("--limit", required=True)
    nl.add_argument("--value", default=None)
    nl.add_argument("--effect", required=True)
    nl.add_argument("--count", type=int, default=1)
    nl.add_argument("--detail", default="")

    for verb in ("input", "output"):
        io = sub.add_parser(verb, help=f"record an {verb} by AMČR distribution name and SHA-512")
        io.add_argument("--state", required=True)
        io.add_argument("--name", required=True, help="the distribution name, e.g. atrium/document-json")
        io.add_argument("--file", default=None, help="the file to hash (its path is not recorded)")
        io.add_argument("--sha512", default=None)
        io.add_argument("--media-type", default=None)
        io.add_argument("--not-stored", action="store_true", help="the file is not stored (D23)")
        if verb == "output":
            io.add_argument("--record-fields", default=None, help="JSON file: {block: [fields]} written")

    pa = sub.add_parser("parameter", help="record a parameter that shaped the output")
    pa.add_argument("--state", required=True)
    pa.add_argument("--key", required=True)
    pa.add_argument("--value", required=True)

    fl = sub.add_parser("fail", help="mark the run failed")
    fl.add_argument("--state", required=True)
    fl.add_argument("--error", required=True)

    fi = sub.add_parser("finish")
    fi.add_argument("--state", required=True)
    fi.add_argument("--input-total", type=int, default=None)

    me = sub.add_parser("merge")
    me.add_argument("--paths", nargs="+", required=True)
    me.add_argument("--out", required=True)
    me.add_argument("--pipeline", default=None)

    mi = sub.add_parser("migrate")
    mi.add_argument("--path", required=True)

    args = p.parse_args()

    if args.cmd == "start":
        cfg: Dict[str, Any] = {}
        for kv in args.config or []:
            k, _, v = kv.partition("=")
            cfg[k.strip()] = v.strip()
        logger = ParadataLogger(
            program=args.program,
            config=cfg,
            paradata_dir=args.paradata_dir,
            output_types=args.output_types or None,
        )
        for name in args.component or []:
            logger.log_component(name)
        state_path = os.path.join(args.paradata_dir, f".state_{logger._run_id}_{args.program}.json")
        with open(state_path, "w", encoding="utf-8") as fh:
            json.dump(logger._to_state_dict(), fh, ensure_ascii=False)
        print(state_path)

    elif args.cmd == "merge":
        merge_run_paradata(args.paths, args.out, pipeline=args.pipeline)
        return

    elif args.cmd == "migrate":
        data = load_paradata(args.path)
        with open(args.path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
        print(f"[paradata] Migrated {args.path} to {SCHEMA_VERSION}", flush=True)
        return

    else:
        logger = ParadataLogger.from_state_file(args.state)

        if args.cmd == "skip":
            logger.log_skip(args.file, args.reason)
        elif args.cmd == "success":
            logger.log_success(args.type, args.count)
            for name in args.component or []:
                logger.log_component(name)
        elif args.cmd == "component":
            logger.log_component(args.name, args.license, args.version)
        elif args.cmd == "note-limit":
            try:
                logger.note_limit(args.limit, _cli_number(args.value), args.effect, args.count, args.detail)
            except ContentLimitError as exc:
                print(f"[paradata] {exc}", file=sys.stderr)
                sys.exit(1)
        elif args.cmd in ("input", "output"):
            kwargs: Dict[str, Any] = {
                "path": args.file,
                "sha512": args.sha512,
                "media_type": args.media_type,
                "stored": not args.not_stored,
            }
            if args.cmd == "output":
                if args.record_fields:
                    with open(args.record_fields, "r", encoding="utf-8") as fh:
                        kwargs["record_fields"] = json.load(fh)
                logger.add_output(args.name, **kwargs)
            else:
                logger.add_input(args.name, **kwargs)
        elif args.cmd == "parameter":
            logger.set_parameter(args.key, args.value)
        elif args.cmd == "fail":
            logger.fail(args.error)
        elif args.cmd == "finish":
            logger.finalize(input_total=getattr(args, "input_total", None))
            os.remove(args.state)
            return

        with open(args.state, "w", encoding="utf-8") as fh:
            json.dump(logger._to_state_dict(), fh, ensure_ascii=False)


def _cli_number(raw: Optional[str]) -> Any:
    if raw is None:
        return None
    try:
        number = float(raw)
    except ValueError:
        return raw
    return int(number) if number.is_integer() else number


if __name__ == "__main__":
    _cli()
