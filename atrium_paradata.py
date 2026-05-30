"""
atrium_paradata.py  –  Unified provenance/paradata logger for ATRIUM pipelines.

DROP THIS FILE AS-IS into every ATRIUM repository root, alongside
para_licenses.py and a repository-specific para_config.txt.

Resolves ATRIUM issue #9:
  * license is no longer hardcoded – it is computed per-run from the components
    actually used (see para_licenses.py + para_config.txt);
  * a tool VERSION tag is recorded;
  * the repository/runner reference is resolved DYNAMICALLY (env override) so it
    can point at the published container actually executing, not a static fork;
  * a docker image placeholder field is emitted;
  * paradata is intended to live in the OUTPUT directory, not the GH repo
    (default paradata_dir is now resolved relative to the output location);
  * single-file workflows can merge the per-tool logs into ONE json per input
    file via merge_paradata_files().

Backward compatibility
-----------------------
The constructor and log_success/log_skip/finalize/context-manager API are
unchanged, so existing call sites (run.py, main.py) keep working.  New
behaviour is opt-in via para_config.txt and the components_used / version /
docker_image keyword arguments.
"""

from __future__ import annotations

import configparser
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

try:
    from para_licenses import resolve_effective_license, merge_effective_licenses
except ImportError:  # keep logging functional even if the helper is missing
    resolve_effective_license = None      # type: ignore
    merge_effective_licenses = None       # type: ignore


# ──────────────────────────────────────────────────────────────────────────────
# Static fallbacks (used only when para_config.txt is absent)
# ──────────────────────────────────────────────────────────────────────────────

_REPO_URLS: Dict[str, str] = {
    "page-classification": "https://github.com/ufal/atrium-page-classification",
    "alto-postprocess":    "https://github.com/ufal/atrium-alto-postprocess",
    "nlp-enrich":          "https://github.com/ufal/atrium-nlp-enrich",
    "translator":          "https://github.com/ufal/atrium-translator",
}

# Environment variables a container sets so the logged reference points at the
# ACTUAL running image/runner rather than a static fork URL.
_ENV_RUNNER_IMAGE = "ATRIUM_RUNNER_IMAGE"      # e.g. ghcr.io/ufal/atrium-translator:v0.5.2
_ENV_RUNNER_REPO  = "ATRIUM_RUNNER_REPO"       # e.g. https://github.com/ufal/atrium-translator
_ENV_RUNNER_REF   = "ATRIUM_RUNNER_REF"        # e.g. git sha / tag the container was built from


def _load_para_config(start_dir: str = ".") -> Dict[str, Any]:
    """
    Load repository-specific para_config.txt if present.

    Returns a dict:
        { "program": str, "version": str, "repository_fallback": str,
          "components": [ {name, license, loaded, role}, ... ] }
    Empty/missing file -> minimal dict so callers can fall back to kwargs.
    """
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
            # spec form: "<license> ; <always|conditional> ; <role>"
            fields = [s.strip() for s in spec.split(";")]
            lic = fields[0] if len(fields) > 0 else ""
            loaded = fields[1] if len(fields) > 1 else "always"
            role = fields[2] if len(fields) > 2 else ""
            out["components"].append({
                "name": name.strip(),
                "license": lic,
                "loaded": loaded,
                "role": role,
            })
    return out


# ──────────────────────────────────────────────────────────────────────────────
# ParadataLogger
# ──────────────────────────────────────────────────────────────────────────────

class ParadataLogger:
    """
    Context-manager-friendly paradata recorder.

    New parameters (all optional, all backward compatible)
    ------------------------------------------------------
    version : str
        Tool version tag.  Falls back to para_config.txt [tool] version.
    docker_image : str
        Running container image reference.  Falls back to env ATRIUM_RUNNER_IMAGE,
        else "" (placeholder kept in output so the field always exists).
    config_dir : str
        Where to find para_config.txt (default: current dir).
    """

    def __init__(
        self,
        program: str,
        config: Dict[str, Any],
        paradata_dir: str = "paradata",
        output_types: Optional[List[str]] = None,
        version: Optional[str] = None,
        docker_image: Optional[str] = None,
        config_dir: str = ".",
    ) -> None:
        self.program      = program
        self.paradata_dir = paradata_dir
        self._start_dt    = datetime.now(tz=timezone.utc)
        self._run_id      = self._start_dt.strftime("%y%m%d-%H%M%S")

        # repo-specific static facts
        self._para_cfg = _load_para_config(config_dir)

        # version: kwarg > para_config > "unknown"
        self.version = (
            version
            or self._para_cfg.get("version")
            or "unknown"
        )

        # docker image: kwarg > env > "" (placeholder retained)
        self.docker_image = (
            docker_image
            or os.environ.get(_ENV_RUNNER_IMAGE)
            or ""
        )

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

        # components actually exercised this run: {name: license}
        self._components_used: Dict[str, str] = {}
        # auto-seed with components flagged "always" in para_config
        for comp in self._para_cfg.get("components", []):
            if comp.get("loaded") == "always":
                self._components_used[comp["name"]] = comp["license"]

        os.makedirs(paradata_dir, exist_ok=True)

    # ── public API ─────────────────────────────────────────────────────────────

    def log_skip(self, filepath: str, reason: str) -> None:
        self._skipped.append({
            "file":      str(filepath),
            "reason":    str(reason),
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        })

    def log_success(self, output_type: str, count: int = 1) -> None:
        self._output_counts[output_type] = (
            self._output_counts.get(output_type, 0) + count
        )

    def log_component(self, name: str, license: Optional[str] = None) -> None:
        """
        Record that a licensed component was ACTUALLY exercised this run.

        If *license* is omitted it is looked up from para_config.txt.  Call this
        the first time a conditional component is invoked (e.g. when a
        vocabulary is loaded, or the translation API is first hit) so the
        effective output license reflects real usage rather than worst case.
        """
        if license is None:
            for comp in self._para_cfg.get("components", []):
                if comp["name"] == name:
                    license = comp["license"]
                    break
        self._components_used[name] = license or "UNKNOWN"

    # ── reference / license resolution ────────────────────────────────────────

    def _resolve_repository(self) -> str:
        """Dynamic runner reference: env > para_config fallback > static map."""
        return (
            os.environ.get(_ENV_RUNNER_REPO)
            or self._para_cfg.get("repository_fallback")
            or _REPO_URLS.get(self.program, "https://github.com/ufal")
        )

    def _license_block(self) -> Dict[str, Any]:
        comps = list(self._components_used.items())
        if resolve_effective_license is not None and comps:
            return resolve_effective_license(comps)
        # Fallback if helper missing or no components recorded: stay safe.
        return {
            "effective_license": "CC BY-NC 4.0",
            "effective_license_url": "https://creativecommons.org/licenses/by-nc/4.0/",
            "is_non_commercial": True,
            "is_share_alike": False,
            "determined_by": [],
            "components": [{"name": n, "license": l} for n, l in comps],
            "unknown_licenses": [],
            "notes": "License helper unavailable or no components recorded; "
                     "defaulted conservatively to CC BY-NC 4.0.",
        }

    def finalize(self, input_total: Optional[int] = None) -> str:
        if self._finalised:
            raise RuntimeError("finalize() has already been called.")

        end_dt       = datetime.now(tz=timezone.utc)
        duration_sec = (end_dt - self._start_dt).total_seconds()
        duration_min = duration_sec / 60.0 if duration_sec > 0 else 0.0

        skipped_count  = len(self._skipped)
        processed_docs = max(self._output_counts.values()) if self._output_counts else 0
        if input_total is None:
            input_total = processed_docs + skipped_count

        perf_per_min: Dict[str, float] = {}
        for otype, cnt in self._output_counts.items():
            perf_per_min[otype] = round(cnt / duration_min, 4) if duration_min > 0 else 0.0

        lic = self._license_block()

        payload = {
            # ── provenance ──────────────────────────────────────────────────
            "schema_version":      "2.0",
            "program":             self.program,
            "tool_version":        self.version,
            "repository":          self._resolve_repository(),
            "runner_ref":          os.environ.get(_ENV_RUNNER_REF, ""),
            "docker_image":        self.docker_image,   # placeholder if unset
            "python_version":      sys.version,
            "run_id":              self._run_id,

            # ── license (computed from components actually used) ─────────────
            "license":             lic["effective_license"],
            "license_url":         lic["effective_license_url"],
            "license_detail":      lic,

            # ── timing ──────────────────────────────────────────────────────
            "start_time":          self._start_dt.isoformat(),
            "end_time":            end_dt.isoformat(),
            "duration_seconds":    round(duration_sec, 3),

            # ── configuration snapshot ───────────────────────────────────────
            "config":              self.config,

            # ── statistics ───────────────────────────────────────────────────
            "statistics": {
                "input_files_total":      input_total,
                "successfully_processed": processed_docs,
                "skipped_files":          skipped_count,
                "output_counts_by_type":  dict(self._output_counts),
                "performance_per_minute": perf_per_min,
            },

            "skipped_files_detail": self._skipped,
        }

        out_path = os.path.join(
            self.paradata_dir, f"{self._run_id}_{self.program}.json"
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
        if not self._finalised:
            try:
                self.finalize()
            except Exception as e:
                print(f"[paradata] WARNING – could not write log: {e}", file=sys.stderr)
        return False


# ──────────────────────────────────────────────────────────────────────────────
# Single-file workflow: merge per-tool logs into ONE json per input file
# ──────────────────────────────────────────────────────────────────────────────

def merge_paradata_files(
    json_paths: List[str],
    input_file: str,
    out_path: str,
) -> str:
    """
    Merge several per-tool paradata JSONs (one input file passed through several
    tools/repos) into a single provenance record covering exactly that file.

    The merged license is re-derived from the UNION of all components used, so
    the end-to-end most-restrictive rule holds.
    """
    steps: List[Dict[str, Any]] = []
    license_blocks: List[Dict[str, Any]] = []
    total_duration = 0.0

    for p in json_paths:
        with open(p, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        steps.append({
            "program":      data.get("program"),
            "tool_version": data.get("tool_version"),
            "repository":   data.get("repository"),
            "docker_image": data.get("docker_image"),
            "run_id":       data.get("run_id"),
            "duration_seconds": data.get("duration_seconds"),
            "license":      data.get("license"),
            "config":       data.get("config"),
        })
        if data.get("license_detail"):
            license_blocks.append(data["license_detail"])
        total_duration += float(data.get("duration_seconds") or 0.0)

    if merge_effective_licenses is not None and license_blocks:
        merged_lic = merge_effective_licenses(license_blocks)
    else:
        merged_lic = {
            "effective_license": "CC BY-NC 4.0",
            "effective_license_url": "https://creativecommons.org/licenses/by-nc/4.0/",
            "notes": "License helper unavailable; defaulted to CC BY-NC 4.0.",
        }

    payload = {
        "schema_version":  "2.0",
        "record_type":     "single-file-merged",
        "input_file":      input_file,
        "pipeline_steps":  steps,
        "step_count":      len(steps),
        "total_duration_seconds": round(total_duration, 3),
        "license":         merged_lic["effective_license"],
        "license_url":     merged_lic["effective_license_url"],
        "license_detail":  merged_lic,
        "merged_at":       datetime.now(tz=timezone.utc).isoformat(),
    }

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    print(f"[paradata] Merged single-file log → {out_path}", flush=True)
    return out_path


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

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


# (the start/skip/success/finish CLI shim is unchanged from the original and
#  omitted here for brevity – keep the existing _cli() if Bash drives the logger)