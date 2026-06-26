"""
atrium_paradata.py  –  Unified provenance/paradata logger for ATRIUM pipelines.
"""

from __future__ import annotations

import configparser
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

try:
    from para_licenses import merge_effective_licenses, resolve_effective_license
except ImportError:
    resolve_effective_license = None  # type: ignore
    merge_effective_licenses = None  # type: ignore

# ──────────────────────────────────────────────────────────────────────────────
# Constants & Schema version
# ──────────────────────────────────────────────────────────────────────────────

SCHEMA_VERSION = "2.0"
LICENSE_NAME = "CC BY-NC 4.0"
LICENSE_URL = "https://creativecommons.org/licenses/by-nc/4.0/"

_REPO_URLS: Dict[str, str] = {
    "page-classification": "https://github.com/ufal/atrium-page-classification",
    "alto-postprocess": "https://github.com/ufal/atrium-alto-postprocess",
    "nlp-enrich": "https://github.com/ufal/atrium-nlp-enrich",
    "translator": "https://github.com/ufal/atrium-translator",
}

_ENV_RUNNER_IMAGE = "ATRIUM_RUNNER_IMAGE"
_ENV_RUNNER_REPO = "ATRIUM_RUNNER_REPO"
_ENV_RUNNER_REF = "ATRIUM_RUNNER_REF"


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


class ParadataLogger:
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
        self.program = program
        self.paradata_dir = paradata_dir
        self._start_dt = datetime.now(tz=timezone.utc)
        self._run_id = self._start_dt.strftime("%y%m%d-%H%M%S")

        self._para_cfg = _load_para_config(config_dir)
        self.version = version or self._para_cfg.get("version") or "unknown"
        self.docker_image = docker_image or os.environ.get(_ENV_RUNNER_IMAGE) or ""
        self.config = _sanitise(config)

        self._output_counts: Dict[str, int] = {}
        if output_types:
            for t in output_types:
                self._output_counts[t] = 0

        self._docs_processed: int = 0
        self._components_used: Dict[str, str] = {}
        for comp in self._para_cfg.get("components", []):
            if comp.get("loaded") == "always":
                self._components_used[comp["name"]] = comp["license"]

        self._skipped: List[Dict[str, str]] = []
        self._input_total: int = 0
        self._finalised: bool = False

        os.makedirs(paradata_dir, exist_ok=True)

    def log_skip(self, filepath: str, reason: str) -> None:
        self._skipped.append(
            {
                "file": str(filepath),
                "reason": str(reason),
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            }
        )

    def log_success(self, output_type: str, count: int = 1) -> None:
        self._output_counts[output_type] = self._output_counts.get(output_type, 0) + count

    def log_document_success(self) -> None:
        self._docs_processed += 1

    def log_component(self, name: str, license: Optional[str] = None) -> None:
        if license is None:
            for comp in self._para_cfg.get("components", []):
                if comp["name"] == name:
                    license = comp["license"]
                    break
        self._components_used[name] = license or "UNKNOWN"

    def _resolve_repository(self) -> str:
        return (
            os.environ.get(_ENV_RUNNER_REPO)
            or self._para_cfg.get("repository_fallback")
            or _REPO_URLS.get(self.program, "https://github.com/ufal")
        )

    def _license_block(self) -> Dict[str, Any]:
        comps = list(self._components_used.items())
        if resolve_effective_license is not None and comps:
            return resolve_effective_license(comps)
        return {
            "effective_license": LICENSE_NAME,
            "effective_license_url": LICENSE_URL,
            "is_non_commercial": True,
            "is_share_alike": False,
            "determined_by": [],
            "components": [{"name": n, "license": lic} for n, lic in comps],
            "unknown_licenses": [],
            "notes": "License helper unavailable or no components recorded; defaulted conservatively to CC BY-NC 4.0.",
        }

    def finalize(
        self,
        input_total: Optional[int] = None,
        processed_total: Optional[int] = None,
    ) -> str:
        """
        Write the paradata JSON.
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

        lic = self._license_block()

        payload = {
            "schema_version": SCHEMA_VERSION,
            "program": self.program,
            "tool_version": self.version,
            "repository": self._resolve_repository(),
            "runner_ref": os.environ.get(_ENV_RUNNER_REF, ""),
            "docker_image": self.docker_image,
            "python_version": sys.version,
            "run_id": self._run_id,
            "license": lic["effective_license"],
            "license_url": lic["effective_license_url"],
            "license_detail": lic,
            "start_time": self._start_dt.isoformat(),
            "end_time": end_dt.isoformat(),
            "duration_seconds": round(duration_sec, 3),
            "config": self.config,
            "statistics": {
                "input_files_total": input_total,
                "successfully_processed": processed_docs,
                "skipped_files": skipped_count,
                "output_counts_by_type": dict(self._output_counts),
                "performance_per_minute": perf_per_min,
            },
            "skipped_files_detail": self._skipped,
        }

        out_path = os.path.join(self.paradata_dir, f"{self._run_id}_{self.program}.json")
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)

        self._finalised = True
        print(f"[paradata] Log written → {out_path}", flush=True)
        return out_path

    def __enter__(self) -> "ParadataLogger":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
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
            "skipped": self._skipped,
            "docs_processed": self._docs_processed,
            "start_iso": self._start_dt.isoformat(),
            "run_id": self._run_id,
            "para_cfg": self._para_cfg,
        }

    @classmethod
    def _from_state_dict(cls, d: Dict[str, Any]) -> "ParadataLogger":
        inst = cls.__new__(cls)
        inst.program = d["program"]
        inst.version = d.get("version", "unknown")
        inst.config = d["config"]
        inst.paradata_dir = d["paradata_dir"]
        inst._output_counts = d["output_counts"]
        inst._components_used = d.get("components_used", {})
        inst._skipped = d["skipped"]
        inst._docs_processed = d.get("docs_processed", 0)
        inst._run_id = d["run_id"]
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


def migrate_paradata(record: Dict[str, Any]) -> Dict[str, Any]:
    """Applies schema migrations up to the current SCHEMA_VERSION."""
    v = record.get("schema_version")
    if not v or v.startswith("1."):
        record = _migrate_1_0_to_2_0(record)
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


def merge_paradata_files(json_paths: List[str], input_file: str, out_path: str) -> str:
    steps: List[Dict[str, Any]] = []
    license_blocks: List[Dict[str, Any]] = []
    total_duration = 0.0

    for p in json_paths:
        data = load_paradata(p)
        steps.append(
            {
                "program": data.get("program"),
                "tool_version": data.get("tool_version"),
                "repository": data.get("repository"),
                "docker_image": data.get("docker_image"),
                "run_id": data.get("run_id"),
                "duration_seconds": data.get("duration_seconds"),
                "license": data.get("license"),
                "config": data.get("config"),
            }
        )
        if data.get("license_detail"):
            license_blocks.append(data["license_detail"])
        total_duration += float(data.get("duration_seconds") or 0.0)

    if merge_effective_licenses is not None and license_blocks:
        merged_lic = merge_effective_licenses(license_blocks)
    else:
        merged_lic = {
            "effective_license": LICENSE_NAME,
            "effective_license_url": LICENSE_URL,
            "notes": "License helper unavailable; defaulted to CC BY-NC 4.0.",
        }

    payload = {
        "schema_version": SCHEMA_VERSION,
        "record_type": "single-file-merged",
        "input_file": input_file,
        "pipeline_steps": steps,
        "step_count": len(steps),
        "total_duration_seconds": round(total_duration, 3),
        "license": merged_lic["effective_license"],
        "license_url": merged_lic["effective_license_url"],
        "license_detail": merged_lic,
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
    license_blocks: List[Dict[str, Any]] = []
    formats: Dict[str, int] = {}
    total_duration = 0.0
    total_inputs = 0
    total_processed = 0
    total_skipped = 0
    all_skips: List[Dict[str, Any]] = []
    repo = ""
    tool_version = ""
    earliest: Optional[str] = None
    latest: Optional[str] = None
    first_stage = True

    for order, p in enumerate(json_paths, 1):
        data = load_paradata(p)

        repo = repo or data.get("repository", "")
        tool_version = tool_version or data.get("tool_version", "")

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
                "input_dir": cfg.get("input_dir"),
                "input_csv": cfg.get("input_csv"),
                "output_dir": cfg.get("output_dir") or cfg.get("output_csv") or cfg.get("output_manifest"),
                "output_formats": out_counts,
                "duration_seconds": data.get("duration_seconds"),
                "license": data.get("license"),
                "input_files_total": stats.get("input_files_total"),
                "successfully_processed": stats.get("successfully_processed"),
                "skipped_files": stats.get("skipped_files"),
            }
        )

        if data.get("license_detail"):
            license_blocks.append(data["license_detail"])

    if merge_effective_licenses is not None and license_blocks:
        merged_lic = merge_effective_licenses(license_blocks)
    else:
        merged_lic = {
            "effective_license": LICENSE_NAME,
            "effective_license_url": LICENSE_URL,
            "notes": "License helper unavailable; defaulted to CC BY-NC 4.0.",
        }

    payload = {
        "schema_version": SCHEMA_VERSION,
        "record_type": "pipeline-run-merged",
        "pipeline": pipeline or (stages[0].get("program") if stages else "unknown"),
        "method": method or "",
        "repository": repo,
        "tool_version": tool_version,
        "runner_ref": os.environ.get(_ENV_RUNNER_REF, ""),
        "request_id": os.environ.get("ATRIUM_REQUEST_ID", ""),
        "python_version": sys.version,
        "run_id": datetime.now(tz=timezone.utc).strftime("%y%m%d-%H%M%S"),
        "stage_count": len(stages),
        "pipeline_stages": stages,
        "intermediate_formats": formats,
        "license": merged_lic["effective_license"],
        "license_url": merged_lic["effective_license_url"],
        "license_detail": merged_lic,
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
        "skipped_stages": skipped_stages or [],
        "license_note": (
            "Effective license/intermediate_formats reflect EXECUTED stages only; skipped: " + ", ".join(skipped_stages)
        )
        if skipped_stages
        else "",
        "merged_at": datetime.now(tz=timezone.utc).isoformat(),
    }

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    print(f"[paradata] Merged pipeline-run log → {out_path}", flush=True)
    return out_path


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

    elif args.cmd in ("skip", "success", "component", "finish"):
        with open(args.state, "r", encoding="utf-8") as fh:
            state_dict = json.load(fh)
        logger = ParadataLogger._from_state_dict(state_dict)

        if args.cmd == "skip":
            logger.log_skip(args.file, args.reason)
        elif args.cmd == "success":
            logger.log_success(args.type, args.count)
            for name in args.component or []:
                logger.log_component(name)
        elif args.cmd == "component":
            logger.log_component(args.name, args.license)
        elif args.cmd == "finish":
            logger.finalize(input_total=getattr(args, "input_total", None))
            os.remove(args.state)
            return

        with open(args.state, "w", encoding="utf-8") as fh:
            json.dump(logger._to_state_dict(), fh, ensure_ascii=False)


if __name__ == "__main__":
    _cli()
