#!/usr/bin/env python3
"""Verify that a tool repo's declared version agrees everywhere.

Canonical source: ``ufal/atrium-project`` → ``docs/templates/shared/check_version.py``.
Vendored byte-identically into every tool repo root and held there by
``para-drift.reusable.yml`` — edit the hub copy, never the vendored one.

Why this file exists
--------------------
The same ~45-line ``python - <<'PY'`` heredoc was pasted into five ``release.yml``
files and again, in a subtly different three-way variant, into
``security.reusable.yml`` — roughly 360 lines of copy-paste that had already
drifted apart (issue #18, 2026-07-29 audit).

It stays a *vendored script* rather than a reusable workflow on purpose: the
release gate must not depend on a cross-repo ``uses: …@ref`` resolving at
tag-push time. That ordering gap is what let ``v1.0.0.-beta`` and ``v1.16.2``
publish unchecked. A file in the repo is always there.

Checks
------
* ``CITATION.cff`` top-level ``version:`` parses.
* ``para_config.txt`` ``[tool] version`` parses.
* The two agree.
* If a tag is supplied, it agrees with both.

A leading ``v`` is tolerated and stripped everywhere, so ``v1.2.3`` in
``para_config.txt`` matches ``1.2.3`` in ``CITATION.cff``.

Usage
-----
    # release.yml — a tag is mandatory
    python check_version.py --citation CITATION.cff \
        --para-config setup/para_config.txt \
        --tag "${GITHUB_REF_NAME}" --require-tag

    # security.reusable.yml — consistency check, tag optional
    python check_version.py --citation CITATION.cff --para-config para_config.txt

Exit status is 0 when everything agrees, 1 otherwise. Failures are emitted as
``::error::`` workflow commands so they surface on the run's summary.
"""

from __future__ import annotations

import argparse
import configparser
import re
import sys


def _strip_v(value: str | None) -> str | None:
    """Drop a single leading ``v``/``V`` so v1.2.3 and 1.2.3 compare equal."""
    if value and value[:1] in ("v", "V"):
        return value[1:]
    return value


def citation_version(path: str) -> str | None:
    """First column-0 ``version:`` in a CITATION.cff.

    Anchored at column 0 deliberately: CFF nests ``version:`` under
    ``references:``/``preferred-citation:`` for cited works, and those are not
    this tool's version.
    """
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            match = re.match(r'version:\s*["\']?v?([^"\'\s]+)', line)
            if match:
                return match.group(1)
    return None


def paraconfig_version(path: str) -> str | None:
    """``[tool] version`` from a para_config.txt (INI)."""
    parser = configparser.ConfigParser()
    parser.read(path, encoding="utf-8")
    return _strip_v(parser.get("tool", "version", fallback=None))


def collect_errors(citation_path: str, para_path: str, tag: str, require_tag: bool) -> list[str]:
    """Return every disagreement found. Empty list means consistent."""
    errors: list[str] = []

    try:
        cit = citation_version(citation_path)
    except OSError as exc:
        return [f"could not read {citation_path}: {exc}"]
    try:
        para = paraconfig_version(para_path)
    except (OSError, configparser.Error) as exc:
        return [f"could not read {para_path}: {exc}"]

    tag = _strip_v(tag.strip()) or ""
    print(f"CITATION.cff={cit!r}  para_config={para!r}  tag={tag!r}")

    if not cit:
        errors.append(f"could not parse version from {citation_path}")
    if not para:
        errors.append(f"could not parse [tool] version from {para_path}")
    if require_tag and not tag:
        errors.append("no tag ref on this event, but --require-tag was given")

    if cit and para and cit != para:
        errors.append(f"version mismatch: {citation_path}={cit} != {para_path}={para}")
    if tag and cit and tag != cit:
        errors.append(f"tag v{tag} != {citation_path} {cit}")
    if tag and para and tag != para:
        errors.append(f"tag v{tag} != {para_path} {para}")

    return errors


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--citation", default="CITATION.cff", help="path to CITATION.cff")
    ap.add_argument(
        "--para-config",
        default="para_config.txt",
        help="path to para_config.txt; repos with a setup/ layout pass setup/para_config.txt",
    )
    ap.add_argument("--tag", default="", help="tag being released, e.g. v1.2.3; empty to skip tag comparison")
    ap.add_argument("--require-tag", action="store_true", help="fail when --tag is empty (release gate)")
    args = ap.parse_args(argv)

    errors = collect_errors(args.citation, args.para_config, args.tag, args.require_tag)
    for err in errors:
        print(f"::error::{err}")
    if errors:
        print(
            "::error::Version check failed — make the tag, CITATION.cff and "
            "para_config.txt [tool] version identical, then re-tag."
        )
        return 1
    print("OK — versions agree")
    return 0


if __name__ == "__main__":
    sys.exit(main())
