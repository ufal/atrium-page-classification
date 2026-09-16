"""
tests/test_dockerfile_security_layer.py — the release gate must stay passable.

CANONICAL FILE. Source of truth: ufal/atrium-project ->
``docs/templates/shared/test_dockerfile_security_layer.py``. Vendored
byte-identically into every tool repo's ``tests/`` and held there by
``para-drift.reusable.yml`` — edit the hub copy, never the vendored one.

``ufal/atrium-project``'s ``docker-tool.reusable.yml`` blocks a release on
*fixable* CRITICAL vulnerabilities in the published image and, because the
tag-promotion step is ``if: success()``, a failure means the image is published
by digest with no ``:<version>`` or ``:latest`` tag on it.

That has now happened twice, in two repos, for the same three CVEs:

* translator ``v1.0.0-beta``, 2026-09-13 — both matrix targets.
* nlp-enrich ``v0.20.2``, 2026-09-15 (run 34970419474) — all three matrix
  targets.

Both times ``python:3.11-slim`` carried perl-base 5.40.1-6 with three fixable
CRITICALs (CVE-2026-13221, CVE-2026-42496, CVE-2026-8376), fixed upstream in
5.40.1-6+deb13u1. The base image is a *floating tag* that nothing in this
ecosystem bumps, so the only defence inside a repo is to apply the distro's
available patches at build time.

Each Dockerfile now does that. These tests pin the properties that make it work,
because every one of them is invisible on inspection and degrades silently
rather than breaking:

1. **The layer exists at all.** Without it the image ships whatever Docker Hub
   last built, and the next release is blocked rather than the next build.
2. **Position relative to the cache-bust anchor.** The build runs with
   ``cache-from: type=gha``. An apt layer placed above the ``ENV`` block that
   embeds ``ATRIUM_RUNNER_REF`` would be served from cache indefinitely and stop
   patching anything, while still *looking* correct in the file. CI passes that
   ARG as ``github.ref_name``, unique per release tag, so sitting below it is
   what makes the layer re-run for every released image.
3. **Position relative to the non-root switch.** apt needs root; below
   ``USER atrium`` the layer fails the build outright.
4. **The base is still Debian.** ``apt-get upgrade`` is only the right mechanism
   while the base image is Debian-derived.

This file deliberately has no imports beyond the standard library and no
repo-local helper import, so that it is byte-identical in all five tool repos
(``preconditions: []`` in ``docs/templates/shared/MANIFEST.json``).

It is also a ruff-format FIXED POINT at line-length 100 *and* 120 -- the two values
in use across this ecosystem (100 in nlp-enrich and llm-enrich, 120 in the other
three). ``docs/templates/ruff.toml``'s ``[format] exclude`` already shields the
canonical files from being reflowed by whichever repo committed last, but that
protection is a config entry someone can omit; being stable at both widths means an
omission cannot split this file into two variants the way it split
``tests/test_document_originators.py`` on 2026-08-05. If you edit a message here,
re-check with::

    ruff format --line-length 100 --diff tests/test_dockerfile_security_layer.py
    ruff format --line-length 120 --diff tests/test_dockerfile_security_layer.py

Both must report no change.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

# Inline rather than `from atrium_test_support import REPO_ROOT`: that helper
# exists in exactly one of the five tool repos, and this file has to be
# byte-identical in all of them to be para-drift-guarded.
REPO_ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE_PATH = REPO_ROOT / "Dockerfile"

# In a tool repo this file sits at tests/, so REPO_ROOT is the repo root and the
# Dockerfile is beside it. In the hub the canonical copy sits at
# docs/templates/shared/, so REPO_ROOT is docs/templates/ -- which is why
# `docs/templates/Dockerfile` exists: the hub's reference Dockerfile template is
# then held to the same five assertions as the five real ones, by the same file.
#
# The skip is the safety net for every OTHER context. Reading the Dockerfile at
# import time with no guard turns a missing file into a COLLECTION ERROR, which
# does not fail one test -- it aborts the whole run. That is not hypothetical:
# this file shipped without the guard on 2026-09-16 and broke `pytest` at the hub
# root and `hub-self-check.yml`'s "Run canonical shared-module tests" step, which
# runs `pytest .` with working-directory docs/templates/shared. Every other
# canonical test that inspects a consuming repo's tree (test_logging_contract.py,
# test_env_contract.py) already skips at module level for exactly this reason.
# An honest skip beats an aborted run.
if not DOCKERFILE_PATH.is_file():
    pytest.skip(
        f"no Dockerfile at {DOCKERFILE_PATH} -- this file inspects a TOOL REPO's "
        "Dockerfile, or the hub's docs/templates/Dockerfile when run from "
        "docs/templates/shared/ (atrium-project#53)",
        allow_module_level=True,
    )

DOCKERFILE = DOCKERFILE_PATH.read_text()
LINES = DOCKERFILE.splitlines()


def _line_of(pattern: str) -> int:
    """Index of the first line matching `pattern`, or -1. Comments count as text.

    Every caller below searches for a directive that only ever appears as a
    directive (`RUN apt-get upgrade`, `USER atrium`), so a mention inside a
    comment is not a realistic false positive here.
    """
    rx = re.compile(pattern)
    for i, line in enumerate(LINES):
        if rx.search(line):
            return i
    return -1


def _upgrade_line() -> int:
    """The `apt-get upgrade` RUN directive, ignoring any comment that names it."""
    rx = re.compile(r"apt-get\s+upgrade")
    for i, line in enumerate(LINES):
        if rx.search(line) and not line.lstrip().startswith("#"):
            return i
    return -1


def test_security_patches_are_applied():
    """Without this layer the base image ships whatever Docker Hub last built."""
    assert _upgrade_line() != -1, (
        "the Dockerfile applies no distro security patches, so the released image "
        "carries every fixable CVE in the python:3.11-slim base layer and the "
        "release gate blocks tag promotion (the image publishes by digest with no "
        ":<version> or :latest tag)"
    )


def test_apt_lists_are_cleaned_up():
    assert _line_of(r"rm -rf /var/lib/apt/lists") != -1, "apt lists left in the image layer"


def test_upgrade_runs_below_the_cache_bust_anchor():
    """Above the ref-bearing ENV block this layer would be cached forever."""
    env_ref = _line_of(r"ATRIUM_RUNNER_REF=\$\{ATRIUM_RUNNER_REF\}")
    upgrade = _upgrade_line()
    assert env_ref != -1, "the ENV block that embeds ATRIUM_RUNNER_REF is gone; the anchor moved"
    assert upgrade > env_ref, (
        f"apt-get upgrade is at line {upgrade + 1}, above the ATRIUM_RUNNER_REF ENV at line "
        f"{env_ref + 1}. With cache-from: type=gha that layer is served from cache on every "
        "later build and silently stops patching."
    )


def test_upgrade_runs_as_root():
    """apt needs root; below USER atrium the build fails."""
    upgrade = _upgrade_line()
    user_switch = _line_of(r"^USER atrium")
    assert user_switch != -1, "the non-root USER switch is gone — that is its own problem"
    assert upgrade < user_switch, (
        f"apt-get upgrade at line {upgrade + 1} runs after USER atrium at line "
        f"{user_switch + 1}; apt requires root, so the build would fail outright here"
    )


def test_base_image_is_still_a_known_distro():
    """If the base moves off Debian, `apt-get upgrade` stops being the right fix."""
    assert re.search(r"^FROM\s+python:3\.11-slim", DOCKERFILE, re.MULTILINE), (
        "base image changed; re-check that apt-get upgrade is still the correct "
        "mechanism for applying this distribution's available security patches"
    )
