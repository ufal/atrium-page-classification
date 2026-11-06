#!/usr/bin/env python3
"""
container_entrypoint.py — the ENTRYPOINT of a tool's batch image (atrium-project#66).

Canonical source: ``ufal/atrium-project`` → ``docs/templates/shared/container_entrypoint.py``.
Vendored byte-identically into every tool repo root and held there by
``para-drift.reusable.yml`` — edit the hub copy, never the vendored one.

The Dockerfile names the program the image runs, by its absolute path::

    ENTRYPOINT ["python", "/app/container_entrypoint.py", "/app/run_pipeline.py"]

Everything after the program is what ``docker run IMAGE …`` passed:

* **Arguments go to the program**, as with the old ``ENTRYPOINT ["python", "run_pipeline.py"]``:
  ``docker run IMAGE --config c.txt`` runs ``python /app/run_pipeline.py --config c.txt``.
* **An interpreter runs as given.** When the first argument names a shell or Python (``/bin/sh``,
  ``bash``, ``python3``), that command runs instead: ``docker run IMAGE /bin/sh job.sh``. This is
  how Galaxy starts a tool's container, in the job's own working directory; with every argument
  forwarded, ``/bin/sh`` became the program's input and the job failed with exit 2.

Only the interpreters in ``INTERPRETERS`` run. Several programs take a positional input (a file
to convert or translate), and an input file that happens to be executable, common on files copied
from a Windows share, must still reach the program rather than be executed. The program is named
by its absolute path because ``python run_pipeline.py`` resolved it against the working directory,
which under Galaxy is not ``/app``. The working directory itself is left alone, so a program that
needs ``/app`` still gets it from the image's ``WORKDIR`` when ``docker run`` sets none.

The process is replaced with ``os.execv``, no shell in between, so the program is PID 1 and
receives signals directly. ``python3 container_entrypoint.py --selftest`` checks the dispatch;
para-drift runs it on every vendored copy.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from typing import List, Optional, Sequence

#: The programs a first argument may name to run instead of the image's program.
INTERPRETERS = frozenset({"sh", "bash", "dash", "python", "python3"})

USAGE = "usage: container_entrypoint.py /ABSOLUTE/PATH/TO/PROGRAM.py [ARGUMENT ...]  (or --selftest)"


class EntrypointError(Exception):
    """The image's ENTRYPOINT names no usable program: a fault of the Dockerfile, not of the caller."""


def check_program(program: str) -> str:
    """The program the Dockerfile names: an absolute path to an existing file."""
    if not os.path.isabs(program):
        raise EntrypointError(f"the program must be an absolute path, got {program!r}")
    if not os.path.isfile(program):
        raise EntrypointError(f"the program {program} does not exist in this image")
    return program


def command_for(program: str, args: Sequence[str], python: Optional[str] = None) -> List[str]:
    """The argv the container runs for the program and the arguments ``docker run`` passed.

    The first argument runs as a command when its base name is one of ``INTERPRETERS`` and it
    resolves to an executable (an absolute path, or a name on PATH). Anything else is the
    program's arguments.
    """
    if args and os.path.basename(args[0]) in INTERPRETERS:
        resolved = shutil.which(args[0])
        if resolved:
            return [resolved, *args[1:]]
    return [python or sys.executable, program, *args]


def selftest() -> int:
    """Check the dispatch without executing anything; 0 when every case holds."""
    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        program = os.path.join(tmp, "tool.py")
        with open(program, "w", encoding="utf-8") as handle:
            handle.write("")
        executable_input = os.path.join(tmp, "page.alto.xml")
        with open(executable_input, "w", encoding="utf-8") as handle:
            handle.write("<alto/>")
        os.chmod(executable_input, 0o755)
        folder_named_sh = os.path.join(tmp, "sh")
        os.mkdir(folder_named_sh)

        cases = [
            ("options", ["--config", "c.txt"], ["py", program, "--config", "c.txt"]),
            ("no arguments", [], ["py", program]),
            ("an executable input", [executable_input, "-o", "out"], ["py", program, executable_input, "-o", "out"]),
            ("a folder named sh", [folder_named_sh], ["py", program, folder_named_sh]),
        ]
        shell = shutil.which("sh")
        if shell:
            cases.append(("a job script", [shell, "job.sh"], [shell, "job.sh"]))
        for label, args, expected in cases:
            got = command_for(check_program(program), args, python="py")
            if got != expected:
                failures.append(f"{label}: {got} != {expected}")
        for bad in ("tool.py", os.path.join(tmp, "missing.py")):
            try:
                check_program(bad)
                failures.append(f"accepted the program {bad!r}")
            except EntrypointError:
                pass
    for failure in failures:
        print(f"container_entrypoint selftest FAILED: {failure}", file=sys.stderr)
    if not failures:
        print("container_entrypoint selftest OK")
    return 1 if failures else 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args[:1] == ["--selftest"]:
        return selftest()
    if not args:
        print(USAGE, file=sys.stderr)
        return 2
    program, rest = args[0], args[1:]
    try:
        check_program(program)
    except EntrypointError as exc:
        print(f"container_entrypoint.py: {exc}", file=sys.stderr)
        return 2
    command = command_for(program, rest)
    os.execv(command[0], command)
    return 0  # not reached: execv replaces the process


if __name__ == "__main__":
    sys.exit(main())
