"""
entrypoint.py — the batch image's ENTRYPOINT (`python3 /app/entrypoint.py`).

Options go to the classifier, as before: `docker run IMAGE -d /data/input --best` runs
`python3 /app/run.py -d /data/input --best`, after a one-line report of the GPU it sees.

A command goes to the command. When the first argument is not an option and names a program
(`/bin/sh`, `python3`, `bash`), it runs as given. Galaxy starts a tool's container as
`docker run … IMAGE /bin/sh …/tool_script.sh`; with every argument forwarded to run.py, that
failed with "run.py: error: unrecognized arguments" (atrium-project#66). This is the convention of
the official images' docker-entrypoint scripts. run.py takes no positional argument, so no
classifier call changes meaning.

The process is replaced with os.execv, no shell in between, so the program is PID 1 and receives
signals directly (issue #55). Until v1.10.3-beta this file was written by the Dockerfile; it is
tracked now, so the dispatch has tests (tests/test_container_entrypoint.py).
"""

from __future__ import annotations

import os
import shutil
import sys
from typing import List, Optional, Sequence

RUN_PY = "/app/run.py"


def command_for(args: Sequence[str], python: Optional[str] = None) -> List[str]:
    """The argv the container runs for its arguments `args`.

    The first argument runs as a program when it is not an option and resolves to an executable
    (an absolute path, or a name on PATH). Anything else is the classifier's arguments.
    """
    if args and not args[0].startswith("-"):
        program = shutil.which(args[0])
        if program:
            return [program, *args[1:]]
    python = python or sys.executable
    return [python, RUN_PY, *args]


def report_gpu() -> None:
    """Informational only: CPU inference is supported (slower), so the container always starts."""
    try:
        import torch

        if torch.cuda.is_available():
            print(f"[gpu] CUDA device detected: {torch.cuda.get_device_name(0)}", flush=True)
        else:
            print("[gpu] No CUDA device visible — running on CPU (slower).", flush=True)
            print("[gpu] For GPU: build with TORCH_INDEX_URL=...cu126 and use docker-compose.gpu.yml.", flush=True)
    except Exception as exc:
        print(f"[gpu] GPU probe unavailable ({exc}) — continuing.", flush=True)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    command = command_for(args)
    if command[1:2] == [RUN_PY]:
        report_gpu()
    os.execv(command[0], command)


if __name__ == "__main__":
    main()
